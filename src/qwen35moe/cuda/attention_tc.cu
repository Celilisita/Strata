#include "strata/qwen35moe/attention_tc.hpp"
#include "attention_register.hpp"
#include "attention_rows32.hpp"
#include "attention_rows64.hpp"
#include "attention_rows64_early.hpp"
#include "attention_rows64_early_qpad8.hpp"
#include "attention_rows64_qpad8.hpp"
#include "attention_rows64_subgroup4.hpp"
#include "attention_rows64_fast_exp.hpp"
#include "attention_rows64_warps8.hpp"
#include "attention_rows32_direct.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <array>
#include <limits>
#include <utility>

namespace strata::qwen35moe {
namespace {
namespace wm = nvcuda::wmma;
bool register_output_enabled() {
    const char* flag=std::getenv("STRATA_QWEN35_ATTN_REGISTER_OUTPUT");
    if(!flag || std::strcmp(flag,"0")==0) return false;
    if(std::strcmp(flag,"1")!=0) throw std::invalid_argument("ATTN_REGISTER_OUTPUT must be0 or1");
    return true;
}
constexpr int Rows=16, Keys=64, Dim=256, Heads=16, CacheStride=512;
__device__ float maximum(float x) {
    for(int d=16; d; d>>=1) x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,d));
    return x;
}
__device__ float total(float x) {
    for(int d=16; d; d>>=1) x+=__shfl_xor_sync(0xffffffffu,x,d);
    return x;
}
__global__ void zero_tail(half* k, half* v, int width, int cells) {
    int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<cells) { k[size_t(width)*CacheStride+i]=__float2half_rn(0); v[size_t(width)*CacheStride+i]=__float2half_rn(0); }
}
template<bool Compact, bool Reuse=false>
__global__ __launch_bounds__(128,(Compact || Reuse) ? 3 : 2) void prefill(const float* q, const half* k, const half* v,
    int first, int count, float* out) {
#if __CUDA_ARCH__ >= 700
    const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const int head=blockIdx.x, kv=head/8, qbase=blockIdx.y*Rows;
    const int active=min(Rows,count-qbase), width=first+qbase+active;
    union QueryPhase { uint16_t cells[Rows*Dim]; float rows[Rows]; };
    union TilePhase {
        struct { float scores[Rows*Keys]; uint16_t probability[Rows*Keys]; } softmax;
        float product[Rows*(Reuse ? 128 : 64)];
    };
    __shared__ __align__(32) QueryPhase query_phase;
    __shared__ __align__(32) TilePhase tile;
    __shared__ __align__(32) float running[Rows*Dim];
    auto* query=reinterpret_cast<half*>(query_phase.cells);
    auto* scores=tile.softmax.scores;
    auto* probability=reinterpret_cast<half*>(tile.softmax.probability);
    float* stats=nullptr;
    if constexpr (!Reuse) {
        __shared__ float shared_stats[3*Rows];
        stats=shared_stats;
    }
    float previous_max[4],previous_sum[4];
    if constexpr (Reuse) {
#pragma unroll
        for(int own=0;own<4;++own) { previous_max[own]=-CUDART_INF_F; previous_sum[own]=0; }
    }
    for(int i=tid;i<Rows*Dim;i+=128) {
        const int row=i/Dim, d=i%Dim;
        const float value=row<active?q[(size_t(qbase+row)*Heads+head)*Dim+d]:0;
        query[i]=__hmul(__float2half_rn(value*0.0625f),__float2half_rn(0.25f));
        running[i]=0;
    }
    if constexpr (!Reuse) {
        if(tid<Rows) { stats[tid]=-CUDART_INF_F; stats[Rows+tid]=0; }
    }
    __syncthreads();
    for(int key=0;key<width;key+=Keys) {
        {
            wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major> a;
            wm::fragment<wm::matrix_b,16,16,16,half,wm::col_major> b;
            wm::fragment<wm::accumulator,16,16,16,float> c;
            wm::fill_fragment(c,0.0f);
            for(int d=0;d<Dim;d+=16) {
                wm::load_matrix_sync(a,query+d,Dim);
                wm::load_matrix_sync(b,k+size_t(key+warp*16)*CacheStride+kv*Dim+d,CacheStride);
                wm::mma_sync(c,a,b,c);
            }
            wm::store_matrix_sync(scores+warp*16,c,Keys,wm::mem_row_major);
        }
        __syncthreads();
        // Four rows per warp, with the same maximum/sum shuffle and FP32 recurrence.
#pragma unroll
        for(int own=0;own<4;++own) {
            const int row=warp+own*4;
            const bool valid=row<active;
            const int end=first+qbase+row+1;
            float s0=valid && key+lane<end?scores[row*Keys+lane]*4.0f:-CUDART_INF_F;
            float s1=valid && key+lane+32<end?scores[row*Keys+lane+32]*4.0f:-CUDART_INF_F;
            const float old_max=Reuse?previous_max[own]:stats[row];
            const float old_sum=Reuse?previous_sum[own]:stats[Rows+row];
            const float m=valid?fmaxf(old_max,maximum(fmaxf(s0,s1))):0;
            const float p0=valid?expf(s0-m):0, p1=valid?expf(s1-m):0;
            const float scale=valid?expf(old_max-m):0;
            const float sum=total(p0+p1);
            const float denominator=valid?old_sum*scale+sum:1;
            probability[row*Keys+lane]=__float2half_rn(p0);
            probability[row*Keys+lane+32]=__float2half_rn(p1);
            if constexpr (Reuse) {
                // Lane0 remains the authoritative recurrence value, as in shared_stats.
                previous_max[own]=__shfl_sync(0xffffffffu,m,0);
                previous_sum[own]=__shfl_sync(0xffffffffu,denominator,0);
                if(lane==0) query_phase.rows[row]=scale;
            } else if(lane==0) {
                stats[row]=m; stats[Rows+row]=denominator; stats[2*Rows+row]=scale;
            }
        }
        __syncthreads();
        if constexpr (Compact) {
            for(int i=tid;i<Rows*16;i+=128) scores[i]=stats[2*Rows+i/16];
            __syncthreads();
        }
        // One canonical PV body. Preserve four independent MMA chains in every schedule.
        wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major> a;
        wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major> b;
        wm::fragment<wm::accumulator,16,16,16,float> c[4];
        for(int j=0;j<4;++j) wm::fill_fragment(c[j],0.0f);
        for(int d=0;d<Keys;d+=16) {
            wm::load_matrix_sync(a,probability+d,Keys);
            for(int j=0;j<4;++j) {
                wm::load_matrix_sync(b,v+size_t(key+d)*CacheStride+kv*Dim+warp*64+j*16,CacheStride);
                wm::mma_sync(c[j],a,b,c[j]);
            }
        }
        if constexpr (Compact) {
            wm::fragment<wm::accumulator,16,16,16,float> previous,scale;
            wm::load_matrix_sync(scale,scores,16,wm::mem_row_major);
            for(int j=0;j<4;++j) {
                wm::load_matrix_sync(previous,running+warp*64+j*16,Dim,wm::mem_row_major);
                for(int i=0;i<c[j].num_elements;++i) c[j].x[i]=fmaf(previous.x[i],scale.x[i],c[j].x[i]);
                wm::store_matrix_sync(running+warp*64+j*16,c[j],Dim,wm::mem_row_major);
            }
            __syncthreads();
        } else if constexpr (Reuse) {
            // Product storage aliases probability: all warps must finish PV reads.
            __syncthreads();
            float scale[Rows];
            const volatile float* scale_rows=query_phase.rows;
#pragma unroll
            for(int row=0;row<Rows;++row) scale[row]=scale_rows[row];
#pragma unroll
            for(int pair=0;pair<4;pair+=2) {
                wm::store_matrix_sync(tile.product+warp*32,c[pair],128,wm::mem_row_major);
                wm::store_matrix_sync(tile.product+warp*32+16,c[pair+1],128,wm::mem_row_major);
                __syncthreads();
                // Every scale has reached registers. Restore the exact first32 query
                // halves while merges use registers, before the next QK iteration.
                if(pair==0 && tid<32) {
                    const float value=q[(size_t(qbase)*Heads+head)*Dim+tid];
                    query[tid]=__hmul(__float2half_rn(value*0.0625f),__float2half_rn(0.25f));
                }
#pragma unroll
                for(int row=0;row<Rows;++row) {
                    const int col=warp*64+pair*16+lane;
                    const int cell=row*Dim+col;
                    running[cell]=running[cell]*scale[row]+tile.product[row*128+tid];
                }
                __syncthreads();
            }
        } else {
            __shared__ __align__(32) float product[Rows*Dim];
            for(int j=0;j<4;++j)
                wm::store_matrix_sync(product+warp*64+j*16,c[j],Dim,wm::mem_row_major);
            __syncthreads();
            for(int i=tid;i<Rows*Dim;i+=128) running[i]=running[i]*stats[2*Rows+i/Dim]+product[i];
            __syncthreads();
        }
    }
    if constexpr (Reuse) {
        if(lane==0) {
#pragma unroll
            for(int own=0;own<4;++own) query_phase.rows[warp+own*4]=previous_sum[own];
        }
        __syncthreads();
    }
    for(int i=tid;i<active*Dim;i+=128) {
        const float denominator=Reuse?query_phase.rows[i/Dim]:stats[Rows+i/Dim];
        out[(size_t(qbase+i/Dim)*Heads+head)*Dim+i%Dim]=running[i]/denominator;
    }
#endif
}
#include "attention_tc_register.cuh"
#include "attention_tc_query32.cuh"
#include "attention_tc_halfpv.cuh"
#include "attention_tc_gqa_async.cuh"
#include "attention_tc_gqa32_async.cuh"
#include "attention_tc_gqa_key64.cuh"
#include "attention_tc_query64.cuh"
void prepare_query64() {
    // One selected device owns this runtime; configure before a launch/capture.
    static const bool configured=[] {
        const auto error=cudaFuncSetAttribute(prefill_query64,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              int(Query64SharedBytes));
        if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return true;
    }();
    (void)configured;
}
int selected_gqa_async() {
    static const int enabled=[] {
        const char* raw=std::getenv("STRATA_QWEN35_ATTN_GQA_ASYNC");
        if(!raw || !std::strcmp(raw,"0")) return 0;
        if(!std::strcmp(raw,"1")) return 1;
        if(!std::strcmp(raw,"2")) return 2;
        throw std::invalid_argument("attention GQA_ASYNC requires0,1 or2");
    }();
    return enabled;
}
void prepare_gqa_async() {
    int device=0;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    static thread_local int prepared=-1;
    if(prepared==device) return;
    cudaDeviceProp properties{};error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if(properties.major!=12 || properties.minor!=0) throw std::invalid_argument("GQA async requires SM120");
    error=cudaFuncSetAttribute(prefill_gqa_async,cudaFuncAttributeMaxDynamicSharedMemorySize,GqaAsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    prepared=device;
}
void prepare_gqa32_async() {
    int device=0;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    static thread_local int prepared=-1;
    if(prepared==device) return;
    cudaDeviceProp properties{};error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if(properties.major!=12 || properties.minor!=0) throw std::invalid_argument("GQA async requires SM120");
    error=cudaFuncSetAttribute(prefill_gqa32_async,cudaFuncAttributeMaxDynamicSharedMemorySize,Gqa32AsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    prepared=device;
}
bool selected_half_pv() {
    static const bool enabled=[] {
        const char* raw=std::getenv("STRATA_QWEN35_ATTN_TC_HALF_PV");
        if(!raw || !std::strcmp(raw,"0")) return false;
        if(!std::strcmp(raw,"1")) return true;
        throw std::invalid_argument("attention HALF_PV requires0 or1");
    }();
    return enabled;
}
int selected_query_rows() {
    static const int rows=[] {
        const char* value=std::getenv("STRATA_QWEN35_ATTN_TC_QUERY_ROWS");
        if(!value || !std::strcmp(value,"16")) return 16;
        if(!std::strcmp(value,"32")) return 32;
        if(!std::strcmp(value,"64")) return 64;
        throw std::invalid_argument("attention query rows require16,32 or64");
    }();
    return rows;
}
}
bool attention_rows32_selected(int first,int count) {
    const char* flag=std::getenv("STRATA_QWEN35_ATTN_QUERY_ROWS32");
    if(!flag || std::strcmp(flag,"0")==0) return false;
    if(std::strcmp(flag,"1")!=0) throw std::invalid_argument("ATTN_QUERY_ROWS32 must be0 or1");
    return count>=1024 && first+count>=8192;
}
void attention_tc_prefill_rows32_direct_request(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream,bool compact,bool reuse,bool request_direct) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity) || (compact && reuse))
        throw std::invalid_argument("unsupported Tensor Core PP attention geometry");
    if(reuse && register_output_enabled()) {
        if(attention_rows64_early_qpad8_selected(first,count,capacity) &&
           attention_rows64_early_qpad8_prefill(q,k,v,first,count,capacity,out,stream)) return;
        if(attention_rows64_early_selected(first,count,capacity) &&
           attention_rows64_early_prefill(q,k,v,first,count,capacity,out,stream)) return;
        if(request_direct && attention_rows32_direct_candidate(first,count,capacity) &&
           attention_rows32_direct_prefill(q,k,v,first,count,capacity,out,stream)) return;
        const bool rows64=attention_rows64_selected(first,count,capacity);
        if(attention_rows32_selected(first,count)) {
            if(attention_rows64_warps8_selected(first,count,capacity) &&
               attention_rows64_warps8_prefill(q,k,v,first,count,capacity,out,stream)) return;
            if(rows64 && attention_rows64_qpad8_selected(first,count,capacity) &&
               attention_rows64_qpad8_prefill(q,k,v,first,count,capacity,out,stream)) return;
            if(rows64 && attention_rows64_subgroup4_selected(first,count,capacity) &&
               attention_rows64_subgroup4_prefill(q,k,v,first,count,capacity,out,stream)) return;
            if(rows64 && attention_rows64_fast_exp_selected(first,count,capacity) &&
               attention_rows64_fast_exp_prefill(q,k,v,first,count,capacity,out,stream)) return;
            if(rows64 && attention_rows64_prefill(q,k,v,first,count,capacity,out,stream)) return;
            if(attention_rows32_prefill(q,k,v,first,count,capacity,out,stream)) return;
        }
        attention_register_prefill(q,k,v,first,count,capacity,out,stream);
        return;
    }
    const int query_rows=compact ? selected_query_rows() : 16;
    const int gqa=selected_gqa_async();
    const bool key64=selected_gqa_key64();
    if(gqa && compact && query_rows==32 && count>=1024 && first+count<=30720) {
        if(gqa==2 && key64) attention_tc_prefill_gqa_key64(q,k,v,first,count,capacity,out,stream);
        else if(gqa==2) attention_tc_prefill_gqa32_async(q,k,v,first,count,capacity,out,stream);
        else attention_tc_prefill_gqa_async(q,k,v,first,count,capacity,out,stream);
        return;
    }
    const bool half_pv=selected_half_pv();
    if(half_pv && compact && query_rows==32 && count>=1024 && first+count<=30720) {
        attention_tc_prefill_half_pv(q,k,v,first,count,capacity,out,stream); return;
    }
    if(compact && query_rows!=16 && count>=1024) {
        attention_tc_prefill_query_rows(q,k,v,first,count,capacity,out,stream,query_rows); return;
    }
    const int width=first+count, padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    const dim3 grid(Heads,(count+Rows-1)/Rows);
    if(reuse) prefill<false,true><<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    else if(compact) prefill_register<<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    else prefill<false,false><<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void attention_tc_prefill(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream,bool compact,bool reuse) {
    attention_tc_prefill_rows32_direct_request(q,k,v,first,count,capacity,out,stream,compact,reuse,
        attention_rows32_direct_selected(first,count,capacity));
}
#include "attention_tc_query_wrappers.inl"

}
