#include "strata/qwen35moe/attention_tc.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
namespace wm = nvcuda::wmma;
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
}
void attention_tc_prefill(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream,bool compact,bool reuse) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity) || (compact && reuse))
        throw std::invalid_argument("unsupported Tensor Core PP attention geometry");
    const int width=first+count, padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    const dim3 grid(Heads,(count+Rows-1)/Rows);
    if(reuse) prefill<false,true><<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    else if(compact) prefill<true,false><<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    else prefill<false,false><<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
AttentionTcResources attention_tc_resources(bool compact,bool reuse) {
    cudaFuncAttributes attr{};
    const auto error=reuse?cudaFuncGetAttributes(&attr,prefill<false,true>):
        compact?cudaFuncGetAttributes(&attr,prefill<true,false>):cudaFuncGetAttributes(&attr,prefill<false,false>);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes};
}
}
