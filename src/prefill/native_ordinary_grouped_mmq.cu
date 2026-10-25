// Default-off native grouped adapter; pinned ordinary MMA math qualified independently.
#if !defined(STRATA_DEEPSEEK_V4_GGML_API) || STRATA_DEEPSEEK_V4_GGML_API != 1
#error "ordinary grouped requires the actual D4 private GGML ABI"
#endif
#include "native_ordinary_grouped_mmq.hpp"
#include "mmq.cuh"
#include <stdexcept>
namespace strata::prefill::mmq {
namespace {
constexpr int J=64;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<bool D4>
__global__ void gather_q8(const block_q8_1* input,block_q8_1_mmq* output,int cols,int entries,
    int source_rows,const int32_t* ids) {
    const int cell=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell>=entries*(cols/128))return;
    const int row=ids[cell%entries],kb=cell/entries;
    if(row<0||row>=source_rows)return; // Host native_mmq_plan already rejects these.
    const auto* source=input+size_t(row)*(cols/32)+4*kb;
    auto& dst=output[cell];
#pragma unroll
    for(int p=0;p<4;++p){
        if constexpr(D4)dst.d4[p]=__low2float(source[p].ds);
        else {
            int sum=0;
#pragma unroll
            for(int k=0;k<32;++k)sum+=int(source[p].qs[k]);
            dst.ds4[p]=make_half2(__low2half(source[p].ds),__float2half_rn(__low2float(source[p].ds)*sum));
        }
#pragma unroll
        for(int k=0;k<8;++k)reinterpret_cast<uint32_t*>(dst.qs+p*32)[k]=reinterpret_cast<const uint32_t*>(source[p].qs)[k];
    }
}
template<ggml_type Type,bool Active>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(Type,J,false,GGML_PREC_Q8),
    ggml_cuda_mmq_get_occupancy(Type,J,false,GGML_PREC_Q8))
__global__ void ordinary_grouped_kernel(const NativeMmqMatrix* matrices,const int32_t* output_ids,
    const int* packed,float* output,int entries,int cols,int outputs,const NativeMmqTile* tiles) {
#if __CUDA_ARCH__ == 750
    constexpr auto c=ggml_cuda_mmq_get_config(Type,J,false,GGML_PREC_Q8);
    static_assert(c.use_mma_data_layout()&&c.I==128&&c.nthreads==256&&c.K_vram==256&&c.J==64,
        "ordinary grouped SM75 host/device MMA contract");
    constexpr int I=c.I,warps=c.nthreads/32;
    const auto tile=Active?tiles[blockIdx.y]:NativeMmqTile{int(blockIdx.z),int(blockIdx.y)};
    const auto m=matrices[tile.matrix];const int count=m.end-m.begin,jt=tile.tile;
    if(jt*J>=count)return;
    const int out_tile=int(blockIdx.x),part=out_tile/(outputs/I),out_first=(out_tile%(outputs/I))*I;
    extern __shared__ int ids[];
    const int j=int(threadIdx.y)*32+int(threadIdx.x);
    if(j<J)ids[j]=jt*J+j<count?output_ids[m.begin+jt*J+j]:0;
    __syncthreads();
    constexpr int sz=sizeof(block_q8_1_mmq)/sizeof(int);
    mul_mat_q_process_tile<Type,J,false,false,GGML_PREC_Q8>(
        reinterpret_cast<const char*>(uintptr_t(part?m.up:m.gate)),out_first*(cols/256),
        packed+(m.begin+jt*J)*sz,ids,output+size_t(part)*entries*outputs+out_first,nullptr,nullptr,
        cols/256,entries,outputs,outputs-out_first-1,count-jt*J-1,0,cols/256);
#endif
}
ggml_cuda_mmq_config host_config(int type) {
    const auto& d=ggml_cuda_info().devices[ggml_cuda_get_device()];
    const auto c=ggml_cuda_mmq_get_config(ggml_type(type),J,false,d.cc,GGML_PREC_Q8);
    if(d.cc!=750||ggml_cuda_highest_compiled_arch(d.cc)!=750||c.type!=type||!c.use_mma_data_layout(d.cc)||
        c.I!=128||c.J!=64||c.nthreads!=256||c.K_vram!=256||mmq_get_nbytes_shared(c,d.cc)>d.smpbo)
        throw std::runtime_error("ordinary grouped actual SM75 MMA/resource contract");
    return c;
}
template<ggml_type Type,bool Active>
void launch(const NativeMmqMatrix* matrices,int groups,int parts,const int32_t* output_ids,const void* packed,
    float* output,int entries,int max_rows,int cols,int outputs,void* stream,const NativeMmqTile* tiles,int tile_count) {
    const auto c=host_config(Type);const int shared=int(mmq_get_nbytes_shared(c,750));
    CUDA_SET_SHARED_MEMORY_LIMIT((ordinary_grouped_kernel<Type,Active>),shared);
    const dim3 grid(unsigned(parts*outputs/c.I),unsigned(Active?tile_count:(max_rows+J-1)/J),unsigned(Active?1:groups));
    ordinary_grouped_kernel<Type,Active><<<grid,dim3(32,unsigned(c.nthreads/32)),shared,static_cast<cudaStream_t>(stream)>>>(
        matrices,output_ids,static_cast<const int*>(packed),output,entries,cols,outputs,tiles);
    ck(cudaGetLastError());
}
template<ggml_type Type,bool Active> NativeOrdinaryGroupedMmqResources resource_info() {
    const auto& d=ggml_cuda_info().devices[ggml_cuda_get_device()];
    NativeOrdinaryGroupedMmqResources r{};r.cc=d.cc;r.highest_arch=ggml_cuda_highest_compiled_arch(d.cc);
    if(d.cc!=750||r.highest_arch!=750){r.reason_mask=1;return r;}
    const auto c=ggml_cuda_mmq_get_config(Type,J,false,d.cc,GGML_PREC_Q8);
    r.I=c.I;r.J=c.J;r.threads=c.nthreads;r.K_vram=c.K_vram;r.mma_layout=c.use_mma_data_layout(d.cc);
    r.shared=mmq_get_nbytes_shared(c,d.cc);
    if(c.type!=Type||!r.mma_layout||c.I!=128||c.J!=64||c.nthreads!=256||c.K_vram!=256||r.shared!=48384||r.shared>d.smpbo)
        {r.reason_mask=2;return r;}
    cudaFuncAttributes a{};ck(cudaFuncGetAttributes(&a,ordinary_grouped_kernel<Type,Active>));
    r.registers=a.numRegs;r.local=a.localSizeBytes;r.binary=a.binaryVersion;
    if(a.binaryVersion!=75||a.maxThreadsPerBlock<256||a.numRegs>255)r.reason_mask|=4;
    // Q4/Q6 retain their observed limits. Q5 <=8B is an explicit new qualification ceiling, not an observed result.
    if(a.localSizeBytes>(Type==GGML_TYPE_Q4_K?0u:8u))r.reason_mask|=8;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,ordinary_grouped_kernel<Type,Active>,256,r.shared));
    if(r.active_blocks<1)r.reason_mask|=16;
    return r;
}
}
void ordinary_grouped_repack(const void* input,void* output,int type,int cols,int source_rows,int entries,
    const int32_t* source_ids,void* stream) {
    if((type!=12&&type!=13&&type!=14)||!input||!output||!source_ids||!stream||(cols!=512&&cols!=2048)||(type==13&&cols!=512)||
        entries<1||entries>65536||source_rows<1||source_rows>65536)throw std::invalid_argument("ordinary gathered Q8 geometry");
    const int cells=entries*(cols/128);auto st=static_cast<cudaStream_t>(stream);
    ck(cudaMemsetAsync(static_cast<block_q8_1_mmq*>(output)+cells,0,128*sizeof(block_q8_1_mmq),st));
    if(type==12||type==13)gather_q8<false><<<(cells+127)/128,128,0,st>>>(static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1_mmq*>(output),cols,entries,source_rows,source_ids);
    else gather_q8<true><<<(cells+127)/128,128,0,st>>>(static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1_mmq*>(output),cols,entries,source_rows,source_ids);
    ck(cudaGetLastError());
}
void ordinary_grouped_tile(int type,const NativeMmqMatrix* matrices,int groups,int parts,const int32_t* output_ids,
    const void* packed,float* output,int entries,int max_rows,int cols,int outputs,void* stream,const NativeMmqTile* tiles,int tile_count) {
    if((type!=12&&type!=13&&type!=14)||!matrices||!output_ids||!packed||!output||!stream||groups<1||groups>256||
        (parts!=1&&parts!=2)||entries<1||entries>65536||max_rows<1||max_rows>8192||max_rows>entries||
        !((cols==2048&&outputs==512&&parts==2&&type==12)||(cols==512&&outputs==2048&&parts==1))||
        (type==13&&tiles)||(tiles?(tile_count<1||tile_count>1280):tile_count!=0))throw std::invalid_argument("ordinary grouped measured shape/tiles");
#define CALL(T,A) launch<T,A>(matrices,groups,parts,output_ids,packed,output,entries,max_rows,cols,outputs,stream,tiles,tile_count)
    if(type==12){if(tiles)CALL(GGML_TYPE_Q4_K,true);else CALL(GGML_TYPE_Q4_K,false);}
    else if(type==13)CALL(GGML_TYPE_Q5_K,false);
    else{if(tiles)CALL(GGML_TYPE_Q6_K,true);else CALL(GGML_TYPE_Q6_K,false);}
#undef CALL
}
NativeOrdinaryGroupedMmqResources native_ordinary_grouped_resources(int type,bool active) {
    if(type==12)return active?resource_info<GGML_TYPE_Q4_K,true>():resource_info<GGML_TYPE_Q4_K,false>();
    if(type==13){if(active){NativeOrdinaryGroupedMmqResources r{};r.reason_mask=32;return r;}
        return resource_info<GGML_TYPE_Q5_K,false>();}
    if(type==14)return active?resource_info<GGML_TYPE_Q6_K,true>():resource_info<GGML_TYPE_Q6_K,false>();
    NativeOrdinaryGroupedMmqResources r{};r.reason_mask=32;return r;
}
bool native_ordinary_grouped_supported(int type,int cols,int outputs,int parts,bool active) {
    if(!((type==12&&cols==2048&&outputs==512&&parts==2)||
        ((type==12||type==13||type==14)&&cols==512&&outputs==2048&&parts==1)))return false;
    return native_ordinary_grouped_resources(type,active).reason_mask==0;
}
}
