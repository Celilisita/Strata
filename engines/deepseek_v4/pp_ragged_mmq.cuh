#pragma once
#include "mmq.cuh"
namespace strata::deepseek_v4 {
// Dispatch actual expert column tiles; arithmetic/load/writeback are the
// canonical MMQ primitive. No rectangular empty tiles or split-K fixup work.
template<ggml_type type, int J, ggml_prec precision>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type,J,false,precision),
                  ggml_cuda_mmq_get_occupancy(type,J,false,precision))
static __global__ void pp_ragged_tile(const char* x,const int* y,const int32_t* identity,
    const int32_t* bounds,const int32_t* prefix,float* dst,int experts,int cols,int rows,
    int quant_rows,int destination_rows,int expert_blocks) {
    if constexpr(ggml_cuda_mmq_get_config(type,J,false,precision).type==GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;return;
    } else {
        constexpr int I=ggml_cuda_mmq_get_I(type,J,false,precision);
        constexpr int threads=ggml_cuda_mmq_get_nthreads(type,J,false,precision);
        extern __shared__ int data[];
        __shared__ int expert,first_column;
        const int lane=threadIdx.y*blockDim.x+threadIdx.x;
        if(!lane) {
            int low=0,high=experts;
            while(low+1<high) {const int middle=(low+high)/2;if(prefix[middle]<=int(blockIdx.y))low=middle;else high=middle;}
            expert=low;first_column=bounds[low]+(int(blockIdx.y)-prefix[low])*J;
        }
        __syncthreads();
        for(int j=lane;j<J;j+=threads)data[j]=identity[first_column+j];
        __syncthreads();
        const int row=int(blockIdx.x)*I;
        const int offset_x=expert*expert_blocks+row*(cols/ggml_cuda_type_traits<type>::qk);
        const int offset_y=first_column*(sizeof(block_q8_1_mmq)/sizeof(int));
        mul_mat_q_process_tile<type,J,false,false,precision>(
            x,offset_x,y+offset_y,data,dst+row,nullptr,nullptr,cols/ggml_cuda_type_traits<type>::qk,quant_rows,destination_rows,
            rows-row-1,bounds[expert+1]-first_column-1,0,cols/ggml_cuda_type_traits<type>::qk);
    }
}
template<ggml_type type, int J,ggml_prec precision>
static void pp_ragged_launch(const mmq_args& a,const int32_t* prefix,int tiles,int cc,cudaStream_t stream) {
    const auto c=ggml_cuda_mmq_get_config(type,J,false,cc,precision);
    const int shared=mmq_get_nbytes_shared(c,cc);
    CUDA_SET_SHARED_MEMORY_LIMIT((pp_ragged_tile<type,J,precision>),shared);
    const dim3 grid((a.nrows_x+c.I-1)/c.I,tiles,1),block(ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size,
        c.nthreads/ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size,1);
    pp_ragged_tile<type,J,precision><<<grid,block,shared,stream>>>(a.x,a.y,a.ids_dst,a.expert_bounds,
        prefix,a.dst,int(a.nchannels_x),int(a.ncols_x),int(a.nrows_x),int(a.ncols_y),
        int(a.nrows_dst),int(a.stride_channel_x));
}
template<ggml_type type, ggml_prec precision>
static void pp_ragged_dispatch(const mmq_args& a,const int32_t* prefix,int tiles,int cc,cudaStream_t stream) {
#define STRATA_PP_J(j) case j:pp_ragged_launch<type,j,precision>(a,prefix,tiles,cc,stream);break
    switch(a.J_best) {
        STRATA_PP_J(8);STRATA_PP_J(16);STRATA_PP_J(24);STRATA_PP_J(32);
        STRATA_PP_J(40);STRATA_PP_J(48);STRATA_PP_J(56);STRATA_PP_J(64);
        STRATA_PP_J(72);STRATA_PP_J(80);STRATA_PP_J(88);STRATA_PP_J(96);
        STRATA_PP_J(104);STRATA_PP_J(112);STRATA_PP_J(120);STRATA_PP_J(128);
        default:GGML_ABORT("PP ragged tile width invalid");
    }
#undef STRATA_PP_J
}
}
