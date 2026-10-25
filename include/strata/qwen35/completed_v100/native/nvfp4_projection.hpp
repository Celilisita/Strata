#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>

namespace strata::completed_v100::qwen35::native {
struct Nvfp4BlasLease {
    cublasHandle_t handle;void* workspace;size_t bytes;const float* one;const float* zero;
};
struct GdnPrefillProjection;
class Nvfp4PrefillLt;
// Original NVFP4 blocks and their F32 global scale stay separate. The same
// projection interface uses direct F32 decode for TG and vendor Tensor Cores
// with high/low input decomposition for PP. No CPU compute provider exists.
class Nvfp4Projection {
    cublasHandle_t handle_=nullptr;
    void *weights_=nullptr,*high_=nullptr,*low_=nullptr;
    float* constants_=nullptr;
    void* vendor_workspace_=nullptr;
    void* packed_input_=nullptr;
    bool packed_decode_=false,q8_optimized_=false,packed_cached_=false,mixed_decode_=false,q8_blocked_=false,combined_pp_=false;
    bool q8_pair_=false,q8_fast_=false,q8_soa_=false,q8_fastsoa_=false,q8_pp_p2_=false,q8_head_wide_=false,q8_pair_half_=false,q8_pp_decode8_=false,q8_pp_pair_=false,q8_pp_vector_=false,q8_pp_long_=false,q8_pp_tail_=false,volta_=false;
    int q8_activation_=0;
    bool library_decode_=false;
    bool cutlass_=false;
    bool fused_decode_=false,fused_prefill_=false,fused_selective_=false;
    unsigned long long fused_calls_=0;
    int fused_tile_=0,sm_count_=0,q8_warps_=0,q8_group_=0;
    static constexpr size_t VendorWorkspaceBytes=32*1024*1024;
    size_t weight_values_=0;
    int input_width_=0,maximum_tokens_=0;
    void run_combined_prefill(const void*,const float*,const float*,float*,int,int,int,cudaStream_t);
public:
    // Borrowed on this executor's ordered stream; owner outlives every graph.
    Nvfp4BlasLease blas_lease()const{return {handle_,vendor_workspace_,VendorWorkspaceBytes,constants_,constants_+3};}
    Nvfp4Projection(size_t maximum_weight_values,int maximum_input_width,int maximum_tokens);
    ~Nvfp4Projection();
    Nvfp4Projection(const Nvfp4Projection&)=delete;
    void run(const void* original_blocks,float global_scale,const float* input,float* output,
             int rows,int columns,int tokens,cudaStream_t);
    void run_device(const void* original_blocks,const float* scale_header,const float* input,float* output,
                    int rows,int columns,int tokens,cudaStream_t);
    void run_q8(const void* original_blocks,const float* input,float* output,
                int rows,int columns,int tokens,cudaStream_t);
    void run_q8_library(const void*,const float*,float*,int rows,int columns,int tokens,cudaStream_t);
    void q8_pp_tail(bool value){q8_pp_tail_=value;}
    bool q8_pp_tail()const{return q8_pp_tail_;}
    bool q8_pp_shape(int tokens)const{return tokens==1280||tokens==8192||(q8_pp_tail_&&tokens==6144);}
    void q8_pp_long(bool value){q8_pp_long_=value&&volta_;}
    bool q8_pp_long()const{return q8_pp_long_;}
    void q8_pp_vector(bool value){q8_pp_vector_=value;}
    bool q8_pp_vector()const{return q8_pp_vector_;}
    void q8_pp_pair(bool value){q8_pp_pair_=value;}
    bool q8_pp_pair()const{return q8_pp_pair_;}
    void run_prefill_fused(Nvfp4PrefillLt*,const void*,const float*,const void*,const float*,const void*,const float*,const float*,float*,size_t,float*,int,int,int,cudaStream_t);
    void run_prefill_lt(Nvfp4PrefillLt&,const void*,const float*,const float*,float*,int,int,int,cudaStream_t);
    void run_prefill_gdn(const GdnPrefillProjection*,const float*,float*,size_t,int,cudaStream_t);
    void run_prefill_pair(const void*,const float*,const void*,const float*,const float*,float*,size_t,float*,int,int,int,cudaStream_t);
    void q8_pp_decode8(bool value){q8_pp_decode8_=value;}
    bool q8_pp_decode8()const{return q8_pp_decode8_;}
    void q8_pair_half(bool value){q8_pair_half_=value;}
    bool q8_pair_half()const{return q8_pair_half_;}
    void q8_head_wide(bool value){q8_head_wide_=value;}
    bool q8_head_wide()const{return q8_head_wide_;}
    void q8_pp_p2(bool value){q8_pp_p2_=value;}
    bool q8_pp_p2()const{return q8_pp_p2_;}
    void q8_fastsoa(bool value){q8_fastsoa_=value;}
    bool q8_fastsoa()const{return q8_fastsoa_;}
    void q8_soa(bool value){q8_soa_=value;}
    bool q8_soa()const{return q8_soa_;}
    void q8_fast(bool value){q8_fast_=value;}
    bool q8_fast()const{return q8_fast_;}
    void q8_pair(bool value){q8_pair_=value;}
    bool q8_pair()const{return q8_pair_;}
    void run_pair(const void*,const float*,const void*,const float*,const float*,float*,float*,float*,int,int,int,cudaStream_t);
    void q8_activation(int mode){q8_activation_=mode;}
    int q8_activation()const{return q8_activation_;}
    void library_decode(bool enabled){library_decode_=enabled;if(enabled)packed_decode_=false;}
    bool library_decode()const{return library_decode_;}
    void cutlass(bool enabled){cutlass_=enabled;}
    bool cutlass()const{return cutlass_;}
    void fused_cutlass(bool decode,bool prefill,int tile=0,bool selective=false){
        fused_decode_=decode;fused_prefill_=prefill;fused_tile_=tile;fused_selective_=selective;fused_calls_=0;
    }
    void combined_pp(bool enabled){combined_pp_=enabled;}
    bool combined_pp()const{return combined_pp_;}
    void mixed_decode(bool enabled){mixed_decode_=enabled;}
    bool mixed_decode()const{return mixed_decode_;}
    void q8_blocked(bool enabled){q8_blocked_=enabled;}
    bool q8_blocked()const{return q8_blocked_;}
    void packed_cached(bool enabled){packed_cached_=enabled;}
    bool packed_cached()const{return packed_cached_;}
    void q8_optimized(bool enabled){q8_optimized_=enabled;}
    bool q8_optimized()const{return q8_optimized_;}
    void q8_dp4a(int group){q8_group_=group;}
    int q8_dp4a()const{return q8_group_;}
    void q8_warps(int value){q8_warps_=value;}
    int q8_warps()const{return q8_warps_;}
    bool fused_selective()const{return fused_selective_;}
    unsigned long long fused_calls()const{return fused_calls_;}
    bool fused_decode()const{return fused_decode_;}
    bool fused_prefill()const{return fused_prefill_;}
    void packed_decode(bool enabled){packed_decode_=enabled;if(enabled)library_decode_=false;}
    bool packed_decode()const{return packed_decode_;}
    size_t workspace_bytes()const{return weight_values_*4+size_t(input_width_)*maximum_tokens_*4+20+VendorWorkspaceBytes+size_t(input_width_)*(maximum_tokens_<3?maximum_tokens_:3)*4;}
};
}
