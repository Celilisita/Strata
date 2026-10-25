#pragma once
#include "strata/qwen35/completed_v100/native/original_formats.hpp"
#include "strata/qwen35/completed_v100/native/dense_gemm.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_projection.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_prefill_lt.hpp"
namespace strata::completed_v100::qwen35::native {
constexpr size_t Nvfp4HeaderBytes=OriginalNvfp4HeaderBytes;
// Existing DenseGemm leases carry this format provider; the base ABI/vtable and
// Matrix ABI stay unchanged. Owner precedes model/graphs and outlives them.
class Nvfp4Gemm:public DenseGemm {
    Nvfp4Projection projection_;
    unsigned long long calls_=0;
    bool q8_pp_gdn_=false;
    Nvfp4PrefillLt pp_lt_;bool q8_pp_lt_=false;unsigned long long pp_lt_calls_=0;
    bool q8_pp_fused_=false;unsigned long long pp_fused_calls_=0,pp_long_calls_=0,pp_tail_calls_=0;
public:
    Nvfp4Gemm(size_t weights,int input_width,int tokens):projection_(weights,input_width,tokens),pp_lt_(int(weights/size_t(input_width)),input_width,tokens){}
    Nvfp4BlasLease blas_lease()const{return projection_.blas_lease();}
    void record_prefetched_ffn(){calls_+=3;++pp_fused_calls_;}
    void q8_pp_fused(bool value){q8_pp_fused_=value;pp_fused_calls_=0;}
    bool q8_pp_fused()const{return q8_pp_fused_;}
    unsigned long long pp_fused_calls()const{return pp_fused_calls_;}
    bool prefill_fused(Matrix,Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
    void q8_pp_lt(bool value){if(value&&!pp_lt_.info().ready)throw std::runtime_error("PP Lt plan unavailable");q8_pp_lt_=value;pp_lt_calls_=0;}
    bool q8_pp_lt()const{return q8_pp_lt_;}
    Nvfp4LtInfo pp_lt_info()const{return pp_lt_.info();}
    unsigned long long pp_lt_calls()const{return pp_lt_calls_;}
    void q8_pp_gdn(bool value){q8_pp_gdn_=value;}
    bool q8_pp_gdn()const{return q8_pp_gdn_;}
    bool prefill_gdn(const GdnPrefillProjection*,const float*,float*,size_t,int,cudaStream_t);
    void q8_pp_tail(bool value){projection_.q8_pp_tail(value);pp_tail_calls_=0;}
    bool q8_pp_tail()const{return projection_.q8_pp_tail();}
    bool q8_pp_shape(int tokens)const{return projection_.q8_pp_shape(tokens);}
    unsigned long long pp_tail_calls()const{return pp_tail_calls_;}
    void q8_pp_long(bool value){projection_.q8_pp_long(value);pp_long_calls_=0;}
    bool q8_pp_long()const{return projection_.q8_pp_long();}
    unsigned long long pp_long_calls()const{return pp_long_calls_;}
    void q8_pp_vector(bool value){projection_.q8_pp_vector(value);}
    bool q8_pp_vector()const{return projection_.q8_pp_vector();}
    void q8_pp_pair(bool value){projection_.q8_pp_pair(value);}
    bool q8_pp_pair()const{return projection_.q8_pp_pair();}
    bool prefill_pair(Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
    void q8_pp_decode8(bool value){projection_.q8_pp_decode8(value);}
    bool q8_pp_decode8()const{return projection_.q8_pp_decode8();}
    void q8_pair_half(bool value){projection_.q8_pair_half(value);}
    bool q8_pair_half()const{return projection_.q8_pair_half();}
    void q8_head_wide(bool value){projection_.q8_head_wide(value);}
    bool q8_head_wide()const{return projection_.q8_head_wide();}
    void q8_pp_p2(bool value){projection_.q8_pp_p2(value);}
    bool q8_pp_p2()const{return projection_.q8_pp_p2();}
    void q8_fastsoa(bool value){projection_.q8_fastsoa(value);}
    bool q8_fastsoa()const{return projection_.q8_fastsoa();}
    void q8_soa(bool value){projection_.q8_soa(value);}
    bool q8_soa()const{return projection_.q8_soa();}
    void q8_fast(bool value){projection_.q8_fast(value);}
    bool q8_fast()const{return projection_.q8_fast();}
    void q8_pair(bool value){projection_.q8_pair(value);}
    bool q8_pair()const{return projection_.q8_pair();}
    bool pair_original(Matrix,Matrix,const float*,float*,float*,float*,int,cudaStream_t);
    void q8_activation(int mode){projection_.q8_activation(mode);}
    int q8_activation()const{return projection_.q8_activation();}
    void library_decode(bool enabled){projection_.library_decode(enabled);}
    bool library_decode()const{return projection_.library_decode();}
    void cutlass(bool enabled){projection_.cutlass(enabled);}
    bool cutlass()const{return projection_.cutlass();}
    void fused_cutlass(bool decode,bool prefill,int tile=0,bool selective=false){projection_.fused_cutlass(decode,prefill,tile,selective);}
    void combined_pp(bool enabled){projection_.combined_pp(enabled);}
    bool combined_pp()const{return projection_.combined_pp();}
    void mixed_decode(bool enabled){projection_.mixed_decode(enabled);}
    bool mixed_decode()const{return projection_.mixed_decode();}
    void q8_blocked(bool enabled){projection_.q8_blocked(enabled);}
    bool q8_blocked()const{return projection_.q8_blocked();}
    void packed_cached(bool enabled){projection_.packed_cached(enabled);}
    bool packed_cached()const{return projection_.packed_cached();}
    void q8_optimized(bool enabled){projection_.q8_optimized(enabled);}
    bool q8_optimized()const{return projection_.q8_optimized();}
    void q8_dp4a(int group){projection_.q8_dp4a(group);}
    int q8_dp4a()const{return projection_.q8_dp4a();}
    void q8_warps(int value){projection_.q8_warps(value);}
    int q8_warps()const{return projection_.q8_warps();}
    bool fused_selective()const{return projection_.fused_selective();}
    unsigned long long fused_calls()const{return projection_.fused_calls();}
    bool fused_decode()const{return projection_.fused_decode();}
    bool fused_prefill()const{return projection_.fused_prefill();}
    void packed_decode(bool enabled){projection_.packed_decode(enabled);}
    bool packed_decode()const{return projection_.packed_decode();}
    void project_original(Matrix,const float*,float*,int,cudaStream_t);
    void project(const void*,const void*,float*,int,int,int,cudaStream_t)override {
        throw std::invalid_argument("Original quantized provider requires raw F32 input boundary");
    }
    void batched_fp16(const void*,const void*,float*,GemmBatch,cudaStream_t)override {
        throw std::invalid_argument("Original quantized provider has no unrelated FP16 batch lease");
    }
    size_t workspace_bytes()const override{return projection_.workspace_bytes();}
    unsigned long long calls()const override{return calls_;}
};
void project_original(DenseGemm*,Matrix,const float*,float*,int,cudaStream_t);
void initialize_nvfp4_header(void* header,cudaStream_t);
bool try_q8_prefill_fused(DenseGemm*,Matrix,Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
bool try_q8_prefill_pair(DenseGemm*,Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
bool try_q8activation_pair(DenseGemm*,Matrix,Matrix,const float*,float*,float*,float*,int,cudaStream_t);
}
