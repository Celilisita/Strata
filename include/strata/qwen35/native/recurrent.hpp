#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include "strata/qwen35/native/projection.hpp"
#include "strata/qwen35/native/gdn.hpp"
#include "strata/qwen35/native/gdn_prefix.hpp"
#include <memory>
#include <array>
#include <vector>

namespace strata::qwen35::native {
struct NativeProjection32 {Matrix weight;float *output;};
class SmallQkv32;
class SmallQ4Projection32;
struct RecurrentShape {
    int hidden, dim, query_heads, value_heads, max_tokens;
    float epsilon;
    int mixed_capacity=0;
    size_t partial_rows_capacity=0;
    // Dead FFN activation storage may hold attention intermediates.
    bool compact_workspace=false;
};
struct RecurrentSchedule { ReductionSchedule projections[5]{}; bool wide_grid=false; };
struct RecurrentWeights {
    Matrix qkv, z, alpha, beta, out, ffn_gate, ffn_up, ffn_down;
    const float *norm, *post_norm, *conv, *dt, *a, *gdn_norm;
};
struct RecurrentInputs { const float *hidden, *conv_history, *state; };
// Conv history output must be distinct; GDN state output may alias its input.
struct RecurrentStateOutputs { float *conv_history, *state; GdnPrefix prefix{}; };
struct RecurrentBuffers {
    float *norm, *qkv, *z, *alpha, *beta_raw, *g, *beta;
    float *conv_raw, *conv_silu, *q, *k, *v, *attention, *gated;
    float *projected, *residual, *post_norm, *output;
};
// Borrow one quantized weight inventory for both phases. All allocations are
// setup-only. Input/state/output/stream lifetimes extend through completion.
class RecurrentLayer {
public:
    RecurrentLayer(RecurrentWeights, RecurrentShape, int down_splits = 8, RecurrentSchedule = {});
    ~RecurrentLayer();
    RecurrentLayer(const RecurrentLayer &) = delete;
    RecurrentLayer &operator=(const RecurrentLayer &) = delete;
    using PhaseMarks=std::array<cudaEvent_t,11>;
    // Caller keeps this optional bounded diagnostic lease alive to completion.
    void execute(RecurrentInputs, RecurrentStateOutputs, int tokens, cudaStream_t,const PhaseMarks *diagnostic_marks=nullptr);
    const RecurrentBuffers &buffers() const { return buffers_; }
    const RecurrentShape &shape() const { return shape_; }
    int max_tokens() const { return shape_.max_tokens; }
    Ffn &shared_ffn(int tokens) {
        if(!tg_ && tokens!=32)throw std::invalid_argument("prepared layer FFN is PP32 only");
        return *(tokens==1?tg_:pp_);
    }
    void rebind(RecurrentWeights);
    // Explicit experimental setup; never enabled by the model constructor.
    // A prepared cache pins this layer's raw QKV identity until destruction.
    void enable_native32(cudaStream_t);
    void enable_joint_projections();
    void use_pp_gemm(DenseGemm *service,bool ffn=true,bool projections=true){
        pp_->use_pp_gemm(ffn?service:nullptr);pp_gemm_=projections?service:nullptr;
    }
    // Select the measured PP schedule only while the caller stream is idle.
    void use_pp_gdn_rows(int rows){
        if(rows!=2&&rows!=4)throw std::invalid_argument("unsupported PP GDN row grouping");
        pp_gdn_rows_=rows;
    }
    using PpGdnKernel=void (*)(GdnInputs,GdnShape,float *,float *,void *,size_t,cudaStream_t);
    using PpGdnWorkspace=size_t (*)(GdnShape);
    void use_pp_gdn_kernel(PpGdnKernel kernel,PpGdnWorkspace workspace){
        if(bool(kernel)!=bool(workspace))throw std::invalid_argument("incomplete PP GDN service");
        pp_gdn_kernel_=kernel;pp_gdn_workspace_=workspace;
    }
    void enable_tiny_batches(){tiny_batches_=true;pp_->enable_tiny_batches();}
    void enable_tiny_row_ffn(){enable_tiny_batches();pp_->enable_tiny_row_ffn();}
    void enable_tiny_meta_ffn(){enable_tiny_batches();pp_->enable_tiny_meta_ffn();}
    bool tiny_batches(int tokens)const{return tiny_batches_&&(tokens==2||tokens==3);}
    bool native32_enabled() const {return native32_;}
    void enable_fused_pack32(){
        if(!native32_||shape_.dim!=128||weights_.qkv.quant!=Quant::Q6K)
            throw std::invalid_argument("fused packing requires owned Q6/Q4 PP32 and128-dim GDN");
        fused_pack32_=true;
    }
    void project_native32(const float *,std::initializer_list<NativeProjection32>,cudaStream_t);
    void enable_qkv32(cudaStream_t,bool complete=false,bool register_reuse=false,bool prefetch=false,bool planes=false,bool byte_planes=false);
    size_t qkv32_cache_bytes()const;
    double qkv32_cold_ms()const;
    // Cold, model-owned immutable view. Rebind clears this borrowed execution lease.
    void borrow_native32_qkv_planes(const void *,size_t,cudaStream_t);
    void enable_q4_projection32(cudaStream_t,bool norm_pair=true,bool fused_ops=false);
    // Common projection staging for both Dense layer kinds, ordered on caller stream.
    void project(Matrix, const void *prepared_input, float *, int tokens, ReductionSchedule, cudaStream_t);
    const Buffers &ffn_buffers(int tokens) const {
        if(!tg_ && tokens!=32)throw std::invalid_argument("prepared layer FFN is PP32 only");
        return (tokens == 1 ? tg_ : pp_)->buffers();
    }
    size_t workspace_bytes() const { return workspace_bytes_; }
    unsigned launches(int tokens) const {
        if(!tg_ && tokens!=32)throw std::invalid_argument("prepared layer launches are PP32 only");
        if (tokens == 1) return 21;
        if(tokens<64){
            const bool q4=q4projection32_&&tokens==32;
            const unsigned qkv=qkv32_&&tokens==32?qkv32_extra_submissions_:(native32_&&tokens==32&&weights_.qkv.quant==Quant::Q6K?1:0);
            return (fused_ops_&&q4?12:17)+pp_->launches(tokens)+qkv+(q4?(q4_norm_pair_?4:3):0)-(fused_pack32_&&tokens==32&&weights_.qkv.quant==Quant::Q6K?3:0);
        }
        unsigned count = 22 + pp_->launches(tokens);
        const Matrix matrices[]={weights_.qkv,weights_.z,weights_.alpha,weights_.beta,weights_.out};
        for(int i=0;i<5;++i)count+=projection_partitions(matrices[i],tokens,schedule_.projections[i])>1;
        return count;
    }
private:
    bool tiny_batches_=false;
    int pp_gdn_rows_=2;
    PpGdnKernel pp_gdn_kernel_=nullptr;
    PpGdnWorkspace pp_gdn_workspace_=nullptr;
    DenseGemm *pp_gemm_=nullptr;
    int projection_partitions(Matrix,int tokens,ReductionSchedule)const;
    bool joint_projections_=false;
    size_t decoded_capacity_=0, partial_rows_capacity_=0;
    int input_capacity_=0;
    RecurrentWeights weights_;
    RecurrentShape shape_;
    RecurrentSchedule schedule_;
    RecurrentBuffers buffers_{};
    std::unique_ptr<Ffn> pp_, tg_;
    std::unique_ptr<SmallQkv32> qkv32_;
    std::unique_ptr<SmallQ4Projection32> q4projection32_;
    bool q4_norm_pair_=true,fused_ops_=false,native32_=false;cudaStream_t native32_stream_=nullptr;
    bool fused_pack32_=false;
    unsigned qkv32_extra_submissions_=0;
    const void *qkv_plane_data_=nullptr;size_t qkv_plane_bytes_=0;
    std::vector<void *> allocations_;
    float *projection_partials_ = nullptr;
    size_t workspace_bytes_ = 0;
};
}
