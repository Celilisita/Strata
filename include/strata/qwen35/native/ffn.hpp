#pragma once
#include "strata/qwen35/native/quant.hpp"
#include <cuda_runtime.h>
#include <initializer_list>
#include <memory>

namespace strata::qwen35::native {
class DenseGemm;
class PpWeightOverlap;
struct Capabilities {
    int major = 0, minor = 0, multiprocessors = 0;
    size_t shared_per_block = 0;
    bool asynchronous_copy = false, cluster_launch = false;
    bool tcgen05 = false;
};
Capabilities capabilities(int device);
enum class Provider { Tiled, OwnedStaged, OwnedVector };
struct Tile { int rows = 64, tokens = 32, reduction = 32; };
Tile choose_tile(const Capabilities &, int rows, int tokens);
struct Buffers {
    float *gate = nullptr, *up = nullptr, *activation = nullptr, *output = nullptr;
    float *down_partials = nullptr;
    void *half_weights = nullptr, *half_input = nullptr;
    void *quant_input = nullptr;
};
// Own ordered stream execution. No graph executor, llama/GGML state or callbacks.
class Ffn {
public:
    Ffn(Matrix gate, Matrix up, Matrix down, int max_tokens, Provider provider, int down_splits = 1);
    ~Ffn();
    Ffn(const Ffn &) = delete;
    Ffn &operator=(const Ffn &) = delete;
    // Intermediates are optional diagnostics for fused paths; normal folded activation uses half storage.
    void execute(const float *input, int tokens, cudaStream_t stream, bool diagnostic_projections=false);
    void use_pp_gemm(DenseGemm *);
    void enable_pp_weight_overlap(cudaStream_t,bool enabled=true);
    void select_pp_weight_overlap(bool);
    size_t pp_weight_overlap_bytes()const;
    void enable_tiny_batches(){tiny_batches_=true;}
    // Cold opt-in for the measured Q4 pair and three-position Q4 Down paths.
    void enable_tiny_row_ffn(){enable_tiny_batches();tiny_row_ffn_=true;tiny_meta_ffn_=false;}
    void enable_tiny_meta_ffn(){enable_tiny_batches();tiny_meta_ffn_=true;tiny_row_ffn_=false;}
    // Experimental PP32 policy; configure once on the execution thread/device.
    // Uses the same borrowed Q4/Q6 inventory and existing transient input arena.
    void enable_small_direct32(bool segmented_pair=false);
    // Independent continuous K reduction; retain the admitted Down schedule.
    void enable_whole_k_q4_pair(bool tensor_bias=false);
    // Cold diagnostic view in the idle decoded-weight arena; pins borrowed weights.
    void enable_q4_tiled_pair(cudaStream_t);
    size_t q4_tiled_view_bytes() const { return tiled_pair_bytes_; }
    // Cold PP32 canary in the idle arena. Raw inventory remains authoritative.
    void enable_q4_group_pair(cudaStream_t);
    void verify_q4_group_pair(cudaStream_t);
    void select_q4_group_pair(bool);
    size_t q4_group_view_bytes() const { return group_pair_bytes_; }
    double q4_group_cold_ms() const { return group_pair_cold_ms_; }
    // Cold opt-in: reuse decoded weights across a 128-token Tensor Core tile.
    void enable_wide_tiles();
    // Cold canary: reuse both operands across a 32x32 warp output tile.
    void enable_balanced_tiles();
    void select_balanced_tiles(bool);
    void enable_pipeline_tiles();
    void select_pipeline_tiles(bool);
    // Cold canary: neighboring token CTAs share one feature block's weights.
    void enable_raster_tiles();
    void select_raster_tiles(bool);
    void enable_joint_tiles();
    void select_joint_tiles(bool);
    void enable_token256_pair();
    void enable_folded_pair();
    void select_folded_pair(bool);
    void enable_folded_compact_pair();
    void select_folded_compact_pair(bool);
    void enable_folded_gray_pair();
    void select_folded_gray_pair(bool);
    void enable_folded_wide_warp();
    void select_folded_wide_warp(bool);
    void enable_folded_ring_pair();
    void select_folded_ring_pair(bool);
    bool intermediates_materialized()const{return intermediates_materialized_;}
    void enable_bank_tiles();
    void select_bank_tiles(bool);
    void enable_operand_lookahead();
    void select_operand_lookahead(bool);
    void enable_warp64_pair();
    void select_warp64_pair(bool);
    void enable_full_k_tiles();
    void select_full_k_tiles(bool);
    void enable_shared_tiles();
    void select_shared_tiles(bool);
    // Rebind borrowed matrices of the same geometry; no allocation or CUDA work.
    void rebind(Matrix gate, Matrix up, Matrix down);
    const Buffers &buffers() const { return buffers_; }
    int hidden_features() const { return gate_.columns; }
    int intermediate_features() const { return gate_.rows; }
    size_t half_weight_capacity() const {
        const size_t pair=2*size_t(gate_.rows)*gate_.columns,down=size_t(down_.rows)*down_.columns;
        return 2*(pair>down?pair:down);
    }
    size_t workspace_bytes() const { return workspace_bytes_; }
    bool owned_provider() const { return provider_ == Provider::OwnedStaged || provider_ == Provider::OwnedVector; }
    int multiprocessors() const { return caps_.multiprocessors; }
    unsigned launches(int tokens=0) const {
        if(folded_selected_&&tokens>=128&&tokens%128==0&&gate_.rows%64==0)return down_splits_>1?6:5;
        if(provider_==Provider::OwnedStaged && tokens>1 && tokens<64){
            unsigned extra=0,removed=0;
            if(small_direct32_ && tokens==32 && caps_.major==7 && caps_.minor==0)
                for(Matrix w:{gate_,up_,down_})
                    extra+=w.quant==Quant::Q6K && w.columns>=512 && (w.rows+63)/64<2*caps_.multiprocessors;
            if(small_direct32_ && tokens==32 && caps_.major==7 && caps_.minor==0 && gate_.quant==Quant::Q4K)
                removed=2+(down_.quant==Quant::Q6K);
            if(segmented_pair_&&tokens==32&&down_.quant==Quant::Q4K)++extra;
            return 6+extra-removed;
        }
        return provider_ == Provider::OwnedVector ? 4 : provider_ == Provider::Tiled ? 3 :
               (down_splits_ > 1 ? 8 : 7);
    }
private:
    bool tiny_batches_=false;
    bool tiny_row_ffn_=false;
    bool tiny_meta_ffn_=false;
    DenseGemm *pp_gemm_=nullptr;
    std::unique_ptr<PpWeightOverlap> pp_weight_overlap_;
    bool pp_weight_overlap_enabled_=false;
    void execute_pp_gemm(const float *,int,cudaStream_t,bool);
    Matrix gate_, up_, down_;
    int max_tokens_;
    int down_splits_;
    Provider provider_;
    Capabilities caps_;
    Buffers buffers_;
    size_t workspace_bytes_ = 0;
    bool small_direct32_ = false,segmented_pair_=false,wide_tiles_=false,whole_k_pair_=false,tensor_bias_pair_=false;
    void *tiled_pair_=nullptr;size_t tiled_pair_bytes_=0;cudaStream_t tiled_stream_=nullptr;
    void *group_pair_=nullptr;size_t group_pair_bytes_=0;double group_pair_cold_ms_=0;
    cudaStream_t group_stream_=nullptr;bool group_pair_selected_=false;
    bool balanced_configured_=false,balanced_selected_=false;
    bool pipeline_configured_=false,pipeline_selected_=false;
    bool raster_configured_=false,raster_selected_=false;
    bool joint_configured_=false,joint_selected_=false,token256_pair_=false;
    bool folded_ring_configured_=false,folded_ring_selected_=false;
    bool folded_wide_warp_configured_=false,folded_wide_warp_selected_=false;
    bool folded_gray_configured_=false,folded_gray_selected_=false;
    bool folded_compact_configured_=false,folded_compact_selected_=false;
    bool folded_configured_=false,folded_selected_=false,intermediates_materialized_=true;
    bool bank_configured_=false,bank_selected_=false;
    bool operand_lookahead_configured_=false,operand_lookahead_selected_=false;
    bool warp64_configured_=false,warp64_selected_=false;
    bool full_k_configured_=false,full_k_selected_=false;
    bool shared_configured_=false,shared_selected_=false;
};
void tiled_pair(Matrix, Matrix, const float *, float *, float *, int, Tile, cudaStream_t);
void tiled_linear(Matrix, const float *, float *, int, Tile, cudaStream_t);
void decode_pair(Matrix, Matrix, void *, cudaStream_t);
void decode_pair_interleaved(Matrix,Matrix,void *,cudaStream_t);
void decode_linear(Matrix, void *, cudaStream_t);
void to_half(const float *, void *, size_t, cudaStream_t);
void swiglu(const float *, const float *, float *, size_t, cudaStream_t);
void dense_pair(const void *, const void *, float *, float *, int, int, int, cudaStream_t);
void dense_linear(const void *, const void *, float *, int, int, int, cudaStream_t);
void dense_split(const void *, const void *, float *, float *, int, int, int, int, cudaStream_t);
void vector_quantize(const float *, void *, int, cudaStream_t);
void vector_pair(Matrix, Matrix, const void *, float *, float *, float *, cudaStream_t);
void vector_linear(Matrix, const void *, float *, cudaStream_t);
Matrix prepare_q4_pairs(Matrix, void *, cudaStream_t);
size_t prepared_q6_codes_bytes(Matrix);
Matrix prepare_q6_codes(Matrix, void *, cudaStream_t);
void check_cuda(cudaError_t, const char *operation);
} // namespace strata::qwen35::native
