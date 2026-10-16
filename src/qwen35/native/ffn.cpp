#include "strata/qwen35/native/tiny_batch.hpp"
#include "strata/qwen35/native/tiny_rows.hpp"
#include "strata/qwen35/native/tiny_meta.hpp"
#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "strata/qwen35/native/projection.hpp"
#include "small_prefill_half_raw.hpp"
#include "small_prefill_q4_grouped.hpp"
#include "small_prefill_q4_tile_codes.hpp"
#include "small_prefill_q4_group_codes.hpp"
#include <algorithm>
#include "strata/qwen35/native/weight_overlap.hpp"
#include <chrono>
#include <string>

namespace strata::qwen35::native {
void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
namespace {
void release(Buffers &b) {
    // Up is a view into the second half of the owned Gate/Up allocation.
    cudaFree(b.gate); cudaFree(b.activation); cudaFree(b.output);
    cudaFree(b.half_weights); cudaFree(b.half_input);
    cudaFree(b.down_partials);
    cudaFree(b.quant_input);
    b = {};
}
}
Capabilities capabilities(int device) {
    cudaDeviceProp p{};
    check_cuda(cudaGetDeviceProperties(&p, device), "query native device capabilities");
    if (p.major < 7) throw std::runtime_error("native FP16 Tensor Core path requires CC >= 7.0");
    int clusters = 0;
    check_cuda(cudaDeviceGetAttribute(&clusters, cudaDevAttrClusterLaunch, device), "query clusters");
    return {p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin,
            p.major >= 8, clusters != 0, p.major == 10};
}
Tile choose_tile(const Capabilities &caps, int rows, int tokens) {
    // The two compiled geometries have exact resource bounds. Crossovers need
    // measured service curves; no GPU name or model-specific N/K whitelist.
    if (rows <= 0 || tokens <= 0) throw std::invalid_argument("invalid native tile problem");
    const size_t shared = (std::max)(size_t((2 * 64 + 64) * 80 * 2), size_t(2 * 64 * 64 * 4));
    return tokens >= 64 && caps.shared_per_block >= shared ? Tile{64, 64, 64} : Tile{64, 32, 32};
}
Ffn::Ffn(Matrix gate, Matrix up, Matrix down, int max_tokens, Provider provider, int down_splits)
    : gate_(gate), up_(up), down_(down), max_tokens_(max_tokens), down_splits_(down_splits), provider_(provider) {
    matrix_bytes(gate,false,false,true); matrix_bytes(up,false,false,true); matrix_bytes(down,true);
    if((gate.layout==BlockLayout::Q4Tiles||up.layout==BlockLayout::Q4Tiles)&&
       (gate.layout!=up.layout||provider==Provider::Tiled))throw std::invalid_argument("tiled FFN inventory requires compatible paired provider");
    if(down.layout==BlockLayout::Q6Codes && (provider!=Provider::OwnedStaged || max_tokens!=32))
        throw std::invalid_argument("Q6 code FFN admission currently supports independent PP32 only");
    if (gate.rows != up.rows || gate.columns != up.columns ||
        down.columns != gate.rows || down.rows != gate.columns || max_tokens < 1 || max_tokens > 8192)
        throw std::invalid_argument("invalid native Dense FFN geometry");
    if ((down_splits != 1 && down_splits != 4 && down_splits != 8) ||
        (down_splits != 1 && (provider != Provider::OwnedStaged || down.columns % (down_splits * 64))))
        throw std::invalid_argument("invalid native Dense FFN reduction schedule");
    if (provider == Provider::OwnedVector && (max_tokens != 1 || gate.quant != up.quant))
        throw std::invalid_argument("native vector provider needs one token and equal Gate/Up formats");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "native device");
    caps_ = capabilities(device);
    try {
        auto allocate = [this](auto **pointer, size_t bytes) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer), bytes), "native FFN allocation");
            workspace_bytes_ += bytes;
        };
        const size_t hidden = size_t(gate.rows) * max_tokens;
        allocate(&buffers_.gate, hidden * 8);
        buffers_.up=buffers_.gate+hidden;
        allocate(&buffers_.activation, hidden * 4);
        allocate(&buffers_.output, size_t(down.rows) * max_tokens * 4);
        if (down_splits > 1)
            allocate(&buffers_.down_partials, size_t(down_splits) * down.rows * max_tokens * 4);
        if (provider == Provider::OwnedVector)
            allocate(&buffers_.quant_input,
                     size_t((std::max)(gate.columns, down.columns) / 32) * VectorActivationBytes);
        else if (provider != Provider::Tiled) {
            const size_t down_elements = size_t(down.rows) * down.columns;
            const size_t weights = (std::max)(2 * size_t(gate.rows) * gate.columns, down_elements);
            allocate(&buffers_.half_weights, weights * 2);
            allocate(&buffers_.half_input,
                     size_t((std::max)(gate.columns, down.columns)) * max_tokens * 2);
        }
    } catch (...) {
        release(buffers_);
        throw;
    }
}
Ffn::~Ffn() {
    pp_weight_overlap_.reset();
    release(buffers_);
}
void Ffn::enable_wide_tiles(){
    if(provider_!=Provider::OwnedStaged||max_tokens_<128||caps_.shared_per_block<size_t((64+128)*80*2))
        throw std::invalid_argument("wide FFN requires staged operands,128 tokens and30KiB shared memory");
    configure_dense_wide();
    wide_tiles_=true;
}
void Ffn::enable_balanced_tiles(){
    enable_wide_tiles();configure_dense_balanced();
    balanced_configured_=true;balanced_selected_=true;
}
void Ffn::select_balanced_tiles(bool selected){
    if(!balanced_configured_)throw std::invalid_argument("balanced tile selection requires cold configuration");
    balanced_selected_=selected;
}
void Ffn::enable_pipeline_tiles(){
    if(raster_selected_)throw std::invalid_argument("pipeline and raster selections are exclusive");
    enable_balanced_tiles();configure_dense_pipeline();
    pipeline_configured_=true;pipeline_selected_=true;
}
void Ffn::select_pipeline_tiles(bool selected){
    if(!pipeline_configured_||!balanced_selected_)throw std::invalid_argument("pipeline selection requires balanced cold configuration");
    if(selected&&raster_selected_)throw std::invalid_argument("pipeline and raster selections are exclusive");
    pipeline_selected_=selected;
}
void Ffn::enable_raster_tiles(){
    if(pipeline_selected_)throw std::invalid_argument("raster and pipeline selections are exclusive");
    enable_balanced_tiles();configure_dense_raster();
    raster_configured_=true;raster_selected_=true;
}
void Ffn::select_raster_tiles(bool selected){
    if(!selected&&joint_selected_)throw std::invalid_argument("disable joint tiles before raster tiles");
    if(!raster_configured_||!balanced_selected_|| (selected&&pipeline_selected_))
        throw std::invalid_argument("raster selection requires exclusive balanced cold configuration");
    raster_selected_=selected;
}
void Ffn::enable_joint_tiles(){
    if(caps_.shared_per_block<size_t(256*72*2))throw std::invalid_argument("joint FFN requires36KiB shared memory");
    enable_raster_tiles();configure_dense_joint();
    joint_configured_=true;joint_selected_=true;
}
void Ffn::select_joint_tiles(bool selected){
    if(!selected&&(shared_selected_||full_k_selected_||token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_))throw std::invalid_argument("disable joint candidates before joint tiles");
    if(!joint_configured_||!raster_selected_||pipeline_selected_)
        throw std::invalid_argument("joint selection requires exclusive raster cold configuration");
    joint_selected_=selected;
}
void Ffn::enable_folded_pair(){
    if(folded_configured_||!joint_selected_||bank_selected_||shared_selected_||full_k_selected_||token256_pair_||warp64_selected_||operand_lookahead_selected_||max_tokens_<128)
        throw std::invalid_argument("folded pair requires exclusive long joint FFN");
    configure_dense_folded();folded_configured_=true;folded_selected_=true;
}
void Ffn::select_folded_pair(bool selected){
    if((!selected&&(folded_compact_selected_||folded_wide_warp_selected_||folded_ring_selected_))||!folded_configured_||!joint_selected_||bank_selected_||shared_selected_||full_k_selected_||token256_pair_||warp64_selected_||operand_lookahead_selected_)
        throw std::invalid_argument("folded selection requires its exclusive cold configuration");
    folded_selected_=selected;
}
void Ffn::enable_folded_compact_pair(){
    if(!folded_selected_||folded_ring_selected_||folded_wide_warp_selected_||folded_compact_configured_)
        throw std::invalid_argument("compact folded pair requires cold folded configuration");
    configure_dense_folded_compact();folded_compact_configured_=true;folded_compact_selected_=true;
}
void Ffn::select_folded_compact_pair(bool selected){
    if((selected&&(folded_wide_warp_selected_||folded_ring_selected_))||(!selected&&folded_gray_selected_)||!folded_compact_configured_||!folded_selected_)
        throw std::invalid_argument("compact folded selection requires folded configuration");
    folded_compact_selected_=selected;
}
void Ffn::enable_folded_gray_pair(){
    if(!folded_compact_selected_||folded_gray_configured_)
        throw std::invalid_argument("gray folded pair requires compact cold configuration");
    configure_dense_folded_gray();folded_gray_configured_=true;folded_gray_selected_=true;
}
void Ffn::select_folded_gray_pair(bool selected){
    if((selected&&(folded_wide_warp_selected_||folded_ring_selected_))||!folded_gray_configured_||!folded_selected_)
        throw std::invalid_argument("gray folded selection requires folded configuration");
    folded_gray_selected_=selected;folded_compact_selected_=selected;
}
void Ffn::enable_folded_wide_warp(){
    if(!folded_selected_||folded_ring_selected_||folded_compact_selected_||folded_wide_warp_configured_)
        throw std::invalid_argument("wide warp requires exclusive folded cold configuration");
    configure_dense_folded_widewarp();folded_wide_warp_configured_=true;folded_wide_warp_selected_=true;
}
void Ffn::select_folded_wide_warp(bool selected){
    if(!folded_wide_warp_configured_||!folded_selected_||(selected&&(folded_compact_selected_||folded_ring_selected_)))
        throw std::invalid_argument("wide warp selection requires exclusive folded configuration");
    folded_wide_warp_selected_=selected;
}
void Ffn::enable_folded_ring_pair(){
    if(!folded_selected_||folded_compact_selected_||folded_wide_warp_selected_||folded_ring_configured_)
        throw std::invalid_argument("folded ring requires exclusive folded cold configuration");
    configure_dense_folded_ring();folded_ring_configured_=true;folded_ring_selected_=true;
}
void Ffn::select_folded_ring_pair(bool selected){
    if(!folded_ring_configured_||!folded_selected_||(selected&&(folded_compact_selected_||folded_wide_warp_selected_)))
        throw std::invalid_argument("folded ring selection requires exclusive folded configuration");
    folded_ring_selected_=selected;
}
void Ffn::enable_bank_tiles(){
    if(folded_selected_||bank_configured_||!joint_selected_||shared_selected_||full_k_selected_||token256_pair_||warp64_selected_||operand_lookahead_selected_||max_tokens_<128)
        throw std::invalid_argument("bank-layout requires exclusive long joint FFN");
    configure_dense_bank();bank_configured_=true;bank_selected_=true;
}
void Ffn::select_bank_tiles(bool selected){
    if(folded_selected_||!bank_configured_||!joint_selected_||shared_selected_||full_k_selected_||token256_pair_||warp64_selected_||operand_lookahead_selected_)
        throw std::invalid_argument("bank-layout selection requires its exclusive cold configuration");
    bank_selected_=selected;
}
void Ffn::enable_operand_lookahead(){
    if(folded_selected_||bank_selected_||operand_lookahead_configured_||!joint_selected_||token256_pair_||warp64_selected_||shared_selected_||full_k_selected_||max_tokens_<128)
        throw std::invalid_argument("operand lookahead requires exclusive long joint FFN admission");
    configure_dense_operand_lookahead();operand_lookahead_configured_=true;operand_lookahead_selected_=true;
}
void Ffn::select_operand_lookahead(bool selected){
    if(folded_selected_||bank_selected_||!operand_lookahead_configured_||!joint_selected_||token256_pair_||warp64_selected_||shared_selected_||full_k_selected_)
        throw std::invalid_argument("operand lookahead requires exclusive configured joint FFN");
    operand_lookahead_selected_=selected;
}
void Ffn::enable_warp64_pair(){
    if(warp64_configured_||folded_selected_||bank_selected_||operand_lookahead_selected_||!joint_selected_||token256_pair_||shared_selected_||full_k_selected_||max_tokens_<128)
        throw std::invalid_argument("warp64 requires exclusive long joint FFN admission");
    configure_dense_warp64();warp64_configured_=true;warp64_selected_=true;
}
void Ffn::select_warp64_pair(bool selected){
    if(!warp64_configured_||folded_selected_||bank_selected_||operand_lookahead_selected_||!joint_selected_||token256_pair_||shared_selected_||full_k_selected_)
        throw std::invalid_argument("warp64 selection requires exclusive configured joint FFN");
    warp64_selected_=selected;
}
void Ffn::enable_token256_pair(){
    if(token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_||!joint_selected_||shared_selected_||full_k_selected_||max_tokens_<256)
        throw std::invalid_argument("token256 pair requires exclusive long joint FFN admission");
    configure_dense_token256();token256_pair_=true;
}
void Ffn::enable_full_k_tiles(){
    if(shared_selected_||token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_)throw std::invalid_argument("full K and shared selections are exclusive");
    enable_joint_tiles();configure_dense_full_k();
    full_k_configured_=true;full_k_selected_=true;
}
void Ffn::select_full_k_tiles(bool selected){
    if(!full_k_configured_||!joint_selected_||pipeline_selected_||(selected&&(shared_selected_||token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_)))
        throw std::invalid_argument("full K selection requires exclusive joint cold configuration");
    full_k_selected_=selected;
}
void Ffn::enable_shared_tiles(){
    if(full_k_selected_||token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_)throw std::invalid_argument("shared and full K selections are exclusive");
    enable_joint_tiles();configure_dense_shared();
    shared_configured_=true;shared_selected_=true;
}
void Ffn::select_shared_tiles(bool selected){
    if(!shared_configured_||!joint_selected_||pipeline_selected_||(selected&&(full_k_selected_||token256_pair_||warp64_selected_||folded_selected_||bank_selected_||operand_lookahead_selected_)))
        throw std::invalid_argument("shared selection requires exclusive joint cold configuration");
    shared_selected_=selected;
}
void Ffn::enable_small_direct32(bool segmented_pair) {
    const auto supported=[](Matrix m){
        return (m.quant==Quant::Q4K && (m.layout==BlockLayout::Q4Pairs||m.layout==BlockLayout::Q4Tiles)) ||
               (m.quant==Quant::Q6K && (m.layout==BlockLayout::Gguf || m.layout==BlockLayout::Q6Codes));
    };
    if(provider_!=Provider::OwnedStaged || max_tokens_<32 || !supported(gate_) ||
       !supported(up_) || !supported(down_))
        throw std::invalid_argument("direct FFN32 requires owned staging and compatible Q4/Q6 layouts");
    if(down_.layout==BlockLayout::Q6Codes && (caps_.major!=7 || caps_.minor!=0 || gate_.quant!=Quant::Q4K))
        throw std::invalid_argument("Q6 code FFN32 service curve is SM70 Q4-pair only");
    if(caps_.major==7 && caps_.minor==0 && (gate_.quant!=up_.quant ||
       small_prefill_half_input_bytes((std::max)(gate_.columns,down_.columns),32)*(gate_.quant==Quant::Q4K && down_.quant==Quant::Q6K?2:1)+
       size_t(down_.layout==BlockLayout::Q6Codes?3:2)*32*(std::max)(gate_.rows,down_.rows)*4>half_weight_capacity()))
        throw std::invalid_argument("prepared FFN input format or arena capacity mismatch");
    configure_small_prefill_direct();
    if(caps_.major==7 && caps_.minor==0 && gate_.quant==Quant::Q4K)configure_small_prefill_q4_pair();
    small_direct32_=true;segmented_pair_=segmented_pair;
}
void Ffn::enable_q4_tiled_pair(cudaStream_t stream){
    if(tiled_pair_||group_pair_||!small_direct32_||!segmented_pair_||caps_.major!=7||caps_.minor!=0||gate_.quant!=Quant::Q4K||up_.quant!=Quant::Q4K||gate_.rows%64)
        throw std::invalid_argument("tiled Q4 requires configured segmented PP32 and stable raw-pair weights");
    const size_t bytes=2*matrix_bytes(gate_),capacity=half_weight_capacity();
    const size_t transient=2*small_prefill_half_input_bytes((std::max)(gate_.columns,down_.columns),32)+size_t(3)*32*(std::max)(gate_.rows,down_.rows)*4;
    if(bytes>capacity||transient>capacity-bytes)throw std::invalid_argument("tiled Q4 view exceeds idle arena capacity");
    auto *base=static_cast<unsigned char *>(buffers_.half_weights)+capacity-bytes;
    prepare_q4_tile_codes(gate_,base,stream);prepare_q4_tile_codes(up_,base+bytes/2,stream);
    check_cuda(cudaStreamSynchronize(stream),"cold tiled Q4 view completion");
    tiled_pair_=base;tiled_pair_bytes_=bytes;tiled_stream_=stream;
}
void Ffn::enable_whole_k_q4_pair(bool tensor_bias){
    if(!small_direct32_||!segmented_pair_||tiled_pair_||caps_.major!=7||caps_.minor!=0||
       gate_.quant!=Quant::Q4K||up_.quant!=Quant::Q4K||gate_.layout!=BlockLayout::Q4Pairs||up_.layout!=BlockLayout::Q4Pairs)
        throw std::invalid_argument("whole K pair requires configured raw Q4 PP32 inventory");
    whole_k_pair_=true;tensor_bias_pair_=tensor_bias;
}
void Ffn::enable_q4_group_pair(cudaStream_t stream){
    if(group_pair_||tiled_pair_||!whole_k_pair_||!tensor_bias_pair_||gate_.rows%64)
        throw std::invalid_argument("group Q4 canary requires configured550 raw whole K pair");
    const size_t bytes=2*matrix_bytes(gate_),capacity=half_weight_capacity();
    const size_t transient=2*small_prefill_half_input_bytes((std::max)(gate_.columns,down_.columns),32)+size_t(3)*32*(std::max)(gate_.rows,down_.rows)*4;
    if(bytes>capacity||transient>capacity-bytes)throw std::invalid_argument("group Q4 view exceeds idle arena capacity");
    const auto begin=std::chrono::steady_clock::now();
    auto *base=static_cast<unsigned char *>(buffers_.half_weights)+capacity-bytes;
    prepare_q4_group_codes(gate_,base,stream);prepare_q4_group_codes(up_,base+bytes/2,stream);
    check_cuda(cudaStreamSynchronize(stream),"cold group Q4 view completion");
    group_pair_=base;group_pair_bytes_=bytes;group_stream_=stream;group_pair_selected_=true;
    group_pair_cold_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
void Ffn::verify_q4_group_pair(cudaStream_t stream){
    if(!group_pair_||stream!=group_stream_)throw std::invalid_argument("group Q4 verification lease mismatch");
    const auto *base=static_cast<const unsigned char *>(group_pair_);const size_t bytes=group_pair_bytes_/2;
    verify_q4_group_codes(gate_,{base,gate_.rows,gate_.columns,bytes},buffers_.gate,stream);
    verify_q4_group_codes(up_,{base+bytes,up_.rows,up_.columns,bytes},buffers_.gate,stream);
}
void Ffn::select_q4_group_pair(bool enabled){
    if(!group_pair_)throw std::invalid_argument("group Q4 selection requires cold initialization");
    group_pair_selected_=enabled;
}
void Ffn::rebind(Matrix gate, Matrix up, Matrix down) {
    matrix_bytes(gate,false,false,true);matrix_bytes(up,false,false,true);matrix_bytes(down,true);
    if((gate.layout==BlockLayout::Q4Tiles||up.layout==BlockLayout::Q4Tiles)&&
       (gate.layout!=up.layout||provider_==Provider::Tiled))throw std::invalid_argument("rebound tiled FFN requires compatible paired provider");
    if((tiled_pair_||group_pair_)&&(gate.data!=gate_.data||up.data!=up_.data))throw std::invalid_argument("cold Q4 view pins raw weight identity");
    if(down.layout==BlockLayout::Q6Codes && (provider_!=Provider::OwnedStaged || max_tokens_!=32))
        throw std::invalid_argument("rebound Q6 code FFN requires independent PP32 admission");
    if(gate.rows!=gate_.rows || gate.columns!=gate_.columns || up.rows!=up_.rows || up.columns!=up_.columns ||
       down.rows!=down_.rows || down.columns!=down_.columns || (provider_==Provider::OwnedVector && gate.quant!=up.quant))
        throw std::invalid_argument("shared FFN weight geometry changed");
    if(small_direct32_ && (gate.quant!=gate_.quant || up.quant!=up_.quant ||
       gate.layout!=gate_.layout || up.layout!=up_.layout ||
       ((down.layout==BlockLayout::Q6Codes||down_.layout==BlockLayout::Q6Codes)&&down.layout!=down_.layout)))
        throw std::invalid_argument("direct FFN32 weight format changed");
    gate_=gate;up_=up;down_=down;
}
void Ffn::execute(const float *input, int tokens, cudaStream_t stream,bool diagnostic_projections) {
    if (!input || tokens < 1 || tokens > max_tokens_)
        throw std::invalid_argument("invalid native FFN invocation");
    intermediates_materialized_=true;
    if(tiled_pair_&&(tokens!=32||stream!=tiled_stream_))throw std::invalid_argument("tiled Q4 view requires its PP32 stream lease");
    if(group_pair_&&(tokens!=32||stream!=group_stream_))throw std::invalid_argument("group Q4 canary requires its PP32 stream lease");
    if(gate_.layout==BlockLayout::Q4Tiles&&tokens!=1&&
       (tokens!=32||!small_direct32_||!segmented_pair_||caps_.major!=7||caps_.minor!=0))
        throw std::invalid_argument("tiled FFN inventory has PP32 and own vector operators only");
    if(down_.layout==BlockLayout::Q6Codes && (!small_direct32_ || tokens!=32))
        throw std::invalid_argument("Q6 code FFN requires configured PP32 service");
    if(pp_gemm_&&tokens>=128){execute_pp_gemm(input,tokens,stream,diagnostic_projections);return;}
    if (provider_ == Provider::OwnedVector) {
        vector_quantize(input, buffers_.quant_input, gate_.columns, stream);
        vector_pair(gate_, up_, buffers_.quant_input, buffers_.gate, buffers_.up, buffers_.activation, stream);
        vector_quantize(buffers_.activation, buffers_.quant_input, down_.columns, stream);
        vector_linear(down_, buffers_.quant_input, buffers_.output, stream);
    } else if(tiny_batches_&&(tokens==2||tokens==3)){
        void *codes=buffers_.half_input;
        const auto pair=tiny_row_ffn_&&gate_.quant==Quant::Q4K&&tokens==2?tiny_row_pair:tiny_batch_pair;
        const auto linear=tiny_meta_ffn_&&tokens==3&&down_.quant==Quant::Q4K?tiny_meta_linear:
            tiny_row_ffn_&&down_.quant==Quant::Q4K?tiny_row_linear:tiny_batch_linear;
        tiny_batch_quantize(input,codes,gate_.columns,tokens,stream);
        pair(gate_,up_,codes,buffers_.gate,buffers_.up,buffers_.activation,tokens,stream);
        tiny_batch_quantize(buffers_.activation,codes,down_.columns,tokens,stream);
        linear(down_,codes,buffers_.output,tokens,stream);
    } else if (provider_ == Provider::OwnedStaged && tokens>1 && tokens<64) {
        // Forty bytes per32 codes fit inside the existing two-byte input arena.
        const bool direct=small_direct32_ && tokens==32;
        const bool half=direct && caps_.major==7 && caps_.minor==0;
        const auto quantize=direct?small_prefill_vector_quantize:small_prefill_quantize;
        // The decoded-weight arena is otherwise idle in small PP. Reuse it for
        // the slightly larger half-code input; do not allocate another arena.
        void *codes=half?buffers_.half_weights:buffers_.half_input;
        const size_t encoded=half?small_prefill_half_input_bytes((std::max)(gate_.columns,down_.columns),32):0;
        const bool fused=half && gate_.quant==Quant::Q4K && down_.quant==Quant::Q6K;
        void *activation_codes=fused?static_cast<unsigned char *>(codes)+encoded:codes;
        const size_t transient=encoded*(fused?2:1);
        RawHalfScratch partials{half?static_cast<unsigned char *>(codes)+transient:nullptr,half?half_weight_capacity()-transient-tiled_pair_bytes_-group_pair_bytes_:0};
        auto project=[&](Matrix w,float *out,const void *prepared){
            if(half)small_prefill_half_raw_linear(w,prepared,out,tokens,stream,caps_.multiprocessors,partials);
            else (direct?small_prefill_direct_blocked_linear:small_prefill_linear)(w,prepared,out,tokens,stream,caps_.multiprocessors);
        };
        if(half)small_prefill_half_raw_quantize(input,codes,gate_.quant,gate_.columns,tokens,stream);
        else quantize(input,codes,gate_.columns,tokens,stream);
        if(group_pair_selected_){
            const auto *base=static_cast<const unsigned char *>(group_pair_);const size_t bytes=group_pair_bytes_/2;
            small_prefill_q4_group_pair({base,gate_.rows,gate_.columns,bytes},{base+bytes,up_.rows,up_.columns,bytes},codes,buffers_.gate,buffers_.up,buffers_.activation,stream,diagnostic_projections,fused?activation_codes:nullptr);
        }else if(gate_.layout==BlockLayout::Q4Tiles){
            const size_t bytes=matrix_bytes(gate_,false,false,true);
            small_prefill_q4_tiled_pair({gate_.data,gate_.rows,gate_.columns,bytes},{up_.data,up_.rows,up_.columns,bytes},codes,buffers_.gate,buffers_.up,buffers_.activation,stream,diagnostic_projections,fused?activation_codes:nullptr,caps_.multiprocessors);
        }else if(tiled_pair_){
            const auto *base=static_cast<const unsigned char *>(tiled_pair_);const size_t bytes=tiled_pair_bytes_/2;
            small_prefill_q4_tiled_pair({base,gate_.rows,gate_.columns,bytes},{base+bytes,up_.rows,up_.columns,bytes},codes,buffers_.gate,buffers_.up,buffers_.activation,stream,diagnostic_projections,fused?activation_codes:nullptr,caps_.multiprocessors);
        }else if(half && gate_.quant==Quant::Q4K)
            small_prefill_q4_pair_linear(gate_,up_,codes,buffers_.gate,buffers_.up,buffers_.activation,tokens,stream,diagnostic_projections,fused?activation_codes:nullptr,segmented_pair_&&!whole_k_pair_?caps_.multiprocessors:0,whole_k_pair_,tensor_bias_pair_);
        else {project(gate_,buffers_.gate,codes);project(up_,buffers_.up,codes);
            swiglu(buffers_.gate,buffers_.up,buffers_.activation,size_t(gate_.rows)*tokens,stream);
        }
        if(!fused){
            if(half)small_prefill_half_raw_quantize(buffers_.activation,activation_codes,down_.quant,down_.columns,tokens,stream);
            else quantize(buffers_.activation,activation_codes,down_.columns,tokens,stream);
        }
        if(half&&segmented_pair_&&down_.quant==Quant::Q4K&&q4_single_partitions(down_)==2)
            small_prefill_q4_single_linear(down_,activation_codes,buffers_.output,32,stream,partials.data,partials.bytes);
        else if(half&&segmented_pair_&&down_.quant==Quant::Q6K&&down_.layout==BlockLayout::Gguf&&down_.columns%512==0)
            small_prefill_q6_raw_prefetch_linear(down_,activation_codes,buffers_.output,stream,partials);
        else project(down_,buffers_.output,activation_codes);
    } else if (provider_ == Provider::OwnedStaged) {
        const bool wide=wide_tiles_&&tokens>=128;
        const bool balanced=balanced_selected_&&wide,pipeline=pipeline_selected_&&balanced,raster=raster_selected_&&balanced,joint=joint_selected_&&raster,bank=bank_selected_&&joint,shared=shared_selected_&&joint,full_k=full_k_selected_&&joint;
        const auto pair=bank?dense_bank_pair:operand_lookahead_selected_&&joint?dense_operand_lookahead_pair:warp64_selected_&&joint?dense_warp64_pair:token256_pair_&&tokens>=256?dense_token256_pair:full_k?dense_full_k_pair:shared?dense_shared_pair:joint?dense_joint_pair:raster?dense_raster_pair:pipeline?dense_pipeline_pair:balanced?dense_balanced_pair:wide?dense_wide_pair:dense_pair;
        const auto linear=bank?dense_bank_linear:full_k?dense_full_k_linear:shared?dense_shared_linear:joint?dense_joint_linear:raster?dense_raster_linear:pipeline?dense_pipeline_linear:balanced?dense_balanced_linear:wide?dense_wide_linear:dense_linear;
        const auto split=bank?dense_bank_split:full_k?dense_full_k_split:shared?dense_shared_split:joint?dense_joint_split:raster?dense_raster_split:pipeline?dense_pipeline_split:balanced?dense_balanced_split:wide?dense_wide_split:dense_split;
        decode_pair(gate_, up_, buffers_.half_weights, stream);
        to_half(input, buffers_.half_input, size_t(gate_.columns) * tokens, stream);
        const bool folded=folded_selected_&&joint&&tokens%128==0&&gate_.rows%64==0;
        if(folded){
            const auto fused_pair=folded_ring_selected_?dense_folded_ring_pair:folded_wide_warp_selected_?dense_folded_widewarp_pair:folded_gray_selected_?dense_folded_gray_pair:folded_compact_selected_?dense_folded_compact_pair:dense_folded_pair;
            fused_pair(buffers_.half_weights,buffers_.half_input,buffers_.gate,buffers_.up,buffers_.activation,
                              gate_.rows,gate_.columns,tokens,diagnostic_projections,stream);
            intermediates_materialized_=diagnostic_projections;
        }else{
            pair(buffers_.half_weights,buffers_.half_input,buffers_.gate,buffers_.up,gate_.rows,gate_.columns,tokens,stream);
            swiglu(buffers_.gate,buffers_.up,buffers_.activation,size_t(gate_.rows)*tokens,stream);
        }
        decode_linear(down_,buffers_.half_weights,stream);
        if(intermediates_materialized_)to_half(buffers_.activation,buffers_.half_input,size_t(down_.columns)*tokens,stream);
        const void *down_input=intermediates_materialized_?buffers_.half_input:buffers_.activation;
        if (down_splits_ > 1)
            split(buffers_.half_weights, down_input, buffers_.output, buffers_.down_partials,
                        down_.rows, down_.columns, tokens, down_splits_, stream);
        else
            linear(buffers_.half_weights, down_input, buffers_.output,
                         down_.rows, down_.columns, tokens, stream);
    } else {
        tiled_pair(gate_, up_, input, buffers_.gate, buffers_.up, tokens,
                   choose_tile(caps_, gate_.rows, tokens), stream);
        swiglu(buffers_.gate, buffers_.up, buffers_.activation, size_t(gate_.rows) * tokens, stream);
        tiled_linear(down_, buffers_.activation, buffers_.output, tokens,
                     choose_tile(caps_, down_.rows, tokens), stream);
    }
}
} // namespace strata::qwen35::native
