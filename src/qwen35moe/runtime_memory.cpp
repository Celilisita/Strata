#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "mtp_state.hpp"
#include "moe_graph_state.hpp"
#include "sampling_state.hpp"
#include "head_readback.hpp"
#include "session_state.hpp"
#include "visual_state.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/attention_tile.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_dense_q4_row2.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace strata::qwen35moe {

void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

DeviceBuffer::DeviceBuffer(size_t bytes) { reset(bytes); }
DeviceBuffer::~DeviceBuffer() { if (data_ && owned_) cudaFree(data_); }
DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)),
      owned_(std::exchange(other.owned_, true)) {}
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (data_ && owned_) cudaFree(data_);
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        owned_ = std::exchange(other.owned_, true);
    }
    return *this;
}
void DeviceBuffer::reset(size_t bytes) {
    if (data_ && owned_) cuda_check(cudaFree(data_), "free device buffer");
    data_ = nullptr;
    bytes_ = 0;
    owned_ = true;
    if (bytes) {
        cuda_check(cudaMalloc(&data_, bytes), "allocate device buffer");
        bytes_ = bytes;
    }
}

void DeviceBuffer::view(const DeviceBuffer& owner, size_t offset, size_t bytes) {
    if (&owner == this || !owner.data() || offset > owner.bytes() || bytes > owner.bytes() - offset)
        throw std::invalid_argument("invalid device buffer view");
    reset();
    data_ = static_cast<uint8_t*>(owner.data()) + offset;
    bytes_ = bytes; owned_ = false;
}

void GpuWeight::load(const GgufFile& gguf, const TensorInfo* tensor) {
    if (!tensor) throw std::invalid_argument("missing Qwen35 dense weight");
    int elements = 0, block_bytes = 0;
    if (!block_geometry(tensor->type, elements, block_bytes) || tensor->elements() % elements)
        throw std::runtime_error("unsupported GGUF tensor: " + tensor->name);
    const size_t bytes = size_t(tensor->elements() / elements * block_bytes);
    data.reset(bytes);
    cuda_check(cudaMemcpy(data.data(), gguf.tensor_data(*tensor), bytes, cudaMemcpyHostToDevice),
               tensor->name.c_str());
    type = int(tensor->type);
}

void Workspace::allocate(int vocab, size_t embedding_row_bytes) {
    auto f32 = [](size_t n) { return n * sizeof(float); };
    hidden.reset(f32(2048)); norm.reset(f32(2048)); mixer.reset(f32(2048));
    q8_input.reset(kernels::native_q8_1_bytes(2048));
    q8_wide.reset(kernels::native_q8_1_bytes(4096));
    q8_small.reset(kernels::native_q8_1_bytes(512));
    qkv.reset(f32(8192)); z.reset(f32(4096)); alpha.reset(f32(32)); beta.reset(f32(32));
    gate.reset(f32(32)); raw.reset(f32(8192)); activated.reset(f32(8192));
    core.reset(f32(4096)); post.reset(f32(4096));
    qfull.reset(f32(8192)); q.reset(f32(4096)); k.reset(f32(512)); v.reset(f32(512));
    attn.reset(f32(4096)); gated.reset(f32(4096));
    step.reset(4 * sizeof(int32_t)); status.reset(sizeof(int32_t));
    qpos.reset(16 * sizeof(int32_t)); kpos.reset(2 * sizeof(int32_t));
    logits.reset(f32(257)); ids.reset(8 * sizeof(int32_t)); weights.reset(f32(8));
    parts.reset(f32(8 * 2048));
    gu_gate.reset(kernels::native_mmvq_weight_bytes(12, 2048, 512));
    gu_up.reset(kernels::native_mmvq_weight_bytes(12, 2048, 512));
    down.reset(kernels::native_mmvq_weight_bytes(14, 512, 2048));
    gate_out.reset(f32(512)); up_out.reset(f32(512)); expert_h.reset(f32(512));
    shared.reset(f32(2048)); ffn.reset(f32(2048));
    emb_row.reset(embedding_row_bytes); head_logits.reset(f32(size_t(vocab)));
}

Runtime::Runtime(const std::string& model_path, int max_context, bool enable_expert_cache, const kernels::SamplerParams* sampling)
    : gguf_(model_path), max_context_(max_context),
      enable_expert_cache_(enable_expert_cache) {
    int device = -1;
    cuda_check(cudaGetDevice(&device), "get CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "inspect CUDA device");
    if (properties.major < 7)
        throw std::runtime_error("Qwen3.6 kernels require CUDA compute capability 7.0 or newer");
    if (max_context < 1 || max_context > kAttentionMaxContext)
        throw std::invalid_argument("Qwen3.6 context capacity must be 1..262144");
    std::string error;
    if (!load_spec(gguf_, spec_, error)) throw std::runtime_error(error);
    if (uint64_t(max_context_) > spec_.context_length)
        throw std::invalid_argument("context exceeds GGUF trained context length");
    if (spec_.n_layers != 40 || spec_.n_embd != 2048 || spec_.n_expert != 256 ||
        spec_.n_expert_used != 8 || spec_.n_ff_exp != 512 || spec_.n_ff_shared != 512 ||
        spec_.n_head != 16 || spec_.n_head_kv != 2 || spec_.head_dim != 256 ||
        spec_.rope_dim != 64 || spec_.ssm_inner != 4096 || spec_.ssm_state != 128 ||
        spec_.ssm_group != 16 || spec_.ssm_rank != 32 || spec_.ssm_conv != 4)
        throw std::runtime_error("unsupported Qwen3.6-35B-A3B runtime geometry");
    const char* mtp = std::getenv("STRATA_QWEN35_MTP");
    if (mtp && std::string(mtp) != "0" && std::string(mtp) != "1")
        throw std::invalid_argument("MTP must be0 or1");
    mtp_enabled_ = mtp && std::string(mtp) == "1";
    expert_sources_ = spec_.layers;
    if (mtp_enabled_) {
        if (spec_.mtp.size() != 1) throw std::invalid_argument("this MTP provider requires one NextN decoder");
        expert_sources_.push_back(spec_.mtp[0].decoder);
    }
    embedding_row_bytes_ = kernels::native_mmvq_weight_bytes(int(spec_.embedding->type), 2048, 1);
    const char* gpu_head = std::getenv("STRATA_QWEN35_GPU_HEAD");
    gpu_head_ = !gpu_head || std::string(gpu_head) != "0";
    const char* tile = std::getenv("STRATA_QWEN35_TILE_ATTENTION");
    tile_attention_ = !tile || std::string(tile) != "0";
    if (!tile_attention_ && max_context_ > 1024)
        throw std::invalid_argument("long context requires tile attention");
    const char* prefill_tc = std::getenv("STRATA_QWEN35_PREFILL_TC");
    prefill_tc_ = !prefill_tc || std::string(prefill_tc) != "0";
    prefill_config_ = PrefillConfig::environment();
    const bool fp4_requested=nvfp4_expert_flag(std::getenv("STRATA_QWEN35_MOE_NVFP4"));
    const bool fp4_weights=nvfp4_main_stack(spec_);
    if (fp4_requested!=fp4_weights)
        throw std::invalid_argument("NVFP4 opt-in and converted main-stack GGUF must match");
    if (fp4_weights) {
        if(properties.major!=12 || properties.minor!=0 || mtp_enabled_ || !enable_expert_cache_)
            throw std::invalid_argument("NVFP4 first slice requires SM120, expert cache and MTP0");
        if(prefill_config_.prefix_stage || prefill_config_.expert_prefetch ||
           prefill_config_.gu_copy_wave || prefill_config_.expert_copy_runs || prefill_config_.expert_copy_stage ||
           prefill_config_.mapped_pp || prefill_config_.down_input_fused)
            throw std::invalid_argument("NVFP4 requires legacy untyped expert copy/fusion paths off");
        nvfp4_=std::make_unique<Nvfp4ExpertState>(gguf_,spec_);
    }
    if (const char* raw = std::getenv("STRATA_QWEN35_RETENTION_PLAN")) {
        const std::string value(raw);
        if (value != "0" && value != "1") throw std::invalid_argument("retention plan requires0 or1");
        expert_cache_.retention.enable_planning(value == "1");
    }
    if (const char* raw = std::getenv("STRATA_QWEN35_CACHE_RETENTION")) {
        const std::string value(raw);
        if (value != "0" && value != "1")
            throw std::invalid_argument("cache retention requires 0 or 1");
        expert_cache_.retention.enable(value == "1");
    }
    const char* routed = std::getenv("STRATA_QWEN35_ROUTED_TC");
    routed_tc_enabled_ = !routed || std::string(routed) != "0";
    const char* gathered = std::getenv("STRATA_QWEN35_GATHERED_GU");
    routed_gu_gathered_ = gathered && std::string(gathered) == "1";
    if(nvfp4_)routed_tc_enabled_=false; // The NVFP4 provider owns its typed grouped launches.
    const char* scan = std::getenv("STRATA_QWEN35_PREFILL_SCAN");
    prefill_scan_ = !scan || std::string(scan) != "0";
    const bool use_head_readback = HeadReadback::enabled(
        nvfp4_ && gpu_head_ && properties.major == 12 && properties.minor == 0);
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create Qwen35 stream");
    // Large configured contexts reserve actual request K/V at a drained
    // request boundary. Small contexts keep the accepted allocation policy.
    kv_capacity_ = (std::min)(max_context_, max_context_ > 16384 ? 256 : max_context_);
    load_weights();
    work_.allocate(int(spec_.n_vocab), embedding_row_bytes_);
    if (tile_attention_ && max_context_ > 256)
        attention_scratch_.reset(flash_attn_tile_scratch_bytes(kv_capacity_) * (max_context_ > 16384 ? 2 : 1));
    host_logits_.resize(spec_.n_vocab);
    initialize_mtp();
    initialize_sampling(sampling);
    initialize_visual();
    initialize_sessions();
    if (enable_expert_cache_) allocate_expert_cache();
    if (gpu_head_) {
        head_partials_.reset(((spec_.n_vocab + 4095) / 4096) * sizeof(HeadPick));
        head_pick_buffer_.reset(sizeof(HeadPick));
    }
    if (use_head_readback)
        head_readback_ = std::make_unique<HeadReadback>(stream_);
}

Runtime::~Runtime() {
    if(nvfp4_&&nvfp4_->tg_admission&&decode_phase_){try{finish_decode();}catch(...){}}
    gu_copy_.reset(); // Complete direct GU copies before cache/GGUF release.
    expert_prefetch_.reset(); // Drain copy reads before releasing GGUF/cache ownership.
    if (stream_) cudaStreamSynchronize(stream_);
    head_readback_.reset(); // The existing drain completes every pick D2H before host release.
    if (miss_service_) miss_service_->stop(); // CPU globals belong to nvfp4_; join even after a CUDA error.
    projection_mmq_.reset();
    batch_.down_mmq.reset();
    nvfp4_.reset(); // Ordered work drained; release control leases before stream/mapping.
    moe_graph_.reset();
    miss_service_.reset();
    host_readonly_.clear(); // Stream and CPU misses drained; GGUF mapping remains alive.
    mtp_.reset();
    dense_profile_.clear();
    if (diag_route_ready_) cudaEventDestroy(diag_route_ready_);
    for (const auto& group : prefill_events_)
        for (auto event : group) cudaEventDestroy(event);
    if (stream_) cudaStreamDestroy(stream_);
}

void Runtime::load_weights() {
    head_norm_.load(gguf_, spec_.output_norm);
    head_.load(gguf_, spec_.output);
    layers_.reserve(spec_.layers.size());
    for (const auto& source : spec_.layers) layers_.push_back(load_layer(source));
}

LayerWeights Runtime::load_layer(const LayerTensors& source) {
    LayerWeights w;
    w.attn_norm.load(gguf_, source.attn_norm);
    w.post_norm.load(gguf_, source.post_attn_norm);
    w.router.data.reset(size_t(257) * 2048 * sizeof(float));
    w.router.type = 0;
    cuda_check(cudaMemcpy(w.router.data.data(), gguf_.tensor_data(*source.router),
                          size_t(256) * 2048 * sizeof(float), cudaMemcpyHostToDevice), "router");
    cuda_check(cudaMemcpy(w.router.data.as<float>() + size_t(256) * 2048,
                          gguf_.tensor_data(*source.shared_router), 2048 * sizeof(float),
                          cudaMemcpyHostToDevice), "shared router");
    w.shared_gate.load(gguf_, source.shared_gate);
    w.shared_up.load(gguf_, source.shared_up);
    w.shared_down.load(gguf_, source.shared_down);
    if (source.recurrent) {
        w.qkv.load(gguf_, source.qkv); w.gate.load(gguf_, source.gate);
        w.alpha.load(gguf_, source.alpha); w.beta.load(gguf_, source.beta);
        w.conv.load(gguf_, source.conv); w.dt.load(gguf_, source.dt);
        w.a.load(gguf_, source.a); w.ssm_norm.load(gguf_, source.ssm_norm);
        w.ssm_out.load(gguf_, source.ssm_out);
        w.history.reset(size_t(8192) * 3 * sizeof(float));
        w.state.reset(size_t(128) * 32 * 128 * sizeof(float));
        cuda_check(cudaMemset(w.history.data(), 0, w.history.bytes()), "clear GDN history");
        cuda_check(cudaMemset(w.state.data(), 0, w.state.bytes()), "clear GDN state");
    } else {
        w.q.load(gguf_, source.q); w.k.load(gguf_, source.k);
        w.v.load(gguf_, source.v); w.out.load(gguf_, source.attn_out);
        w.q_norm.load(gguf_, source.q_norm); w.k_norm.load(gguf_, source.k_norm);
        w.k_cache.reset(size_t(kv_capacity_) * 512 * sizeof(uint16_t));
        w.v_cache.reset(size_t(kv_capacity_) * 512 * sizeof(uint16_t));
    }
    return w;
}

void Runtime::project(const GpuWeight& weight, const void* input_q8, float* output,
                      int n_in, int n_out) {
    if (!kernels::native_mmvq_supported(weight.type))
        throw std::runtime_error("unsupported Qwen35 projection type " + std::to_string(weight.type));
    static const bool row2 = kernels::native_dense_q4_row2_enabled(std::getenv("STRATA_QWEN35_TG_DENSE_Q4_ROW2"));
    if (row2 && kernels::native_dense_q4_row2_eligible(decode_phase_, weight.type, n_in, n_out)) {
        kernels::native_q4_k_dense_row2(weight.ptr(), input_q8, output, n_in, n_out, 1, stream_);
        return;
    }
    kernels::native_mmvq(weight.type, weight.ptr(), input_q8, output, n_in, n_out, 1, stream_);
}

} // namespace strata::qwen35moe
