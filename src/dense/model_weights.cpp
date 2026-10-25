#include "strata/dense/model.hpp"
#include "model_graph.hpp"
#include "ggml-cuda.h"
#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <limits>

namespace strata::dense {
namespace {
size_t configured_backend_reserve() {
    const char *value = std::getenv("STRATA_SMALL_DENSE_RESERVE_MIB");
    if (!value) return size_t(512)*1024*1024;
    if (!*value) throw std::invalid_argument("small dense reserve MiB must be a nonnegative integer");
    const size_t limit = (std::numeric_limits<size_t>::max)()/(1024*1024);
    size_t mib = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || mib > (limit-size_t(*p-'0'))/10)
            throw std::invalid_argument("small dense reserve MiB must be a nonnegative integer");
        mib = mib*10+size_t(*p-'0');
    }
    return mib*1024*1024;
}
ggml_context *metadata_context(size_t tensors) {
    ggml_init_params p{ggml_tensor_overhead()*(tensors+32), nullptr, true};
    auto *ctx = ggml_init(p);
    if (!ctx) throw std::runtime_error("dense metadata allocation failed");
    return ctx;
}
}

Model::Model(const std::string &path, int context, size_t budget_bytes, bool pp_half_cache,
             bool projection_fusion, bool compact_async, bool norm_fusion)
    : gguf_(path), projection_fusion_(projection_fusion), compact_async_(compact_async),
      norm_enabled_(norm_fusion), budget_(budget_bytes) {
    try {
        const size_t reserve = configured_backend_reserve();
        memory_.backend_reserve = reserve;
        std::string error;
        if (!load_spec(gguf_, spec_, error)) throw std::runtime_error(error);
        if (context < 1 || uint64_t(context) > spec_.context_length || context % 256)
            throw std::invalid_argument("dense context must be a trained-context multiple of 256");
        capacity_ = context;
        const char *request_kv = std::getenv("STRATA_SMALL_DENSE_REQUEST_KV");
        if (request_kv && std::string(request_kv)!="0" && std::string(request_kv)!="1")
            throw std::invalid_argument("small dense request KV policy must be 0 or 1");
        request_kv_ = request_kv && *request_kv=='1';
        state_capacity_ = request_kv_ ? 256 : capacity_;
        backend_ = ggml_backend_cuda_init(0);
        if (!backend_) throw std::runtime_error("dense CUDA backend unavailable");
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_memory(ggml_backend_get_device(backend_), &free_bytes, &total_bytes);
        if (free_bytes <= reserve) throw std::runtime_error("dense CUDA device has no safe free budget");
        budget_ = std::min(budget_, free_bytes-reserve);
        load_weights();
        allocate_state();
        if (compact_async_) compact_readback_ = std::make_unique<CompactReadback>(backend_);
        if (norm_enabled_) {
            if (spec_.architecture != "qwen35") throw std::invalid_argument("GDN norm fusion requires qwen35");
            const size_t available = memory_.total() < budget_ ? budget_-memory_.total() : 0;
            norm_fusion_ = std::make_unique<NormFusion>(backend_,spec_.ssm_state,available);
            memory_.norm_fusion = norm_fusion_->bytes();
        }
        if (pp_half_cache) {
            const size_t workspace_reserve = size_t(512)*1024*1024;
            const size_t available = memory_.total()+workspace_reserve < budget_ ?
                budget_-memory_.total()-workspace_reserve : 0;
            pp_half_ = std::make_unique<HalfCache>(gguf_,spec_,backend_,
                available,spec_.architecture=="llama");
            memory_.pp_half = pp_half_->bytes();
        }
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (!allocator_) throw std::runtime_error("dense graph allocator unavailable");
        reset();
    } catch (...) { release(); throw; }
}

void Model::load_weights() {
    size_t bytes = 0;
    for (const auto &t : gguf_.tensors()) bytes += tensor_bytes(gguf_, t)+256;
    if (bytes+memory_.backend_reserve > budget_) throw std::runtime_error("dense weights exceed GPU budget");
    if (projection_fusion_) {
        projection_pack_ = std::make_unique<ProjectionPack>(spec_);
        if (!projection_pack_->group_count()) throw std::runtime_error("no eligible projection groups");
    }
    weight_ctx_ = metadata_context(gguf_.tensors().size()+projection_pack_group_count());
    if (projection_pack_) projection_pack_->create_weights(weight_ctx_,weights_);
    for (const auto &t : gguf_.tensors()) {
        if (weights_.count(&t)) continue;
        int64_t shape[4]{1,1,1,1};
        for (size_t i = 0; i < t.shape.size(); ++i) shape[i] = int64_t(t.shape[i]);
        auto *gpu = ggml_new_tensor(weight_ctx_, ggml_type(t.type), int(t.shape.size()), shape);
        ggml_set_name(gpu, t.name.c_str());
        if (ggml_nbytes(gpu) != tensor_bytes(gguf_, t))
            throw std::runtime_error("GGML storage differs from GGUF payload: "+t.name);
        weights_.emplace(&t, gpu);
    }
    weight_buffer_ = ggml_backend_alloc_ctx_tensors(weight_ctx_, backend_);
    if (!weight_buffer_) throw std::runtime_error("dense resident weight allocation failed");
    ggml_backend_buffer_set_usage(weight_buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    memory_.weights = ggml_backend_buffer_get_size(weight_buffer_);
    for (const auto &t : gguf_.tensors())
        ggml_backend_tensor_set(weights_.at(&t), gguf_.tensor_data(t), 0, tensor_bytes(gguf_, t));
}

void Model::allocate_state() {
    state_ctx_ = metadata_context(spec_.n_layers*2);
    states_.resize(spec_.n_layers);
    size_t bytes = 0;
    for (size_t i = 0; i < states_.size(); ++i) {
        auto &state = states_[i];
        if (spec_.layers[i].recurrent) {
            const int64_t channels = 2*spec_.ssm_group*spec_.ssm_state+spec_.ssm_inner;
            state.history = ggml_new_tensor_2d(state_ctx_, GGML_TYPE_F32, spec_.ssm_conv-1, channels);
            state.recurrent = ggml_new_tensor_3d(state_ctx_, GGML_TYPE_F32,
                                                spec_.ssm_state, spec_.ssm_state, spec_.ssm_rank);
            bytes += ggml_nbytes(state.history)+ggml_nbytes(state.recurrent)+512;
        } else {
            const int64_t width = spec_.head_dim*spec_.n_head_kv;
            state.key = ggml_new_tensor_2d(state_ctx_, GGML_TYPE_F16, width, state_capacity_);
            state.value = ggml_new_tensor_2d(state_ctx_, GGML_TYPE_F16, width, state_capacity_);
            bytes += ggml_nbytes(state.key)+ggml_nbytes(state.value)+512;
        }
    }
    if (memory_.weights+bytes+memory_.workspace+memory_.pp_half+memory_.norm_fusion+memory_.backend_reserve > budget_)
        throw std::runtime_error("dense weights and KV/state exceed GPU budget");
    state_buffer_ = ggml_backend_alloc_ctx_tensors(state_ctx_, backend_);
    if (!state_buffer_) throw std::runtime_error("dense persistent state allocation failed");
    memory_.state = ggml_backend_buffer_get_size(state_buffer_);
    ggml_backend_buffer_clear(state_buffer_, 0);
}

void Model::prepare_request(int tokens) {
    if (tokens < 1 || tokens > capacity_)
        throw std::invalid_argument("dense request state exceeds logical context");
    reset();
    if (!request_kv_) return;
    const int required = (tokens+255)/256*256;
    if (state_buffer_ && required <= state_capacity_) return;
    // No graph, host packet or compute may retain a reference to the old cold state.
    ggml_backend_synchronize(backend_);
    if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
    if (state_ctx_) ggml_free(state_ctx_);
    state_buffer_=nullptr;state_ctx_=nullptr;states_.clear();memory_.state=0;
    state_capacity_=required;
    allocate_state();
}

void Model::reset() {
    if (compact_readback_) compact_readback_->drain();
    // PP may resize/reassign the shared workspace after this reset.
    decode_graph_.reset();
    position_ = 0;
    logical_position_ = 0;
    // Old KV rows cannot be visible: every new graph masks beyond its written prefix.
    // Recurrent state has no position mask and must be reset explicitly.
    std::vector<unsigned char> zeros;
    for (const auto &s : states_) if (state_buffer_ && s.recurrent) {
        zeros.assign(std::max(ggml_nbytes(s.history), ggml_nbytes(s.recurrent)), 0);
        ggml_backend_tensor_set(s.history, zeros.data(), 0, ggml_nbytes(s.history));
        ggml_backend_tensor_set(s.recurrent, zeros.data(), 0, ggml_nbytes(s.recurrent));
    }
}

void Model::release() noexcept {
    // Drain before destroying graph metadata, immutable leaves, or host packets.
    if (backend_) ggml_backend_synchronize(backend_);
    decode_graph_.reset();
    projection_pack_.reset();
    if (allocator_) ggml_gallocr_free(allocator_);
    norm_fusion_.reset();
    compact_readback_.reset();
    pp_half_.reset();
    if (state_buffer_) ggml_backend_buffer_free(state_buffer_);
    if (weight_buffer_) ggml_backend_buffer_free(weight_buffer_);
    if (state_ctx_) ggml_free(state_ctx_);
    if (weight_ctx_) ggml_free(weight_ctx_);
    if (backend_) ggml_backend_free(backend_);
    allocator_=nullptr;state_buffer_=nullptr;weight_buffer_=nullptr;
    state_ctx_=nullptr;weight_ctx_=nullptr;backend_=nullptr;
}
Model::~Model() { release(); }
} // namespace strata::dense
