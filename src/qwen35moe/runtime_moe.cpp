#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"

#include <array>

namespace strata::qwen35moe {

void Runtime::moe_step(size_t il, const LayerWeights& layer, Metrics& metrics) {
    auto& w = work_;
    const auto f = [](const DeviceBuffer& b) { return b.as<float>(); };
    route_project(layer.router.f32(), f(w.norm), f(w.logits), 2048, 257, stream_);
    route_top8(f(w.logits), w.ids.as<int32_t>(), f(w.weights), stream_);
    std::array<int32_t, 8> ids{};
    cuda_check(cudaMemcpyAsync(ids.data(), w.ids.data(), sizeof(ids),
                                cudaMemcpyDeviceToHost, stream_), "read routed experts");
    cuda_check(cudaStreamSynchronize(stream_), "wait for routed experts");
    metrics.route_d2h += sizeof(ids);
    ++metrics.host_continuations;
    ++metrics.explicit_syncs;
    moe_with_ids(il, layer, ids.data(), metrics);
}

void Runtime::moe_with_ids(size_t il, const LayerWeights& layer,
                           const int32_t* ids, Metrics& metrics) {
    auto& w = work_;
    const auto f = [](const DeviceBuffer& b) { return b.as<float>(); };
    kernels::native_quantize_q8_1(f(w.norm), w.q8_input.data(), 2048, 1, stream_);
    const auto& source = expert_sources_[il];
    const int down_type = int(source.down_exps->type);
    for (int rank = 0; rank < 8; ++rank) {
        const int id = ids[rank];
        const auto expert = expert_view(il, id, source, metrics);
        kernels::native_mmvq(int(source.gate_exps->type), expert.gate, w.q8_input.data(), f(w.gate_out),
                             2048, 512, 1, stream_);
        kernels::native_mmvq(int(source.up_exps->type), expert.up, w.q8_input.data(), f(w.up_out),
                             2048, 512, 1, stream_);
        swiglu(f(w.gate_out), f(w.up_out), f(w.expert_h), 512, stream_);
        kernels::native_quantize_q8_1(f(w.expert_h), w.q8_small.data(), 512, 1, stream_);
        kernels::native_mmvq(down_type, expert.down, w.q8_small.data(),
                             f(w.parts) + size_t(rank) * 2048, 512, 2048, 1, stream_);
        metrics.expert_launches += 5;
    }
    kernels::native_mmvq(12, layer.shared_gate.ptr(), w.q8_input.data(),
                         f(w.gate_out), 2048, 512, 1, stream_);
    kernels::native_mmvq(12, layer.shared_up.ptr(), w.q8_input.data(),
                         f(w.up_out), 2048, 512, 1, stream_);
    swiglu(f(w.gate_out), f(w.up_out), f(w.expert_h), 512, stream_);
    kernels::native_quantize_q8_1(f(w.expert_h), w.q8_small.data(), 512, 1, stream_);
    kernels::native_mmvq(layer.shared_down.type, layer.shared_down.ptr(),
                         w.q8_small.data(), f(w.shared), 512, 2048, 1, stream_);
    gate_shared(f(w.shared), f(w.logits) + 256, 2048, stream_);
    kernels::native_moe_combine(f(w.parts), f(w.weights), f(w.shared),
                                f(w.ffn), 2048, 8, stream_);
    kernels::add_inplace(f(w.ffn), f(w.mixer), 2048, stream_);
    cuda_check(cudaMemcpyAsync(w.hidden.data(), w.ffn.data(), 2048 * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_), "next layer residual");
    metrics.expert_launches += 5;
}

} // namespace strata::qwen35moe
