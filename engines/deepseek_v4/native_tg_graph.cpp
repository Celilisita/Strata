#include "native_tg_graph.hpp"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::deepseek_v4 {
static_assert(sizeof(ggml_map_custom3_op_params) == 24);
static void reject_cpu(ggml_tensor*, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*, int, int, void*) {
    GGML_ABORT("Strata native TG requires its CUDA expert service");
}
bool native_tg_enabled() {
    const char* v = std::getenv("STRATA_DSV4_NATIVE_TG");
    if (v && std::strcmp(v, "0") && std::strcmp(v, "1"))
        throw std::runtime_error("STRATA_DSV4_NATIVE_TG must be 0 or 1");
    return v && v[0] == '1';
}
bool native_tg_cpu_only() {
    const char* v = std::getenv("STRATA_DSV4_TG_CPU_ONLY");
    if (v && std::strcmp(v, "0") && std::strcmp(v, "1"))
        throw std::runtime_error("STRATA_DSV4_TG_CPU_ONLY must be 0 or 1");
    return v && v[0] == '1';
}
ggml_tensor* finish_native_tg(ggml_context* ctx, ggml_tensor* start, ggml_tensor* shared) {
    if (!is_native_tg(start) || (native_tg_sources(start).reserved & 3) != 1) return nullptr;
    if (!shared || shared->type != GGML_TYPE_F32 || !ggml_is_contiguous(shared) || ggml_nelements(shared) != ggml_nelements(start))
        throw std::runtime_error("native TG shared expert contribution geometry differs");
    auto* out = ggml_map_custom3(ctx, start, start->src[1], start->src[2], reject_cpu, 1,
        reinterpret_cast<void*>(native_tg_magic));
    auto sources = native_tg_sources(start); sources.reserved = (sources.reserved & ~uint16_t(3)) | 2;
    std::memcpy(reinterpret_cast<char*>(out->op_params) + 24, &sources, sizeof(sources));
    out->src[3] = shared;
    return out;
}
ggml_tensor* build_native_tg(ggml_context* ctx, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* weights,
    ggml_tensor* slots, ggml_tensor* gate_bank, ggml_tensor* up_bank, ggml_tensor* down_bank,
    ggml_tensor* gate_source, ggml_tensor* up_source, ggml_tensor* down_source, float clamp, int layer) {
    auto* lookup = slots && slots->op == GGML_OP_RESHAPE ? slots->src[0] : nullptr;
    if (!device_cache_lookup(lookup)) return nullptr;
    // The router's top-k result is a strided view for two positions. Reuse the
    // contiguous original IDs already materialized for the cache lookup.
    if(ids&&lookup->src[1]&&ggml_nelements(ids)==ggml_nelements(lookup->src[1]))ids=lookup->src[1];
    if(weights&&!ggml_is_contiguous(weights))weights=ggml_cont(ctx,weights);
    if (!device_cache_supported(lookup) || !x || x->type != GGML_TYPE_F32 ||
        (ggml_nelements(x) != 4096 && ggml_nelements(x) != 8192) || !ggml_is_contiguous(x) || !ids || ids->type != GGML_TYPE_I32 ||
        ggml_nelements(ids) != ggml_nelements(x)/4096*6 || !ggml_is_contiguous(ids) || !weights || weights->type != GGML_TYPE_F32 ||
        ggml_nelements(weights) != ggml_nelements(ids) || !ggml_is_contiguous(weights) || layer < 0 || layer >= device_cache_max_layers)
        throw std::runtime_error("native TG route/activation contract differs: layer="+std::to_string(layer)+" x="+std::to_string(x?ggml_nelements(x):0)+" ids="+std::to_string(ids?ggml_nelements(ids):0)+" weights="+std::to_string(weights?ggml_nelements(weights):0)+" lookup_supported="+std::to_string(device_cache_supported(lookup)));
    for (auto* bank : {gate_bank, up_bank, down_bank})
        if (!bank || bank->type != GGML_TYPE_MXFP4 || !ggml_is_contiguous(bank) || bank->ne[2] < 6)
            throw std::runtime_error("native TG device cache bank contract differs");
    for (auto* source : {gate_source, up_source, down_source})
        if (!source || source->type != GGML_TYPE_MXFP4 || !source->data || !source->buffer ||
            !ggml_backend_buffer_is_host(source->buffer) || source->ne[2] != 256 || !ggml_is_contiguous(source))
            throw std::runtime_error("native TG immutable host source contract differs");
    if (gate_bank->ne[0] != 4096 || gate_bank->ne[1] != 2048 || up_bank->ne[0] != 4096 || up_bank->ne[1] != 2048 ||
        down_bank->ne[0] != 2048 || down_bank->ne[1] != 4096 || gate_bank->ne[2] != up_bank->ne[2] ||
        up_bank->ne[2] != down_bank->ne[2] || gate_source->ne[0] != 4096 || gate_source->ne[1] != 2048 ||
        up_source->ne[0] != 4096 || up_source->ne[1] != 2048 || down_source->ne[0] != 2048 || down_source->ne[1] != 4096)
        throw std::runtime_error("native TG Gate/Up/Down geometry differs");
    uintptr_t packet = 0;
    std::memcpy(&packet, lookup->op_params, sizeof(packet));
    if (!packet) throw std::runtime_error("native TG miss carrier absent");
    auto* out = ggml_map_custom3(ctx, x, ids, weights, reject_cpu, 1, reinterpret_cast<void*>(native_tg_magic));
    const char* cached_only=std::getenv("STRATA_DSV4_MTP_CACHE_ONLY");
    const bool skip_misses=layer==43&&cached_only&&!std::strcmp(cached_only,"1");
    const NativeTgSources sources{gate_source, up_source, down_source, packet, clamp, uint16_t(layer),
        uint16_t((native_tg_cpu_only() ? 1 : 0)|(skip_misses ? 4 : 0))};
    std::memcpy(reinterpret_cast<char*>(out->op_params) + 24, &sources, sizeof(sources));
    out->src[3] = gate_bank; out->src[4] = up_bank; out->src[5] = down_bank;
    out->src[6] = lookup->src[0]; out->src[7] = lookup->src[2];
    out->src[8] = lookup->src[3]; out->src[9] = lookup->src[4];
    // Do not include the legacy lookup in this graph: native CUDA publication
    // starts CPU/DMA work; a bank-free publication gate completes DMA before
    // MMVQ names the expert allocation. CPU work can overlap resident math.
    return out;
}
}
