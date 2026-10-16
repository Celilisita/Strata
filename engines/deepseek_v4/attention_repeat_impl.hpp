#pragma once
#include "attention_repeat.hpp"
#include "ggml.h"
#include <cuda_runtime_api.h>
#include <cstring>
namespace strata::deepseek_v4 {
namespace {
struct AttentionRepeatStore {
    char* device_bytes = nullptr;
    size_t k_bytes = 0, mask_bytes = 0;
    uint32_t observed = 0;
    int error = 0;
};
thread_local AttentionRepeatStore* attention_actor = nullptr;
}
bool attention_repeat_api(AttentionRepeatPacket* packet) {
    if (!packet) return false;
    auto* store = static_cast<AttentionRepeatStore*>(packet->handle);
    packet->error = 0;
    if (packet->action == AttentionRepeatAction::Initialize) {
        if (store) return false;
        store = new AttentionRepeatStore;
        packet->error = int(cudaMalloc(reinterpret_cast<void**>(&store->device_bytes), attention_repeat_bytes));
        if (!packet->error) packet->error = int(cudaMemset(store->device_bytes, 0, attention_repeat_bytes));
        if (!packet->error) packet->error = int(cudaDeviceSynchronize());
        if (packet->error) { if (store->device_bytes) cudaFree(store->device_bytes); delete store; return false; }
        packet->handle = store; return true;
    }
    if (!store) return false;
    switch (packet->action) {
    case AttentionRepeatAction::Begin:
        if (attention_actor) return false;
        store->observed = 0; store->error = 0; attention_actor = store; return true;
    case AttentionRepeatAction::End:
        if (attention_actor != store) return false;
        attention_actor = nullptr; return true;
    case AttentionRepeatAction::Read:
        if (attention_actor || !packet->bytes || packet->capacity != attention_repeat_bytes) return false;
        packet->observed = store->observed; packet->k_bytes = store->k_bytes; packet->mask_bytes = store->mask_bytes;
        packet->error = store->error;
        if (!packet->error) packet->error = int(cudaMemcpy(packet->bytes, store->device_bytes, attention_repeat_bytes, cudaMemcpyDeviceToHost));
        return !packet->error && packet->observed >= 1;
    case AttentionRepeatAction::Release:
        if (attention_actor == store) attention_actor = nullptr;
        packet->error = int(cudaFree(store->device_bytes));
        delete store; packet->handle = nullptr; return !packet->error;
    default: return false;
    }
}
void attention_repeat_output(const ggml_tensor* tensor, int sample, uintptr_t stream) noexcept {
    auto* store = attention_actor;
    if (!store || sample < 0 || sample > attention_repeat_count) return;
    auto* destination = store->device_bytes + attention_repeat_output_offset + sample*attention_repeat_row_bytes;
    const auto status = cudaMemcpyAsync(destination,
        static_cast<const char*>(tensor->data)+(tensor->ne[2]-1)*tensor->nb[2], attention_repeat_row_bytes,
        cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) store->error = int(status);
}
bool attention_repeat_capture(const ggml_tensor* tensor, uintptr_t stream) noexcept {
    auto* store = attention_actor;
    if (!store || tensor->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const auto *q = tensor->src[0], *k = tensor->src[1], *v = tensor->src[2], *mask = tensor->src[3], *sink = tensor->src[4];
    if (!k || std::strncmp(ggml_get_name(k), "hca_k_all-37", 12) != 0) return false;
    ++store->observed;
    if (!q || !v || !mask || !sink || q->type != GGML_TYPE_F32 ||
        k->type != GGML_TYPE_F16 || v->data != k->data || mask->type != GGML_TYPE_F16 || sink->type != GGML_TYPE_F32 ||
        q->ne[0] != 512 || q->ne[1] != 8192 || q->ne[2] != 64 || q->ne[3] != 1 || q->nb[0] != 4 ||
        k->ne[0] != 512 || k->ne[2] != 1 || k->ne[3] != 1 || k->nb[1] != 1024 ||
        mask->ne[0] != k->ne[1] || mask->ne[1] != 8192 || sink->ne[0] != 64 ||
        !ggml_is_contiguous(tensor) || tensor->ne[0] != 512 || tensor->ne[1] != 64 || tensor->ne[2] != 8192 || tensor->ne[3] != 1) {
        store->error = int(cudaErrorInvalidValue); return false;
    }
    store->k_bytes = size_t(k->ne[1])*1024;
    store->mask_bytes = size_t(mask->ne[0])*2;
    if (store->k_bytes > attention_repeat_k_capacity || store->mask_bytes > attention_repeat_mask_capacity) {
        store->error = int(cudaErrorInvalidValue); return false;
    }
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    auto record = [&](cudaError_t error) { if (error != cudaSuccess) store->error = int(error); };
    record(cudaMemcpy2DAsync(store->device_bytes, 512*4,
        static_cast<const char*>(q->data)+(q->ne[1]-1)*q->nb[1], q->nb[2], 512*4, 64, cudaMemcpyDeviceToDevice, cuda_stream));
    record(cudaMemcpyAsync(store->device_bytes+attention_repeat_k_offset, k->data, store->k_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
    record(cudaMemcpyAsync(store->device_bytes+attention_repeat_mask_offset,
        static_cast<const char*>(mask->data)+(mask->ne[1]-1)*mask->nb[1], store->mask_bytes, cudaMemcpyDeviceToDevice, cuda_stream));
    record(cudaMemcpyAsync(store->device_bytes+attention_repeat_sink_offset, sink->data, 256, cudaMemcpyDeviceToDevice, cuda_stream));
    attention_repeat_output(tensor, 0, stream);
    return !store->error;
}
}
