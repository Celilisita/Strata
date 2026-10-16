#pragma once
#include "layer_snapshots.hpp"
#include "ggml.h"
#include <cuda_runtime_api.h>
#include <array>
#include <cstdlib>
#include <cstring>
namespace strata::deepseek_v4 {
namespace {
struct LayerSnapshotStore {
    float* device_rows = nullptr;
    std::array<uint32_t, layer_snapshot_layers> observed{};
    std::array<int32_t, layer_snapshot_layers> types{};
    std::array<size_t, layer_snapshot_layers> row_values{};
    uint64_t mask[2]{};
    int device = -1, error = 0;
};
thread_local LayerSnapshotStore* snapshot_actor = nullptr;
}
bool layer_snapshot_api(LayerSnapshotPacket* packet) {
    if (!packet) return false;
    auto* store = static_cast<LayerSnapshotStore*>(packet->handle);
    packet->error = 0;
    if (packet->action == LayerSnapshotAction::Initialize) {
        if (store) return false;
        store = new LayerSnapshotStore;
        auto status = cudaGetDevice(&store->device);
        if (status == cudaSuccess)
            status = cudaMalloc(reinterpret_cast<void**>(&store->device_rows),
                                layer_snapshot_layers * layer_snapshot_values * sizeof(float));
        if (status == cudaSuccess) status = cudaMemset(store->device_rows, 0,
            layer_snapshot_layers * layer_snapshot_values * sizeof(float));
        if (status == cudaSuccess) status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            packet->error = int(status);
            if (store->device_rows) cudaFree(store->device_rows);
            delete store; return false;
        }
        packet->handle = store;
        return true;
    }
    if (!store) return false;
    switch (packet->action) {
    case LayerSnapshotAction::Begin:
        if (snapshot_actor) return false;
        store->observed.fill(0); store->row_values.fill(0); store->types.fill(0);
        store->mask[0] = packet->mask[0]; store->mask[1] = packet->mask[1];
        store->error = 0; snapshot_actor = store; return true;
    case LayerSnapshotAction::End:
        if (snapshot_actor != store) return false;
        snapshot_actor = nullptr; return true;
    case LayerSnapshotAction::Read:
        if (snapshot_actor || !packet->rows || !packet->observed || !packet->types || !packet->row_values ||
            packet->values != layer_snapshot_layers * layer_snapshot_values) return false;
        std::memcpy(packet->observed, store->observed.data(), sizeof(store->observed));
        std::memcpy(packet->types, store->types.data(), sizeof(store->types));
        std::memcpy(packet->row_values, store->row_values.data(), sizeof(store->row_values));
        if (store->error) { packet->error = store->error; return false; }
        packet->error = int(cudaMemcpy(packet->rows, store->device_rows,
                                      packet->values * sizeof(float), cudaMemcpyDeviceToHost));
        return packet->error == 0;
    case LayerSnapshotAction::Release:
        if (snapshot_actor == store) snapshot_actor = nullptr;
        // Benchmark releases only after all owning contexts/graphs are gone.
        packet->error = int(cudaFree(store->device_rows));
        delete store; packet->handle = nullptr; return packet->error == 0;
    default: return false;
    }
}
void layer_snapshot_node(const ggml_tensor* tensor, uintptr_t stream) noexcept {
    auto* store = snapshot_actor;
    if (!store) return;
    const int layer = layer_snapshot_index(ggml_get_name(tensor));
    if (layer < 0 || !(store->mask[layer/64] & (uint64_t(1)<<(layer%64)))) return;
    const int axis = layer_snapshot_axis(size_t(layer));
    size_t values = size_t(tensor->ne[0]);
    for (int dim = 1; dim < 4; ++dim) if (dim != axis) values *= size_t(tensor->ne[dim]);
    if (!tensor->data || tensor->type != (layer_snapshot_integer(size_t(layer)) ? GGML_TYPE_I32 : GGML_TYPE_F32) ||
        tensor->nb[0] != sizeof(float) || values != layer_snapshot_width(size_t(layer)) || values > layer_snapshot_values) {
        store->error = int(cudaErrorInvalidValue); return;
    }
    auto* destination = store->device_rows + layer * layer_snapshot_values;
    if (tensor->ne[3] != 1 || (axis != 1 && axis != 2)) {
        store->error = int(cudaErrorInvalidValue); return;
    }
    const size_t width = size_t(tensor->ne[0])*sizeof(float);
    const int rows_axis = axis == 2 ? 1 : 2;
    const size_t offset = size_t(tensor->ne[axis]-1)*tensor->nb[axis];
    const auto result = cudaMemcpy2DAsync(destination, width,
        static_cast<const char*>(tensor->data)+offset, tensor->nb[rows_axis],
        width, size_t(tensor->ne[rows_axis]), cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream));
    if (result != cudaSuccess) { store->error = int(result); return; }
    store->types[layer] = tensor->type; store->row_values[layer] = values;
    ++store->observed[layer];
}
}
