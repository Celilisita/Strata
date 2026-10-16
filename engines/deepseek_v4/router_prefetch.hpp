#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::deepseek_v4 {
struct RouterPrefetchRoutes {
    uint32_t epoch,layer,tokens,source;
    int32_t ids[12];
    uint32_t masks[2];
    int32_t selected[2];
};
struct alignas(64) RouterPrefetchPacket {
    volatile uint32_t sequence=0,hash_sequence=0;
    RouterPrefetchRoutes learned[2]{},hash[2][3]{};
};
bool router_prefetch_enabled();
RouterPrefetchPacket* router_prefetch_packet(ggml_backend_t);
ggml_tensor* build_router_hash_prefetch(ggml_context*,ggml_cgraph*,ggml_tensor* scalar,
    ggml_tensor* pending,ggml_tensor* draft,ggml_tensor* const* tables,ggml_tensor* const* directories,
    ggml_backend_t,bool both);
}
