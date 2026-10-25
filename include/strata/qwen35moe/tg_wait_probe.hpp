#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::qwen35moe {
struct TgWaitProbeStamp {
    uint64_t await_cycles;
    uint32_t epoch,token,layer,mask;
};
static_assert(sizeof(TgWaitProbeStamp)==24,"fixed TG wait stamp");
// Single diagnostic Runtime per process. Call on the existing compute stream;
// source mapping must survive its existing head/completion sync and final drain.
// Disable with nullptr,0,0 before a later epoch or process teardown.
void configure_native_tg_wait_probe(TgWaitProbeStamp* device_alias,uint32_t first_epoch,
                                    uint32_t last_epoch,cudaStream_t stream);
}
