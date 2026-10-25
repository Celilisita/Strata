#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
namespace strata::qwen35::native {
constexpr int PrefixCopyLayers=48;
constexpr size_t PrefixCopyStateBytes=size_t(128)*128*48*4;
constexpr size_t PrefixCopyHistoryBytes=size_t(3)*128*(2*16+48)*4;
struct PrefixCopyPair {void *state;const void *prefix_state;void *history;const void *prefix_history;};
struct PrefixCopyBatch {PrefixCopyPair rows[PrefixCopyLayers];};
static_assert(sizeof(void*)==8&&sizeof(PrefixCopyPair)==32&&sizeof(PrefixCopyBatch)==1536,"fixed prefix parameter ABI");
static_assert(sizeof(PrefixCopyBatch)<=4096,"SM120 kernel parameter support floor");
struct PrefixCopyResources {int registers,blocks_per_sm,binary_version;size_t shared_bytes,local_bytes,parameter_bytes;};
PrefixCopyResources prefix_copy_resources();
void configure_prefix_copy();
bool prefix_copy_enabled();
std::uint64_t prefix_copy_host_submissions();
void validate_prefix_copy(const PrefixCopyBatch&);
// Raw bits only. CUDA Graph owns a copied by-value descriptor, never this host array.
void prefix_copy_restore(PrefixCopyBatch,cudaStream_t);
}
