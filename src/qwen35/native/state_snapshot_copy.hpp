#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace strata::qwen35::native {
constexpr int StateSnapshotCopyLayers=48;
constexpr size_t StateSnapshotCopyBytes=size_t(128)*128*48*4;
struct StateSnapshotCopyPair {void *saved_state;const void *state;};
struct StateSnapshotCopyBatch {StateSnapshotCopyPair rows[StateSnapshotCopyLayers];};
static_assert(sizeof(void*)==8&&sizeof(StateSnapshotCopyPair)==16&&sizeof(StateSnapshotCopyBatch)==768,"fixed state snapshot parameter ABI");
static_assert(sizeof(StateSnapshotCopyBatch)<=4096,"SM120 kernel parameter support floor");
struct StateSnapshotCopyResources {int registers,blocks_per_sm,binary_version;size_t shared_bytes,local_bytes,parameter_bytes;};
StateSnapshotCopyResources state_snapshot_copy_resources();
void configure_state_snapshot_copy();
bool state_snapshot_copy_enabled();
std::uint64_t state_snapshot_copy_host_submissions();
void validate_state_snapshot_copy(const StateSnapshotCopyBatch&);
// Graph receives an owned copy of the768B descriptor, never a GPU pointer to host storage.
void state_snapshot_copy(StateSnapshotCopyBatch,cudaStream_t);
}
