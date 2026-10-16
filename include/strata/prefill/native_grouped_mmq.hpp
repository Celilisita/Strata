#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
namespace strata::prefill::mmq {
struct NativeMmqMatrix {
    uint64_t gate = 0, up = 0;
    int32_t begin = 0, end = 0;
};
struct NativeMmqTile { int32_t matrix = 0, tile = 0; };
// Canonical Q8 codes/scales, gathered once by route entry; no requantization.
void repack_native_q8_grouped(const void* input, void* output, int type, int cols,
    int source_rows, int rows, const int32_t* source_ids, void* stream);
// One tiled launch for arbitrary resident quantized matrices and scatter rows.
void native_grouped_mmq(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* output_ids, const void* packed, float* output,
    int rows, int max_rows, int cols, int outputs, void* stream,
    const NativeMmqTile* tiles = nullptr, int tile_count = 0);
struct NativeGroupedMmqWork { size_t control_bytes = 0, packed_bytes = 0; int groups = 0, launches = 0, scheduled_tiles = 0, active_tiles = 0; };
class NativeGroupedMmqProvider {
public:
    explicit NativeGroupedMmqProvider(void* stream);
    ~NativeGroupedMmqProvider();
    NativeGroupedMmqWork run(int type, const unsigned long long* first, const unsigned long long* second,
        const int32_t* source_ids, int entries, int source_rows,
        const void* q8, void* controls, size_t controls_bytes,
        void* packed, size_t packed_bytes, float* output, int cols, int outputs, bool active_tiles = false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::prefill::mmq
