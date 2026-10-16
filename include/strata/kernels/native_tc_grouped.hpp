#pragma once
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

namespace strata::kernels {
// Non-owning launch descriptors. The authoritative directory supplies addresses;
// grouping never changes its ages, ownership, route order or output identity.
struct NativeTcMeta { uint32_t aux, dm; };
struct NativeTcTile {
    unsigned long long address = 0;
    int32_t matrix_slot = -1;
    int32_t input_rows[16];
    int32_t output_rows[16];
};
inline std::vector<NativeTcTile> native_tc_plan(const unsigned long long* addresses,
    const int32_t* rows, int entries) {
    std::map<unsigned long long, std::vector<int>> grouped;
    for (int i = 0; i < entries; ++i) {
        if (!addresses[i] || rows[i] < 0) throw std::invalid_argument("invalid TC entry");
        grouped[addresses[i]].push_back(i);
    }
    std::vector<NativeTcTile> result;
    int slot = 0;
    for (const auto& group : grouped) {
        for (size_t start = 0; start < group.second.size(); start += 16) {
            NativeTcTile tile{};
            tile.address = group.first;
            tile.matrix_slot = slot;
            for (int j = 0; j < 16; ++j) {
                const size_t pos = start + size_t(j);
                const int entry = pos < group.second.size() ? group.second[pos] : -1;
                tile.output_rows[j] = entry;
                tile.input_rows[j] = entry >= 0 ? rows[entry] : -1;
            }
            result.push_back(tile);
        }
        ++slot;
    }
    return result;
}
// Q4_K/Q6_K × canonical Q8_1. Exact integer dots use FP16 Tensor Cores;
// canonical scales/min and four-warp reduction remain unchanged. No allocations,
// synchronization or weight conversion. Caller owns device descriptors.
// Canonical MMVQ with eight gathered columns sharing each weight load.
void native_mmvq_gathered(int type, const NativeTcTile* tiles, int tile_count,
    const void* q8, float* output, int n_in, int n_out, void* stream, int tile_rows = 1);
void native_tc_configure(); // Call once for the caller-owned CUDA context.
void native_tc_grouped(int type, const NativeTcTile* tiles, int tile_count,
    const void* q8, float* output, int n_in, int n_out, void* stream);
} // namespace strata::kernels
