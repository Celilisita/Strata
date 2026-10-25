#pragma once
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::prefill::mmq {
enum class NativeMmqTileMode { automatic = 0, rows64 = 64, rows128 = 128 };
inline NativeMmqTileMode native_mmq_tile_mode(const char* value, const char* name) {
    if (!value || !std::strcmp(value, "64")) return NativeMmqTileMode::rows64;
    if (!std::strcmp(value, "128")) return NativeMmqTileMode::rows128;
    if (!std::strcmp(value, "auto")) return NativeMmqTileMode::automatic;
    throw std::invalid_argument(std::string(name) + " requires64,128 orauto");
}
inline int native_mmq_dense_tile_rows(NativeMmqTileMode mode, int type, int rows) {
    if (mode != NativeMmqTileMode::automatic) return int(mode);
    // Same measured Q4/Q5 kernels; short rows and Q6 keep their original tile.
    return (type == 12 || type == 13) && rows >= 1024 ? 128 : 64;
}
inline int native_mmq_group_tile_rows(NativeMmqTileMode mode, int type, int backend, int entries) {
    if (backend != 6 || (type != 12 && type != 13)) return 64;
    if (mode == NativeMmqTileMode::automatic)
        return entries >= 32768 ? 128 : 64; // No inference of benefit for16384 tails.
    return int(mode);
}
} // namespace strata::prefill::mmq
