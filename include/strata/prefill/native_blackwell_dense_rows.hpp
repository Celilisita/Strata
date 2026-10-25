#pragma once
#include "strata/prefill/native_blackwell_mmq.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::prefill::mmq {
// Piece scheduling preserves source/output offsets and ordered scratch reuse.
inline int blackwell_dense_row_tile() {
    static const int selected = [] {
        const char* value = std::getenv("STRATA_BENCH_DENSE_ROWS");
        if (!value || !std::strcmp(value, "2048")) return 2048;
        if (!std::strcmp(value, "4096")) return 4096;
        throw std::invalid_argument("Blackwell dense rows require2048 or4096");
    }();
    return selected;
}
inline int run_blackwell_dense_rows(int type, const void* weights, const void* input,
    size_t input_bytes, void* codes, size_t code_bytes, void* metadata, size_t metadata_bytes,
    float* output, size_t output_bytes, int width, int rows, int outputs, void* stream,
    int tile_rows = blackwell_dense_row_tile()) {
    if (tile_rows != 2048 && tile_rows != 4096)
        throw std::invalid_argument("Blackwell dense row tile requires2048 or4096");
    if ((type != 12 && type != 13 && type != 14) || !weights || !input || !codes || !metadata ||
        !output || !stream || width < 512 || width > 4096 || width % 512 ||
        rows < 1 || rows > 8192 || outputs < 16 || outputs > 8192 || outputs % 16)
        throw std::invalid_argument("invalid bounded Blackwell dense row schedule");
    if (reinterpret_cast<uintptr_t>(weights) % 4 || reinterpret_cast<uintptr_t>(input) % 4 ||
        reinterpret_cast<uintptr_t>(output) % 4 || reinterpret_cast<uintptr_t>(codes) % 16 ||
        reinterpret_cast<uintptr_t>(metadata) % 8)
        throw std::invalid_argument("unaligned Blackwell dense row views");
    const size_t stride = kernels::native_q8_1_bytes(width);
    const size_t input_required = size_t(rows) * stride;
    const size_t output_required = size_t(rows) * size_t(outputs) * sizeof(float);
    const size_t piece = size_t((std::min)(rows, tile_rows));
    const size_t code_required = piece * size_t(width) * 2;
    // GroupInputMeta is one F32 scale and one signed integer code sum.
    const size_t metadata_required = piece * size_t(width / (type == 14 ? 16 : 32)) * 8;
    if (input_bytes < input_required || output_bytes < output_required ||
        code_bytes < code_required || metadata_bytes < metadata_required)
        throw std::invalid_argument("unreserved Blackwell dense row schedule");
    const std::array<std::pair<uintptr_t, size_t>, 5> spans{{
        {reinterpret_cast<uintptr_t>(weights), kernels::native_mmvq_weight_bytes(type, width, outputs)},
        {reinterpret_cast<uintptr_t>(input), input_required},
        {reinterpret_cast<uintptr_t>(output), output_required},
        {reinterpret_cast<uintptr_t>(codes), code_required},
        {reinterpret_cast<uintptr_t>(metadata), metadata_required}}};
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto [begin, bytes] = spans[i];
        if (begin > (std::numeric_limits<uintptr_t>::max)() - bytes)
            throw std::invalid_argument("Blackwell dense view overflow");
        for (size_t j = 0; j < i; ++j)
            if (begin < spans[j].first + spans[j].second && spans[j].first < begin + bytes)
                throw std::invalid_argument("aliased Blackwell dense row views");
    }
    int launches = 0;
    for (int first = 0; first < rows; first += tile_rows) {
        const int count = (std::min)(tile_rows, rows - first);
        // Q8_1 is row-major packed blocks, not F32 or width*sizeof(float).
        const auto* source = static_cast<const uint8_t*>(input) + size_t(first) * stride;
        if (blackwell_dense_group_int8_enabled(type, count, width, outputs) && blackwell_dense_tile_rows(type, count) == 128)
            run_blackwell_dense_group_int8_tile(type, weights, source, codes, code_bytes,
                metadata, metadata_bytes, output + size_t(first) * outputs,
                width, count, outputs, stream, blackwell_dense_tile_rows(type, count));
        else run_blackwell_dense_group_f16(type, weights, source, codes, code_bytes,
            metadata, metadata_bytes, output + size_t(first) * outputs,
            width, count, outputs, stream);
        // One ordered stream completes both consumers before scratch is reused.
        launches += 2;
    }
    return launches;
}
} // namespace strata::prefill::mmq
