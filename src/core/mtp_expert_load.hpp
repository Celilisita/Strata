#pragma once
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "strata/platform/memory.hpp"

namespace strata::core {
// Preserve original Q8_0 payloads; the legacy packed format remains available.
inline bool load_mtp_expert_payload(const std::string& directory, int experts, int width, int ffn,
                                    size_t legacy_bytes, bool& q8, size_t& blob_bytes,
                                    uint8_t*& device, std::string& error) {
    const std::string q8_path = directory + "/experts.q8.bin";
    std::ifstream probe(q8_path, std::ios::binary | std::ios::ate);
    q8 = bool(probe);
    if (q8 && (width <= 0 || ffn <= 0 || width % 32 || ffn % 32)) {
        error = "mtp: Q8_0 expert dimensions must be positive multiples of 32"; return false;
    }
    blob_bytes = q8 ? size_t(3) * size_t(ffn) * size_t(width / 32) * 34 : legacy_bytes;
    const uint64_t bytes = uint64_t(experts) * blob_bytes;
    if (q8 && uint64_t(probe.tellg()) != bytes) {
        error = "mtp: experts.q8.bin size does not match model geometry"; return false;
    }
    probe.close();
    const std::string path = directory + (q8 ? "/experts.q8.bin" : "/experts.bin");
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) { error = "mtp: cannot open " + path; return false; }
    struct Close { FILE* file; ~Close() { std::fclose(file); } } close{file};
    auto code = cudaMalloc(reinterpret_cast<void**>(&device), bytes);
    if (code != cudaSuccess) { error = "mtp: expert allocation: " + std::string(cudaGetErrorString(code)); return false; }
#if !defined(_WIN32)
    strata::platform::advise_willneed(fileno(file), 0, bytes);
#endif
    std::vector<uint8_t> chunk(64u << 20);
    for (uint64_t offset = 0; offset < bytes;) {
        const size_t count = size_t(std::min<uint64_t>(chunk.size(), bytes - offset));
        if (std::fread(chunk.data(), 1, count, file) != count) { error = "mtp: expert payload truncated"; return false; }
        code = cudaMemcpy(device + offset, chunk.data(), count, cudaMemcpyHostToDevice);
        if (code != cudaSuccess) { error = "mtp: expert upload: " + std::string(cudaGetErrorString(code)); return false; }
        offset += count;
    }
    std::fprintf(stderr, "strata mtp: routed experts %s, %d experts, %llu bytes\n",
                 q8 ? "original Q8_0" : "legacy packed", experts, static_cast<unsigned long long>(bytes));
    return true;
}
}
