#pragma once
#include "strata/qwen35/native/mtp.hpp"
#include "mtp_prefix_graph_policy.hpp"
#include <cstdlib>
#include <cstring>

namespace strata::qwen35::native {
inline bool reuse_service_mtp_prefill() {
    const char* control = std::getenv("STRATA_DENSE_REUSE_PREFILL");
    if (control && std::strcmp(control, "0") == 0) return false;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "MTP prefill execution device");
    const auto caps = capabilities(device);
    // Admission follows the full-prefix/capture state proof on SM120.
    // Keep the established preparation on other providers until validated.
    return caps.major == 12 && caps.minor == 0;
}
inline int prepare_service_mtp_prefix(NativeMtp& mtp, DenseModel& model,
                                     const int* input, int tokens, int maximum,
                                     bool reuse, cudaStream_t stream, int offset = 0) {
    if (offset > 0) {
        // Resume: the main and draft KV already hold [0, offset); only the new
        // tail is evaluated and the decode graphs are re-captured uncached.
        mtp.prefill(input + offset, tokens - offset, stream, offset);
        check_cuda(cudaStreamSynchronize(stream), "Dense resumed MTP prefix");
        mtp.enable_graphs(input, tokens, maximum, stream, false);
        return tokens - offset;
    }
    const int warm = reuse ? tokens : (std::min)(tokens,
        model.max_tokens() + tokens % model.max_tokens());
    mtp.prefill(input, warm, stream);
    check_cuda(cudaStreamSynchronize(stream), "Dense MTP shape preparation");
    if(reuse&&skip_service_mtp_prefix_graph(reuse))mtp.prepare_paged_decode(tokens,maximum,stream);
    // Capture instantiates/uploads graphs without executing their kernels.
    // The complete eager prefix remains authoritative for subsequent decode.
    mtp.enable_graphs(input, tokens, maximum, stream,!skip_service_mtp_prefix_graph(reuse));
    if (!reuse) {
        model.reset_prefill(stream);
        mtp.prefill(input, tokens, stream);
    }
    return reuse ? tokens : warm + tokens;
}
}

