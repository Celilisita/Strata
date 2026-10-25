#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
#include <cstdint>
namespace strata::qwen35::native {
struct QkvzRequestLayoutStats {
    int physical_mode=0;
    std::uint64_t epoch=0,conversions=0,matrix_count=0,values_bytes=0,traffic_bytes=0;
    double elapsed_ms=0;
    bool oldgraphs_cleared=false,failed=false;
    std::uintptr_t owner=0;std::size_t scratch_bytes=0;
};
inline QkvzRequestLayoutStats &qkvz_request_layout_stats(){static thread_local QkvzRequestLayoutStats value;return value;}
// PREPARE only: caller releases every active/parked graph before a true change.
bool nvfp4_request_layout_change_required(DenseGemm &,int prompt);
void nvfp4_prepare_request_layout(DenseGemm &,int prompt,bool oldgraphs_cleared,cudaStream_t);
}
