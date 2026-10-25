#pragma once
// Fixture ABI: byte copies on the existing CUDA compute stream, no new storage.
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::glm5_next {
struct ColdSubsetCopy { const void* source=nullptr; void* destination=nullptr; uint64_t bytes=0; };
struct ColdSubsetCopyInfo { uint64_t free_bytes=0,total_bytes=0,copied_bytes=0; uint32_t cuda_error=0; };
// count0 is an untimed memory query; count6 copies two experts in three banks.
// Optional count12 copies four experts, role-major. It uses its own by-value
// packet; the count6 layout/kernel remain unchanged. Empty controls must have
// all count descriptors exactly {nullptr,nullptr,0}; partial-empty is rejected.
using ProbeCopyExpertRows=bool (*)(ggml_backend_t,const ColdSubsetCopy*,size_t,ColdSubsetCopyInfo*);
struct ColdSubsetGpuTimes { float copy_ms=0,first_ms=0,total_ms=0;uint32_t cuda_error=0; };
//0create,1start,2copy-done,3first-done,4read(retired),5destroy(retired).
using ProbeColdTiming=bool (*)(ggml_backend_t,void**,uint32_t,ColdSubsetGpuTimes*);
}
