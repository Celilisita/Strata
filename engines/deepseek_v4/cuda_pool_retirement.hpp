#pragma once
#include "ggml-backend.h"
#include <cstdint>
namespace strata::deepseek_v4 {
struct CudaPoolRetirement {uint64_t retained_bytes=0,released_bytes=0,graphs_destroyed=0;};
using CudaPoolInspect=bool(*)(ggml_backend_t,CudaPoolRetirement*);
using CudaPoolRetire=bool(*)(ggml_backend_t,CudaPoolRetirement*);
}
