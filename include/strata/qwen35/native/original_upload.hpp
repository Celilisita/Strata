#pragma once
#include "strata/runtime/capacity_storage.hpp"
#include "strata/qwen35/spec.hpp"
#include "strata/qwen35/native/original_formats.hpp"
namespace strata::qwen35::native {
// One original payload+sidecar owner. CPU reads metadata and checks format;
// scale preparation and all numerical decoding execute on GPU.
Matrix upload_original_nvfp4(const strata::GgufFile&,const strata::TensorInfo&,strata::runtime::CapacityStorage&,cudaStream_t);
}
