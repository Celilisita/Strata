#pragma once
#include "strata/qwen35/completed_v100/native/weights.hpp"
#include "strata/qwen35/completed_v100/native/original_formats.hpp"
namespace strata::completed_v100::qwen35::native {
// One original payload+sidecar owner. CPU reads metadata and checks format;
// scale preparation and all numerical decoding execute on GPU.
Matrix upload_original_nvfp4(const strata::GgufFile&,const strata::TensorInfo&,DeviceStorage&,cudaStream_t);
}
