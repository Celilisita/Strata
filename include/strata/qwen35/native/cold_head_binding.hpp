#pragma once
#include "strata/qwen35/spec.hpp"
#include "strata/qwen35/native/nvfp4.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdint>
#include <string>
namespace strata::qwen35::native {
class DeviceStorage;
// The caller pins the complete source/fixture hashes before this cold binding.
// One selected payload replaces the logical shared output, never its inventory.
struct ColdHeadBinding {
    const strata::GgufFile *fixture = nullptr;
    uint64_t source_bytes = 0;
    std::string source_model, source_model_sha256, source_head_sha256;
    std::string source_header_sha256, source_norm_sha256;
    bool has_source_identity() const {
        return source_bytes || !source_model.empty() || !source_model_sha256.empty() ||
               !source_head_sha256.empty() || !source_header_sha256.empty() ||
               !source_norm_sha256.empty();
    }
};
// All-or-none explicit identity. Empty fields retain the frozen Qwen3.8 tuple.
void validate_cold_head_source_identity(const ColdHeadBinding &);
void validate_shared_head_identity(const strata::GgufFile &,const strata::qwen35::Spec &);
void validate_cold_head_binding(const strata::GgufFile &,const strata::qwen35::Spec &,ColdHeadBinding);
Matrix load_cold_head_binding(const strata::GgufFile &,const strata::qwen35::Spec &,ColdHeadBinding,
    DeviceStorage &,Nvfp4View &,cudaStream_t);
Matrix load_nvfp4_shared_head(const strata::GgufFile &,const strata::qwen35::Spec &,
    DeviceStorage &,Nvfp4View &,cudaStream_t);
}
