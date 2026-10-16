#include "gguf_fixture.hpp"
#include "strata/models/model_family.hpp"
#include "strata/models/deepseek_v4/tensor_layout.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool check(const fs::path& file, const char* family, uint32_t layers,
                  uint32_t hidden, uint32_t experts, uint32_t used,
                  bool accepted, strata::models::Family expected) {
    fixture::write(file, {fixture::str("general.architecture", family),
                          fixture::u32(std::string(family) + ".block_count", layers),
                          fixture::u32(std::string(family) + ".embedding_length", hidden),
                          fixture::u32(std::string(family) + ".expert_count", experts),
                          fixture::u32(std::string(family) + ".expert_used_count", used)},
                   {{"token_embd.weight", {32}, 0, 1}});
    const strata::GgufFile gguf(file.string());
    strata::models::ModelIdentity identity{};
    std::string error;
    const bool ok = strata::models::identify(gguf, identity, error);
    return ok == accepted && (!ok ||
           (identity.family == expected && identity.layers == layers &&
            identity.hidden == hidden && identity.experts == experts && identity.experts_used == used));
}

int main() {
    const fs::path file = fs::temp_directory_path() / "strata-model-family-test.gguf";
    bool ok = false;
    try {
        using strata::models::Family;
        int elements = 0, bytes = 0;
        const bool mxfp4 = strata::block_geometry(39, elements, bytes) && elements == 32 && bytes == 17;
        const bool i32 = strata::block_geometry(26, elements, bytes) && elements == 1 && bytes == 4;
        strata::TensorInfo expert{"blk.0.ffn_gate_exps.weight", {4096, 2048, 256}, 39, 0};
        const bool tensor = strata::models::deepseek_v4::tensor_contract(expert, 39, {4096, 2048, 256}) &&
                            !strata::models::deepseek_v4::tensor_contract(expert, 8, {4096, 2048, 256}) &&
                            strata::tensor_payload_bytes(expert) == 256ull * 4096 * 2048 / 32 * 17;
        ok = mxfp4 && i32 && tensor &&
             check(file, "qwen4exp", 48, 2560, 512, 10, true, Family::qwen4exp) &&
             check(file, "deepseek4", 43, 4096, 256, 6, true, Family::deepseek4) &&
             check(file, "deepseek4", 43, 4096, 256, 257, false, Family::deepseek4) &&
             check(file, "unknown", 43, 4096, 256, 6, false, Family::deepseek4);
    } catch (...) {}
    std::error_code ignored;
    fs::remove(file, ignored);
    return ok ? 0 : 1;
}
