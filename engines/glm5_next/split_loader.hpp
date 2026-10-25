#pragma once
#include "llama.h"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::glm5_next {
inline llama_model* load(const std::string& first, const llama_model_params& params) {
    const std::filesystem::path path(first);
    // The vendor rewrite contains only the metadata shard. Its four immutable
    // payload siblings live one directory above it. Pass the full ordered list
    // explicitly; never copy, rewrite, or reinterpret a weight shard.
    if (path.parent_path().filename() != "Shard_Rewrite")
        return llama_model_load_from_file(first.c_str(), params);
    const std::string name = path.filename().string();
    const std::string suffix = "-00001-of-00005.gguf";
    if (name.size() <= suffix.size() || name.substr(name.size() - suffix.size()) != suffix)
        throw std::runtime_error("GLM rewrite entry must be shard 1 of the verified five-shard set");
    std::vector<std::string> names{first};
    for (int i = 2; i <= 5; ++i) {
        const std::string shard = name.substr(0, name.size() - suffix.size()) +
            "-0000" + std::to_string(i) + "-of-00005.gguf";
        names.push_back((path.parent_path().parent_path() / shard).string());
    }
    std::vector<const char*> paths;
    for (const auto& item : names) {
        if (!std::filesystem::is_regular_file(item))
            throw std::runtime_error("GLM paired split is absent: " + item);
        paths.push_back(item.c_str());
    }
    return llama_model_load_from_splits(paths.data(), paths.size(), params);
}

}
