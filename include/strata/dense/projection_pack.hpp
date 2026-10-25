#pragma once
#include "strata/dense/spec.hpp"
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace strata::dense {
// Sources borrow the GGUF directory. Roots/views belong exclusively to Model's
// weight metadata context and its single backend buffer; this plan owns neither.
class ProjectionPack {
public:
    using WeightMap = std::unordered_map<const TensorInfo *, ggml_tensor *>;
    struct Member { const TensorInfo *source; std::int64_t row_offset; };
    struct Group {
        std::uint32_t type;
        std::int64_t cols, rows;
        ggml_tensor *root = nullptr;
        std::vector<Member> members;
    };
    struct Key {
        ggml_tensor *root, *input;
        bool operator==(const Key &other) const noexcept { return root==other.root && input==other.input; }
    };
    struct KeyHash {
        std::size_t operator()(Key key) const noexcept {
            const auto a=std::hash<ggml_tensor *>{}(key.root), b=std::hash<ggml_tensor *>{}(key.input);
            return a ^ (b + std::size_t(0x9e3779b9u) + (a<<6) + (a>>2));
        }
    };
    // One cache per request graph construction. It contains metadata pointers,
    // not GPU allocations. Do not share it across graph contexts or threads.
    struct Cache {
        ggml_context *context = nullptr;
        std::unordered_map<Key, ggml_tensor *, KeyHash> outputs;
        void clear() { outputs.clear(); context=nullptr; }
        std::size_t size() const noexcept { return outputs.size(); }
    };

    explicit ProjectionPack(const Spec &);
    ProjectionPack(const ProjectionPack &) = delete;
    ProjectionPack &operator=(const ProjectionPack &) = delete;
    std::size_t group_count() const noexcept { return groups_.size(); }
    std::size_t member_count() const noexcept { return members_.size(); }
    std::size_t extra_tensors() const noexcept { return groups_.size(); }
    const std::vector<Group> &groups() const noexcept { return groups_; }

    // Call once before Model creates unpacked tensors / allocates its buffer.
    // Inserts original-shape views for packed members only. Existing member
    // entries are rejected. Upload each source's original payload to its view.
    void create_weights(ggml_context *, WeightMap &);

    // Returns nullptr for unpacked sources, non-F32/non-contiguous inputs or
    // shapes other than [cols,1,1,1]. Caller uses its original PP/HalfCache path.
    // A hit computes root*input once and returns contiguous F32 [rows,1,1,1].
    ggml_tensor *project(ggml_context *, const TensorInfo *source,
                         ggml_tensor *input, Cache &) const;
private:
    struct Location { std::size_t group, member; };
    std::vector<Group> groups_;
    std::unordered_map<const TensorInfo *, Location> members_;
    ggml_context *weight_context_ = nullptr;
};
} // namespace strata::dense
