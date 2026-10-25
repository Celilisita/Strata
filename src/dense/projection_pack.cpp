#include "strata/dense/projection_pack.hpp"
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::dense {
namespace {
bool eligible(const TensorInfo *tensor) {
    if (!tensor || tensor->shape.size()!=2 || !tensor->shape[0] || !tensor->shape[1] ||
        tensor->shape[0]>std::uint64_t(std::numeric_limits<std::int64_t>::max()) ||
        tensor->shape[1]>std::uint64_t(std::numeric_limits<std::int64_t>::max())) return false;
    const std::uint64_t block = tensor->type==8 ? 32 : 256;
    return (tensor->type==8 || tensor->type==12 || tensor->type==14) && tensor->shape[0]%block==0;
}

std::size_t row_bytes(std::uint32_t type, std::int64_t cols) {
    const std::uint64_t block = type==8 ? 32 : 256;
    const std::uint64_t bytes = type==8 ? 34 : type==12 ? 144 : type==14 ? 210 : 0;
    if (!bytes || cols<=0 || std::uint64_t(cols)%block ||
        std::uint64_t(cols)/block>std::numeric_limits<std::size_t>::max()/bytes)
        throw std::invalid_argument("invalid packed projection quantized row geometry");
    return std::size_t(std::uint64_t(cols)/block*bytes);
}

std::size_t matrix_bytes(std::size_t row, std::int64_t rows) {
    if (rows<=0 || std::uint64_t(rows)>std::numeric_limits<std::size_t>::max()/row)
        throw std::overflow_error("packed projection matrix byte count overflows");
    return row*std::size_t(rows);
}
} // namespace

ProjectionPack::ProjectionPack(const Spec &spec) {
    for (const auto &layer : spec.layers) {
        const std::vector<const TensorInfo *> candidates = layer.recurrent ?
            std::vector<const TensorInfo *>{layer.qkv,layer.gate,layer.beta,layer.alpha} :
            std::vector<const TensorInfo *>{layer.q,layer.k,layer.v};
        // Preserve layer and source order; split mixed formats/columns. A lone
        // projection stays unchanged rather than introducing a useless root.
        std::map<std::pair<std::uint32_t,std::uint64_t>,std::vector<const TensorInfo *>> buckets;
        for (const auto *source : candidates)
            if (eligible(source)) buckets[{source->type,source->shape[0]}].push_back(source);
        for (const auto &bucket : buckets) {
            if (bucket.second.size()<2) continue;
            Group group{bucket.first.first,std::int64_t(bucket.first.second),0,nullptr,{}};
            const auto row = row_bytes(group.type,group.cols);
            for (const auto *source : bucket.second) {
                const auto rows = std::int64_t(source->shape[1]);
                if (rows>std::numeric_limits<std::int64_t>::max()-group.rows)
                    throw std::overflow_error("packed projection row count overflows");
                if (members_.count(source))
                    throw std::invalid_argument("projection source belongs to multiple packed groups");
                members_.emplace(source,Location{groups_.size(),group.members.size()});
                group.members.push_back({source,group.rows});
                group.rows+=rows;
            }
            (void)matrix_bytes(row,group.rows);
            if (std::uint64_t(group.rows)>std::numeric_limits<std::size_t>::max()/sizeof(float))
                throw std::overflow_error("packed projection output bytes overflow");
            groups_.push_back(std::move(group));
        }
    }
}

void ProjectionPack::create_weights(ggml_context *context, WeightMap &weights) {
    if (!context || weight_context_)
        throw std::logic_error("packed projection weights require one live metadata context");
    for (const auto &entry : members_)
        if (weights.count(entry.first))
            throw std::logic_error("packed projection source already has a weight tensor");
    for (std::size_t index=0; index<groups_.size(); ++index) {
        auto &group=groups_[index];
        const auto row=row_bytes(group.type,group.cols);
        const auto bytes=matrix_bytes(row,group.rows);
        group.root=ggml_new_tensor_2d(context,ggml_type(group.type),group.cols,group.rows);
        if (!group.root || ggml_nbytes(group.root)!=bytes)
            throw std::runtime_error("GGML packed projection root storage differs from original payload");
        ggml_set_name(group.root,("dense.projection_pack."+std::to_string(index)).c_str());
        for (const auto &member : group.members) {
            const auto rows=std::int64_t(member.source->shape[1]);
            const auto offset=std::size_t(member.row_offset)*row;
            auto *view=ggml_view_2d(context,group.root,group.cols,rows,row,offset);
            if (!view || ggml_nbytes(view)!=matrix_bytes(row,rows) ||
                view->view_src!=group.root || view->view_offs!=offset)
                throw std::runtime_error("GGML packed projection view changes source storage");
            ggml_set_name(view,member.source->name.c_str());
            weights.emplace(member.source,view);
        }
    }
    weight_context_=context;
}

ggml_tensor *ProjectionPack::project(ggml_context *context, const TensorInfo *source,
                                    ggml_tensor *input, Cache &cache) const {
    const auto found=members_.find(source);
    if (found==members_.end() || !input || input->type!=GGML_TYPE_F32 ||
        input->ne[1]!=1 || input->ne[2]!=1 || input->ne[3]!=1 || !ggml_is_contiguous(input))
        return nullptr;
    const auto &group=groups_.at(found->second.group);
    if (input->ne[0]!=group.cols) return nullptr;
    if (!context || !weight_context_ || !group.root)
        throw std::logic_error("packed projection plan has no live weight metadata");
    if (cache.context && cache.context!=context)
        throw std::logic_error("packed projection cache cannot cross request graph contexts");
    cache.context=context;
    const Key key{group.root,input};
    auto hit=cache.outputs.find(key);
    ggml_tensor *output=nullptr;
    if (hit==cache.outputs.end()) {
        output=ggml_mul_mat(context,group.root,input);
        if (!output || output->type!=GGML_TYPE_F32 || output->ne[0]!=group.rows ||
            output->ne[1]!=1 || output->ne[2]!=1 || output->ne[3]!=1 || !ggml_is_contiguous(output))
            throw std::runtime_error("GGML packed projection result shape is invalid");
        ggml_set_name(output,"dense.projection_pack.output");
        cache.outputs.emplace(key,output);
    } else output=hit->second;
    const auto &member=group.members.at(found->second.member);
    const auto rows=std::int64_t(source->shape[1]);
    // With one activation column a 1D row view is genuinely contiguous: its
    // trailing dimensions/strides allow the existing 3D/4D reshapes unchanged.
    auto *result=ggml_view_1d(context,output,rows,std::size_t(member.row_offset)*sizeof(float));
    if (!result || !ggml_is_contiguous(result) || result->ne[1]!=1 || result->ne[2]!=1 || result->ne[3]!=1)
        throw std::runtime_error("GGML packed projection row view is not contiguous");
    return result;
}
} // namespace strata::dense
