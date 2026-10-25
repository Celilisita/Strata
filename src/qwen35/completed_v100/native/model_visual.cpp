#include "strata/qwen35/completed_v100/native/model.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>

namespace strata::completed_v100::qwen35::native {
std::unique_ptr<DenseVisualInput> read_visual_prompt(const std::string& path, int context, int width, int vocab) {
    std::ifstream in(path, std::ios::binary); int h[3]{};
    if (!in.read(reinterpret_cast<char*>(h), sizeof(h)) || h[0]!=0x31505644 ||
        h[1]<2 || h[1]>=context || h[2]<1 || h[2]>64 || width!=5120)
        throw std::invalid_argument("invalid Dense visual payload header");
    auto data=std::make_unique<DenseVisualInput>(); data->width=width;
    data->ids.resize(size_t(h[1])); data->positions.resize(size_t(h[1]));
    if (!in.read(reinterpret_cast<char*>(data->ids.data()), h[1]*sizeof(int)) ||
        !in.read(reinterpret_cast<char*>(data->positions.data()), h[1]*3*sizeof(int)))
        throw std::invalid_argument("truncated Dense visual tokens/positions");
    for (int id:data->ids) if (id<0 || id>=vocab) throw std::invalid_argument("Dense visual token outside vocabulary");
    for (const auto& p:data->positions) for (int v:p)
        if (v<0 || v>1000000000) throw std::invalid_argument("Dense visual position outside bounds");
    int previous=0;
    for (int i=0;i<h[2];++i) {
        int s[2]{};
        if (!in.read(reinterpret_cast<char*>(s),sizeof(s)) || s[0]<previous || s[1]<1 || s[1]>16384 || s[0]>h[1]-s[1])
            throw std::invalid_argument("invalid Dense visual span");
        VisualSpan span{s[0],s[1],std::vector<float>(size_t(s[1])*width)};
        if (!in.read(reinterpret_cast<char*>(span.rows.data()),std::streamsize(span.rows.size()*4)) ||
            std::any_of(span.rows.begin(),span.rows.end(),[](float x){return !std::isfinite(x);}))
            throw std::invalid_argument("invalid Dense visual embedding payload");
        previous=s[0]+s[1]; data->spans.push_back(std::move(span));
    }
    if (in.peek()!=std::char_traits<char>::eof()) throw std::invalid_argument("Dense visual payload trailing data");
    return data;
}
void DenseVisualInput::upload(int context,cudaStream_t stream) {
    auto all=positions;
    int logical=*(std::max_element)(positions.back().begin(),positions.back().end())+1;
    for (int i=int(all.size());i<context;++i,++logical) all.push_back({logical,logical,logical});
    table=static_cast<int*>(storage.allocate(size_t(context)*3*4));
    check_cuda(cudaMemcpyAsync(table,all.data(),size_t(context)*3*4,cudaMemcpyHostToDevice,stream),"Dense visual positions");
    for (auto& span:spans) {
        auto p=static_cast<float*>(storage.allocate(span.rows.size()*4)); span.device=p;
        check_cuda(cudaMemcpyAsync(p,span.rows.data(),span.rows.size()*4,cudaMemcpyHostToDevice,stream),"Dense visual embeddings");
    }
    // The upload's host vectors and position extension survive all DMA consumers.
    check_cuda(cudaStreamSynchronize(stream),"Dense visual upload lifetime");
}
void DenseVisualInput::overlay(float* out,int start,int count,cudaStream_t stream) const {
    for (const auto& span:spans) {
        const int first=(std::max)(start,span.first),end=(std::min)(start+count,span.first+span.count);
        if (first>=end) continue;
        check_cuda(cudaMemcpyAsync(out+size_t(first-start)*width,span.device+size_t(first-span.first)*width,
            size_t(end-first)*width*4,cudaMemcpyDeviceToDevice,stream),"Dense visual embedding span");
    }
}
void DenseModel::set_visual_input(std::unique_ptr<DenseVisualInput> input,cudaStream_t stream) {
    if(!input&&!visual_)return;
    if (input && (spec_.rope_dim!=64 || spec_.rope_sections[0]!=11 || spec_.rope_sections[1]!=11 ||
        spec_.rope_sections[2]!=10 || spec_.rope_sections[3]!=0))
        throw std::invalid_argument("Dense visual IMRoPE geometry differs from projector");
    check_cuda(cudaStreamSynchronize(stream),"Dense visual request boundary");
    visual_=std::move(input);
    if (visual_) visual_->upload(limits_.context,stream);
    attention_->visual_positions(visual_positions());
}
}
