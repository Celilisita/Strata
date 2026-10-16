#include "packed_expert_store.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
bool packed_experts_requested() {
    const char* value=std::getenv("STRATA_DSV4_PACKED_EXPERTS");
    if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
        throw std::runtime_error("packed expert store must be0 or1");
    return value&&std::strcmp(value,"1")==0;
}
PackedExpertStore::PackedExpertStore(ggml_backend_t backend,const std::vector<ggml_tensor*>& sources) {
    if(!backend||sources.empty()||sources.size()%3)throw std::runtime_error("packed expert inventory incomplete");
    device_=ggml_backend_get_device(backend);
    const auto host_type=ggml_backend_dev_host_buffer_type(device_);
    if(!host_type||std::strcmp(ggml_backend_buft_name(host_type),"CUDA_Host"))
        throw std::runtime_error("packed experts require owned pinned CUDA host storage");
    const auto start=std::chrono::steady_clock::now();
    layers_.reserve(sources.size()/3);
    try {
        for(size_t layer=0;layer<sources.size()/3;++layer) {
            layers_.push_back({});auto& owned=layers_.back();
            for(int role=0;role<3;++role) {
                const auto* tensor=sources[3*layer+role];
                if(!tensor||!tensor->data||tensor->type!=GGML_TYPE_MXFP4||tensor->ne[2]!=256||
                   tensor->ne[3]!=1||!ggml_is_contiguous(tensor)||!tensor->buffer||!ggml_backend_buffer_is_host(tensor->buffer))
                    throw std::runtime_error("packed expert source payload contract differs");
                owned.original[role]=sources[3*layer+role];owned.stride+=tensor->nb[2];
            }
            if(owned.stride!=13369344||owned.stride%4096)
                throw std::runtime_error("packed expert triplet geometry/alignment differs");
            const size_t bytes=256*owned.stride;
            ++metrics_.attempts;metrics_.budget_bytes+=bytes;
            owned.buffer=ggml_backend_buft_alloc_buffer(host_type,bytes);
            if(!owned.buffer||ggml_backend_buffer_get_type(owned.buffer)!=host_type||ggml_backend_buffer_get_size(owned.buffer)<bytes)
                throw std::runtime_error("packed expert allocation failed or fell back to pageable memory at layer "+
                    std::to_string(layer)+" after "+std::to_string(metrics_.registered_bytes)+" pinned bytes");
            owned.data=static_cast<char*>(ggml_backend_buffer_get_base(owned.buffer));
            if(!owned.data||reinterpret_cast<uintptr_t>(owned.data)%4096)
                throw std::runtime_error("packed expert base is not page aligned");
            size_t offset=0;
            for(int role=0;role<3;++role) {
                const auto* src=owned.original[role];auto& view=owned.view[role];view=*src;
                view.data=owned.data+offset;view.buffer=owned.buffer;view.view_src=nullptr;view.view_offs=0;
                view.nb[2]=owned.stride;view.nb[3]=bytes;
                for(int expert=0;expert<256;++expert) {
                    const auto* input=static_cast<const char*>(src->data)+expert*src->nb[2];
                    auto* output=owned.data+expert*owned.stride+offset;
                    std::memcpy(output,input,src->nb[2]);
                    if(std::memcmp(output,input,src->nb[2]))throw std::runtime_error("packed expert payload differs");
                }
                const auto begin=reinterpret_cast<uintptr_t>(src->data);
                ranges_.push_back({begin,begin+ggml_nbytes(src),layer,role});offset+=src->nb[2];
            }
            metrics_.registered_bytes+=bytes;
        }
        std::sort(ranges_.begin(),ranges_.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
        for(size_t i=1;i<ranges_.size();++i)if(ranges_[i-1].end>ranges_[i].begin)
            throw std::runtime_error("packed expert original source ranges overlap");
    } catch(...) {release();throw;}
    metrics_.registration_wall_ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count());
}
void PackedExpertStore::release() noexcept {
    for(auto& layer:layers_)if(layer.buffer){ggml_backend_buffer_free(layer.buffer);layer.buffer=nullptr;}
}
PackedExpertStore::~PackedExpertStore(){release();}
bool PackedExpertStore::matches(ggml_backend_t backend,const std::vector<ggml_tensor*>& sources) const {
    if(!backend||ggml_backend_get_device(backend)!=device_||sources.size()!=3*layers_.size())return false;
    for(size_t i=0;i<sources.size();++i)if(layers_[i/3].original[i%3]!=sources[i])return false;
    return true;
}
const PackedExpertStore::Layer& PackedExpertStore::find(const ggml_tensor* gate) const {
    const auto found=std::find_if(layers_.begin(),layers_.end(),[&](const auto& layer){return layer.original[0]==gate;});
    if(found==layers_.end())throw std::runtime_error("packed expert key absent from source inventory");
    return *found;
}
PackedExpertSpan PackedExpertStore::span(const ggml_tensor* gate,int first,int count) const {
    if(first<0||count<1||first+count>256)throw std::runtime_error("packed expert span outside inventory");
    const auto& layer=find(gate);
    return {layer.data+first*layer.stride,size_t(count)*layer.stride,layer.stride,
        layer.original[0]->nb[2],layer.original[1]->nb[2]};
}
PackedExpertStore::Projection PackedExpertStore::project(const void* original,size_t bytes) const {
    const auto address=reinterpret_cast<uintptr_t>(original);
    auto found=std::upper_bound(ranges_.begin(),ranges_.end(),address,[](uintptr_t p,const Range& r){return p<r.begin;});
    if(found==ranges_.begin())return {nullptr,0};--found;
    if(address>=found->end)return {nullptr,0};
    const auto& layer=layers_[found->layer];const size_t width=layer.original[found->role]->nb[2];
    const size_t offset=address-found->begin,expert=offset/width,in_role=offset%width;
    return {static_cast<const char*>(layer.view[found->role].data)+expert*layer.stride+in_role,
        std::min(bytes,width-in_role)};
}
Mxfp4CpuBank PackedExpertStore::cpu_bank(const ggml_tensor* gate,float clamp) const {
    const auto& layer=find(gate);return {&layer.view[0],&layer.view[1],&layer.view[2],clamp};
}
}
