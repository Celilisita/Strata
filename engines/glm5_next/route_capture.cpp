#include "route_capture.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "../deepseek_v4/cache_cost_audit.hpp"
#include "../deepseek_v4/device_cache.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace strata::glm5_next {
using Json=nlohmann::ordered_json;
bool RouteCapture::requested() {
    const char* p=std::getenv("STRATA_GLM53_ROUTE_CAPTURE");
    if(!p||!std::strcmp(p,"0"))return false;
    if(std::strcmp(p,"1"))throw std::runtime_error("Route capture must be0 or1");
    return true;
}
RouteCapture::RouteCapture(ggml_context* context) {
    tensor_=ggml_new_tensor_1d(context,GGML_TYPE_I8,route_trace_bytes);
    ggml_set_name(tensor_,"strata_glm53.bounded_route_trace");
    ggml_set_input(tensor_);
}
void RouteCapture::initialize() {
    const RouteTraceHeader header{};
    ggml_backend_tensor_set(tensor_,&header,0,sizeof(header));
}
void RouteCapture::begin() {
    RouteTraceHeader header{};header.enabled=1;
    ggml_backend_tensor_set(tensor_,&header,0,sizeof(header));
    active_=true;seed_=nullptr;
}
void RouteCapture::seed(const llama_model& model,llama_moe_cache& cache,const std::vector<uint32_t>& prior) {
    if(!active_)return;
    if(model.arch!=LLM_ARCH_GLM5_NEXT||prior.size()!=model.layers.size()*288+1)
        throw std::runtime_error("Route capture seed model/frequency dimensions differ");
    Json layers=Json::array(),groups=Json::array();std::map<const void*,int> group_ids;
    for(size_t il=0;il<model.layers.size();++il) {
        const auto& layer=model.layers[il];if(!layer.ffn_up_exps)continue;
        const std::array<ggml_tensor*,3> original{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps};
        auto* bank=cache.get_experts(layer.ffn_up_exps);
        if(!bank||!bank->data||bank->ne[2]<8)throw std::runtime_error("Trace seed bank unavailable");
        const bool fresh=!group_ids.count(bank->data);
        const int group=fresh?int(group_ids.size()):group_ids.at(bank->data);
        group_ids[bank->data]=group;
        size_t bytes=0;Json types=Json::array(),strides=Json::array();
        for(auto* source:original){bytes+=source->nb[2];types.push_back(ggml_type_name(source->type));strides.push_back(source->nb[2]);}
        if(fresh)groups.push_back({{"group",group},{"slots",bank->ne[2]},{"expert_bytes",bytes},
            {"types",types},{"strides",strides}});
        const auto* directory=cache.strata_directory(int(il));
        if(!directory||!directory->data||ggml_nelements(directory)!=288)throw std::runtime_error("Trace seed directory absent");
        const auto* map=static_cast<const int32_t*>(directory->data);
        layers.push_back({{"layer",il},{"group",group},{"directory",std::vector<int32_t>(map,map+288)},
            {"frequency",std::vector<uint32_t>(prior.begin()+il*288,prior.begin()+(il+1)*288)}});
    }
    const auto capacity=cache.strata_cost_snapshot();
    seed_={{"groups",groups},{"layers",layers},{"cache_weight_bytes",capacity.weight_tensor_bytes},
        {"cache_slots",capacity.permanent_slots},{"frequency_samples",prior.back()}};
}
void RouteCapture::write(const std::string& path) {
    if(!active_||path.empty()||seed_.is_null())throw std::runtime_error("Route capture has no matching TG phase seed");
    RouteTraceHeader header{};ggml_backend_tensor_get(tensor_,&header,0,sizeof(header));
    const uint32_t count=std::min(header.count,route_trace_capacity);
    std::vector<RouteTraceRecord> records(count);
    if(count)ggml_backend_tensor_get(tensor_,records.data(),sizeof(header),records.size()*sizeof(RouteTraceRecord));
    for(const auto& row:records) {
        if(row.layer<3||row.layer>=46||(row.count!=8&&row.count!=16)||(row.missing>>row.count))
            throw std::runtime_error("Route capture record identity differs");
        for(uint32_t i=0;i<row.count;++i)if(row.ids[i]>=288)throw std::runtime_error("Route capture expert key differs");
    }
    std::ofstream payload(path,std::ios::binary);
    payload.write(reinterpret_cast<const char*>(&header),sizeof(header));
    payload.write(reinterpret_cast<const char*>(records.data()),records.size()*sizeof(RouteTraceRecord));
    if(!payload)throw std::runtime_error("Bounded route payload export failed");
    const Json meta={{"schema","strata-glm53-routes-v1"},{"records",count},{"attempted",header.count},
        {"overflow",header.overflow},{"capacity",route_trace_capacity},{"record_bytes",sizeof(RouteTraceRecord)},
        {"header_bytes",sizeof(header)},{"binary_path",path},{"complete",header.overflow==0},
        {"timing","Optional device writes plus final bounded export are included in request timing"},{"seed",seed_}};
    std::ofstream metadata(path+".json");metadata<<meta.dump(2)<<'\n';
    if(!metadata)throw std::runtime_error("Route metadata export failed");
    initialize();active_=false;
}
}

namespace strata::deepseek_v4 {
void DeviceExpertCache::begin_route_capture() {
    if(!route_capture_)throw std::runtime_error("Route capture was not provisioned for this context");
    settle_background();route_capture_->begin();
}
void DeviceExpertCache::write_route_capture(const std::string& path) {
    if(!route_capture_)throw std::runtime_error("Route capture was not provisioned for this context");
    settle_background();route_capture_->write(path);
}
}
