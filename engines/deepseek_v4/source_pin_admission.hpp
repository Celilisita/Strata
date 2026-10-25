#pragma once
#include "expert_source_pin_backend.hpp"
#include "ggml.h"
#include "nlohmann/json.hpp"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {
// A measured prefix limits registration attempts, never expert reachability.
// Unregistered banks retain the original staging/CPU service path.
struct SourcePinAdmission {
    uint64_t limit=0, prefix_ranges=0;
    bool active=false;
    std::string endpoint,evidence,uuid;
    void report(uint64_t requested,uint64_t registered,const char* state)const {
        if(!active)return;
        std::fprintf(stderr,"strata_source_mapping_admission state=%s requested=%llu admission_limit=%llu admitted=%llu prefix_ranges=%llu endpoint=%s uuid=%s evidence_sha256=%s\n",
            state,(unsigned long long)requested,(unsigned long long)limit,(unsigned long long)registered,
            (unsigned long long)prefix_ranges,endpoint.c_str(),uuid.c_str(),evidence.c_str());
        std::fflush(stderr);
    }
};
inline SourcePinAdmission source_mapping_admission(const std::vector<ggml_tensor*>& sources,
        uint64_t page,uint64_t requested,int device,bool aliases) {
    SourcePinAdmission result;result.limit=requested;
    const char* path=std::getenv("STRATA_GLM53_PROVIDER_CURVES");
    if(!path||!*path)return result;
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("source mapping admission provider file unavailable");
    const auto size=file.tellg();
    if(size<=0||size>2*1024*1024)throw std::runtime_error("source mapping admission provider file size invalid");
    std::string text(size_t(size),'\0');file.seekg(0);
    if(!file.read(text.data(),std::streamsize(text.size())))throw std::runtime_error("source mapping admission provider file read failed");
    const auto document=nlohmann::json::parse(text);
    if(!document.contains("source_mapping_admission"))return result;
    const auto& record=document.at("source_mapping_admission");
    const auto fail=[](){throw std::runtime_error("source mapping admission identity/layout differs");};
    const auto number=[&](const nlohmann::json& value)->uint64_t {
        if(!value.is_number_unsigned())fail();
        return value.get<uint64_t>();
    };
    if(!aliases||!page||!record.is_object()||
       record.at("schema")!="strata-glm-source-mapping-admission-v1"||
       number(record.at("requested_budget_bytes"))!=requested)fail();
    result.limit=number(record.at("safe_prefix_bytes"));
    result.evidence=record.at("evidence_sha256").get<std::string>();
    if(!result.limit||result.limit>requested||result.limit%page||result.evidence.size()!=64)fail();
    for(unsigned char c:result.evidence)if(!std::isxdigit(c))fail();
    result.uuid=record.at("cuda_uuid").get<std::string>();
    char selected_uuid[41]{};uint32_t error=0;
    if(!selected_source_cuda_uuid(device,selected_uuid,sizeof(selected_uuid),error))
        throw std::runtime_error("source mapping admission CUDA UUID unavailable: "+std::to_string(error));
    if(result.uuid!=selected_uuid)fail();
    const auto& endpoint=record.at("prefix_endpoint");
    result.prefix_ranges=number(endpoint.at("source_count"));
    result.endpoint=endpoint.at("tensor_name").get<std::string>();
    const auto& layout=record.at("layout");
    if(!layout.is_array()||!result.prefix_ranges||result.prefix_ranges>sources.size()||
       layout.size()!=result.prefix_ranges)fail();
    uint64_t total=0;size_t index=0;
    for(const auto* tensor:sources) {
        if(index==layout.size())break;
        if(!tensor)fail();
        const uintptr_t begin=reinterpret_cast<uintptr_t>(tensor->data);
        const uint64_t bytes=ggml_nbytes(tensor);
        const uintptr_t maximum=(std::numeric_limits<uintptr_t>::max)();
        if(!tensor->data||bytes>maximum-begin||page-1>maximum-begin)fail();
        const uintptr_t first=(begin+page-1)/page*page,last=(begin+bytes)/page*page;
        if(first>=last)continue;
        const auto& row=layout.at(index);
        if(row.at("name")!=ggml_get_name(tensor)||row.at("type")!=ggml_type_name(tensor->type)||
           number(row.at("tensor_bytes"))!=bytes||number(row.at("page_bytes"))!=last-first)fail();
        const auto& shape=row.at("shape");if(!shape.is_array()||shape.size()!=4)fail();
        for(int d=0;d<4;++d)if(number(shape.at(d))!=uint64_t(tensor->ne[d]))fail();
        if(uint64_t(last-first)>result.limit-total)fail();
        total+=last-first;
        if(number(row.at("cumulative_page_bytes"))!=total)fail();
        ++index;
        if(index==layout.size()&&row.at("name")!=result.endpoint)fail();
    }
    if(index!=layout.size()||total!=result.limit)fail();
    result.active=true;return result;
}
} // namespace strata::deepseek_v4
