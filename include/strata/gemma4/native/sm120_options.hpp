#pragma once
#include <cstdlib>
#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include <map>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native {
struct Sm120Options {
    std::string model,draft;
    int context=262144,batch=2048,mtp=0;
    bool graph=true;
    static Sm120Options parse(int argc,char** argv){
        std::map<std::string,std::string> values;
        for(int i=1;i<argc;i+=2)if(i+1==argc||!values.emplace(argv[i],argv[i+1]).second)
            throw std::invalid_argument("Gemma native options require unique key/value pairs");
        auto take=[&](const char* key,std::string fallback=""){
            auto found=values.find(key);if(found==values.end())return fallback;
            auto value=found->second;values.erase(found);return value;};
        auto integer=[](const std::string& value){std::size_t used=0;int n=std::stoi(value,&used);
            if(used!=value.size())throw std::invalid_argument("Gemma native integer");return n;};
        const auto family=take("--family"),variant=take("--variant");
        Sm120Options o;o.model=take("--model");o.context=integer(take("--context","262144"));
        const auto slots=integer(take("--slots","1"));o.mtp=integer(take("--mtp","0"));
        const char* draft=std::getenv("STRATA_GEMMA4_MTP_DRAFT");
        o.draft=take("--draft",draft?draft:"");
        const auto kv=take("--kv-policy","full");
        const char* batch=std::getenv("STRATA_GEMMA4_MAX_BATCH");
        const char* graph=std::getenv("STRATA_GEMMA4_GRAPH");
        o.batch=integer(take("--max-batch",batch?batch:"2048"));
        const int graph_value=integer(graph?graph:"1");o.graph=graph_value!=0;
        if(!values.empty()||family!="gemma4_dense"||variant!=gemma_weight_variant(gemma_weight_format_requested())||o.model.empty()||
           o.context<4||o.context>262144||slots!=1||o.mtp<0||o.mtp>2||kv!="full"||o.batch<1||o.batch>8192||
           graph_value<0||graph_value>1)
            throw std::invalid_argument("unsupported Gemma native family/variant/context/MTP/KV/batch configuration");
        if((o.mtp==0&&!o.draft.empty())||(o.mtp&&
           (o.draft.empty()||(o.batch!=2048&&o.batch!=8192)||gemma_weight_format_requested()!=GemmaWeightFormat::Nvfp4)))
            throw std::invalid_argument("Gemma sampled MTP requires bound Q4 draft, native NVFP4 and batch2048 or8192");
        return o;
    }
};
int gemma4_sm120_serve(int argc,char** argv);
}
