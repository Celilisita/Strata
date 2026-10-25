#include "strata/gemma4/native/profile.hpp"
#include "strata/gemma4/native/ffn.hpp"
namespace strata::gemma4::native {
Profile::Profile(size_t maximum):points_(maximum){
    if(maximum<2||maximum>4096)throw std::invalid_argument("Gemma bounded event capacity");
    try{for(auto& p:points_)primitive::check_cuda(cudaEventCreate(&p.event),"Gemma cost event create");}
    catch(...){for(auto& p:points_)if(p.event)cudaEventDestroy(p.event);throw;}
}
Profile::~Profile(){for(auto& p:points_)if(p.event)cudaEventDestroy(p.event);}
void Profile::mark(const char* name,int layer,int tokens,cudaStream_t stream){
    if(used_>=points_.size())throw std::runtime_error("Gemma bounded cost probe overflow");
    auto& p=points_[used_++];p.name=name;p.layer=layer;p.tokens=tokens;
    primitive::check_cuda(cudaEventRecord(p.event,stream),"Gemma stage event");
}
std::map<std::string,StageCost> Profile::summary(){
    std::map<std::string,StageCost> result;
    if(!used_)return result;
    primitive::check_cuda(cudaEventSynchronize(points_[used_-1].event),"Gemma completed cost join");
    for(size_t i=1;i<used_;++i){
        const auto& p=points_[i];if(std::string(p.name)=="begin")continue;
        float ms;primitive::check_cuda(cudaEventElapsedTime(&ms,points_[i-1].event,p.event),"Gemma cost interval");
        const std::string phase=p.tokens==1?"TG/":"PP/";
        auto& cost=result[phase+p.name];cost.milliseconds+=ms;++cost.calls;
        if(p.layer>=0&&(std::string(p.name)=="attention"||std::string(p.name)=="FFN")){
            auto& kind=result[phase+(p.layer%6==5?"global_":"SWA_")+p.name];kind.milliseconds+=ms;++kind.calls;
        }
    }
    return result;
}
}
