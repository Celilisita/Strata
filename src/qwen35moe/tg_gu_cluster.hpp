#pragma once
#include "strata/kernels/native_tg_gu_cluster.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
inline bool tg_gu_cluster_flag(const char* value){
    if(!value)return false;
    if((value[0]!='0'&&value[0]!='1')||value[1]!='\0')
        throw std::invalid_argument("TG_GU_CLUSTER requires0 or1");
    return value[0]=='1';
}
inline bool tg_gu_cluster_eligible(bool decode,bool dynamic,bool mtp,int count,
    int gate_type,int up_type,int width,int hidden,int ranks)noexcept{
    return decode&&dynamic&&!mtp&&count==1&&gate_type==12&&up_type==12&&
        width==2048&&hidden==512&&ranks==8;
}
bool tg_gu_cluster_enabled();
struct TgGuClusterView {
    kernels::NativeTgGuCluster native;
    float* expert_gu;
    float* shared_gate_output;
    float* shared_up_output;
};
// Production-linked explicit A/B endpoint. Feature-off Runtime does not call it.
// false executes exactly the old five kernels on original buffers; true requests
// one fused kernel and returns false without mutation if resources do not fit.
bool tg_gu_cluster_provider(bool request,const TgGuClusterView&,void* stream);
}
