#pragma once
#include "strata/kernels/native_tg_quant_down_pair.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
inline bool tg_quant_down_pair_flag(const char* value){
    if(!value)return false;
    if((value[0]!='0'&&value[0]!='1')||value[1]!='\0')
        throw std::invalid_argument("TG_QUANT_DOWN_PAIR requires0 or1");
    return value[0]=='1';
}
inline bool tg_quant_down_pair_dependencies(bool pair,bool gu,bool row4){
    if(pair&&(!gu||!row4))throw std::invalid_argument("TG_QUANT_DOWN_PAIR requires TG_GU_CLUSTER1 and TG_DOWN_ROW4=1");
    return pair;
}
inline bool tg_quant_down_pair_eligible(bool fused_gu,bool decode,bool dynamic,bool mtp,
    int count,int down_type,int shared_type,int width,int outputs,int ranks)noexcept{
    return fused_gu&&decode&&dynamic&&!mtp&&count==1&&down_type==shared_type&&
        (down_type==12||down_type==13||down_type==14)&&width==512&&outputs==2048&&ranks==8;
}
bool tg_quant_down_pair_enabled();
// Production-linked explicit4-vs2-kernel endpoint; false returns the original
// quant/routedDown/sharedQuant/sharedSmallK sequence, not Runtime's private gate.
bool tg_quant_down_pair_provider(bool request,int type,const kernels::NativeTgQuantDownPair&,void* stream);
}
