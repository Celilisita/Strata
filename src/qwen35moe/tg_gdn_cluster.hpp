#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe::detail {
inline bool tg_gdn_cluster_parse(const char* p) {
    if(!p||std::strcmp(p,"0")==0)return false;
    if(std::strcmp(p,"1")==0)return true;
    throw std::invalid_argument("TG_GDN_CLUSTER requires0 or1");
}
inline bool tg_gdn_cluster_enabled() {
    static const bool value=tg_gdn_cluster_parse(std::getenv("STRATA_QWEN35_TG_GDN_CLUSTER"));
    return value; // Native startup configuration; captured graphs do not hot-switch.
}
inline bool tg_gdn_cluster_eligible(bool decode,bool dynamic,int slots,bool paired,
    bool mtp,bool sinks,bool warp_norm) {
    return decode&&dynamic&&slots==1&&!paired&&!mtp&&!sinks&&!warp_norm;
}
}
