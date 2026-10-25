#include "router_prediction.hpp"
#include "router_prefetch.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
bool router_prefetch_enabled() {
    const char* flag=std::getenv("STRATA_DSV4_ROUTER_PREFETCH");
    if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::runtime_error("Router prefetch must be0 or1");return flag&&flag[0]=='1';
}
bool router_prediction_enabled() {
    const char* flag=std::getenv("STRATA_DSV4_ROUTER_PREDICTION_TRACE");
    if(flag&&std::strcmp(flag,"0")&&std::strcmp(flag,"1"))throw std::runtime_error("Router prediction trace must be0 or1");
    return (flag&&flag[0]=='1')||router_prefetch_enabled();
}
}
