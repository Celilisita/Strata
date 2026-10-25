#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline bool use_parallel_ram_group(int required,const std::vector<RamKvCache*>&group){
    if(group.size()!=16)return false;
    for(auto *cache:group)if(cache->view().heads!=4||cache->view().keep_queries!=3)return false;
    static const int forced=[](){int device=0;check_cuda(cudaGetDevice(&device),"RAM group provider device");
        const auto caps=capabilities(device);if(caps.major!=12||caps.minor!=0)return 0;
        const char *p=std::getenv("STRATA_DENSE_RAM_GROUP_PARALLEL");
        return p&&(p[0]=='0'||p[0]=='1')&&!p[1]?int(p[0]-'0'):-1;}();
    if(forced>=0)return forced!=0;
    return required<=16384?false:true; // MEASURED_RAM_GROUP
}
}
