#pragma once
#include <cstdlib>
namespace strata::qwen35::native {
inline int ram_six_mode(){
    static const int selected=[](){
        int device=0;check_cuda(cudaGetDevice(&device),"RAM provider device");
        const auto caps=capabilities(device);
        if(caps.major!=12||caps.minor!=0)return 0;
        const char *flag=std::getenv("STRATA_DENSE_RAM_SIX");
        if(flag&&flag[0]>='0'&&flag[0]<='2'&&!flag[1])return int(flag[0]-'0');
        return 1; // MEASURED_RAM_SIX
    }();return selected;
}
}
