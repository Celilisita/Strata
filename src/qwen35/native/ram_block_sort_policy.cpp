#include "ram_block_sort.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
namespace strata::qwen35::native {
bool ram_block_sort_policy(int pages){
    static const bool enabled=[](){
        int device=0;check_cuda(cudaGetDevice(&device),"RAM block policy device");
        const auto caps=capabilities(device);if(caps.major!=12||caps.minor!=0)return false;
        const char *flag=std::getenv("STRATA_DENSE_RAM_BLOCK_SORT");
        if(flag&&(flag[0]=='0'||flag[0]=='1')&&!flag[1])return flag[0]=='1';
        return true; // MEASURED_RAM_BLOCK
    }();return enabled&&pages>2048&&pages<=4096;
}
}
