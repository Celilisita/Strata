#include "sm120_global_flash.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
std::size_t aligned(std::size_t n){return(n+255)&~std::size_t(255);}
void valid_tokens(int tokens){if(tokens<1||tokens>2048)throw std::invalid_argument("Gemma global flash batch<=2048");}
}
bool global_flash_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA_PP_GLOBAL_FLASH");
        if(!value||std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("STRATA_GEMMA_PP_GLOBAL_FLASH requires0or1");
    }();return enabled;
}
int global_flash_key_bound(int active_upper){
    if(active_upper<1||active_upper>262144)throw std::invalid_argument("Gemma global flash actual capacity");
    return(active_upper+255)&~255;
}
std::size_t global_flash_workspace_bytes(int tokens,int key_upper){
    valid_tokens(tokens);const int keys=global_flash_key_bound(key_upper);
    return aligned(sizeof(unsigned))+2*aligned(std::size_t(keys)*4*512*2)+
        aligned(std::size_t(tokens)*keys*2);
}
namespace global_flash_detail {
Buffers bind(AttentionWorkspace w,int tokens,int key_upper){
    const auto bytes=global_flash_workspace_bytes(tokens,key_upper);
    if(!w.data||reinterpret_cast<std::uintptr_t>(w.data)%256||w.bytes<bytes)
        throw std::invalid_argument("Gemma global flash owned arena");
    const int keys=global_flash_key_bound(key_upper);auto* p=static_cast<unsigned char*>(w.data);
    Buffers b{};b.invalid=reinterpret_cast<unsigned*>(p);p+=aligned(sizeof(unsigned));
    b.key=p;p+=aligned(std::size_t(keys)*4*512*2);
    b.value=p;p+=aligned(std::size_t(keys)*4*512*2);
    b.mask=p;b.keys=keys;return b;
}
}
}
