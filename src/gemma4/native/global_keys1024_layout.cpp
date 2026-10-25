#include "global_keys1024_ops.hpp"
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native::global_keys1024 {
namespace {
std::size_t aligned(std::size_t n){return(n+255)&~std::size_t(255);}
void valid(int t){if(t<1||t>8192)throw std::invalid_argument("Gemma PP TC bounded batch");}
}
bool pp_tc_requested(){
    const char* raw=std::getenv("STRATA_GEMMA_PP_GLOBAL_KEYS1024");const std::string value=raw?raw:"0";
    if(value!="0"&&value!="1")throw std::invalid_argument("STRATA_GEMMA_PP_GLOBAL_KEYS1024 requires0or1");
    return value=="1";
}
std::size_t pp_tc_workspace_bytes(int t){
    valid(t);using namespace pp_tc_detail;
    return aligned(std::size_t(t)*Heads*512*2)+2*aligned(std::size_t(Keys)*4096*2)+
        aligned(std::size_t(Heads)*Query*Keys*2)+aligned(std::size_t(Heads)*Query*Keys*4)+
        aligned(std::size_t(Heads)*Query*512*4)+2*aligned(std::size_t(t)*Heads*4)+
        2*aligned(std::size_t(Heads)*Query*4)+6*aligned(Heads*sizeof(void*));
}
namespace pp_tc_detail {
Buffers bind(void* data,std::size_t bytes,int t){
    valid(t);if(!data||reinterpret_cast<std::uintptr_t>(data)%256||bytes<pp_tc_workspace_bytes(t))
        throw std::invalid_argument("Gemma PP TC stable arena");
    auto* at=static_cast<unsigned char*>(data);
    const auto take=[&](std::size_t n){void* p=at;at+=aligned(n);return p;};
    Buffers b{};
    b.q=static_cast<half*>(take(std::size_t(t)*Heads*512*2));
    b.k=static_cast<half*>(take(std::size_t(Keys)*4096*2));b.v=static_cast<half*>(take(std::size_t(Keys)*4096*2));
    b.probability=static_cast<half*>(take(std::size_t(Heads)*Query*Keys*2));
    b.scores=static_cast<float*>(take(std::size_t(Heads)*Query*Keys*4));
    b.partial=static_cast<float*>(take(std::size_t(Heads)*Query*512*4));
    b.maximum=static_cast<float*>(take(std::size_t(t)*Heads*4));b.sum=static_cast<float*>(take(std::size_t(t)*Heads*4));
    b.a=static_cast<float*>(take(std::size_t(Heads)*Query*4));b.b=static_cast<float*>(take(std::size_t(Heads)*Query*4));
    b.kt=static_cast<const void**>(take(Heads*sizeof(void*)));b.vt=static_cast<const void**>(take(Heads*sizeof(void*)));
    b.qt=static_cast<const void**>(take(Heads*sizeof(void*)));b.pt=static_cast<const void**>(take(Heads*sizeof(void*)));
    b.st=static_cast<void**>(take(Heads*sizeof(void*)));b.ot=static_cast<void**>(take(Heads*sizeof(void*)));
    return b;
}
}
}
