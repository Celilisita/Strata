#include "strata/gemma4/native/verify_model.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native::detail {
namespace {
__global__ void begin(VerifyPacket* packet,const unsigned* source,int rows){
    packet->rows=rows;packet->response={};
    if(source)atomicOr(&packet->invalid,*source);
}
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
}
void begin_main_verify(VerifyPacket* packet,const unsigned* source,int rows,cudaStream_t stream){
    if(!aligned(packet)||(source&&!aligned(source))||(rows!=2&&rows!=3)||!stream)
        throw std::invalid_argument("Gemma main verify metadata lease");
    const auto p=reinterpret_cast<std::uintptr_t>(packet),s=reinterpret_cast<std::uintptr_t>(source);
    if(p>UINTPTR_MAX-sizeof(VerifyPacket)||(source&&(s>UINTPTR_MAX-4||
       (s<p+sizeof(VerifyPacket)&&p<s+4&&source!=&packet->invalid))))
        throw std::invalid_argument("Gemma main verify diagnostic aliases packet");
    begin<<<1,1,0,stream>>>(packet,source,rows);
    const auto error=cudaGetLastError();if(error!=cudaSuccess)
        throw std::runtime_error(std::string("Gemma main verify metadata: ")+cudaGetErrorString(error));
}
}
