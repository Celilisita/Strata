#include "strata/gemma4/native/verify_model.hpp"
#include <stdexcept>
#include <string>
namespace strata::gemma4::native::detail {
namespace {
__device__ bool eog(int token,VerifyPolicy policy){
    for(int i=0;i<policy.eog_count;++i)if(token==policy.eog_ids[i])return true;
    return false;
}
__global__ void begin_tail(VerifyPacket* packet,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy policy){
    packet->rows=1;packet->response={};packet->response.position=*position;
    // Bad admission poisons this request. The host must reset after an invalid
    // response; a queued graph cannot cancel the following ordinary T1 kernels.
    if(*position<1||*position>=capacity||*pending<0||*pending>=vocabulary||
       *policy.remaining<=0||eog(*pending,policy))atomicOr(&packet->invalid,16u);
}
__global__ void finish_tail(VerifyPacket* packet,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy policy){
    const int start=packet->response.position,token=*pending,remaining=*policy.remaining;
    VerifyResponse response{};response.position=*position;
    if(packet->invalid||start<1||start>=capacity||*position!=start+1||
       token<0||token>=vocabulary||remaining<=0){
        atomicOr(&packet->invalid,16u);response.reason=VerifyStop::Invalid;
        response.finished=1;packet->response=response;return;
    }
    response.count=1;response.ids[0]=token;*policy.remaining=remaining-1;
    if(eog(token,policy))response.reason=VerifyStop::Eog;
    else if(remaining==1)response.reason=VerifyStop::Budget;
    response.finished=response.reason!=VerifyStop::Continue;packet->response=response;
}
void check(const char* label){
    const auto error=cudaGetLastError();if(error!=cudaSuccess)
        throw std::runtime_error(std::string(label)+cudaGetErrorString(error));
}
}
void begin_base_tail(VerifyPacket* packet,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy policy,cudaStream_t stream){
    begin_tail<<<1,1,0,stream>>>(packet,position,pending,capacity,vocabulary,policy);
    check("Gemma base tail begin: ");
}
void finish_base_tail(VerifyPacket* packet,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy policy,cudaStream_t stream){
    finish_tail<<<1,1,0,stream>>>(packet,position,pending,capacity,vocabulary,policy);
    check("Gemma base tail commit: ");
}
}
