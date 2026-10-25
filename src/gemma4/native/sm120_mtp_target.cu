#include "strata/gemma4/native/sm120_mtp_target.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native::detail {
namespace {
constexpr unsigned InvalidMtp=256u;
__global__ void begin(Sm120TargetState* state,const int* position,int rows,int capacity,unsigned* invalid){
    const int start=*position;
    const bool valid=start>=1&&rows>=1&&rows<=3&&start<=capacity-rows;
    if(!valid)atomicOr(invalid,InvalidMtp);
    *state={start,rows,0,start,1,*invalid};
}
__global__ void commit(Sm120TargetState* state,const float* features,float* seed,
    int* pending,int* position,const int* proposed_count,const int* next,
    const unsigned* round_invalid,int capacity,unsigned* invalid){
    __shared__ int count,token,start,rows,valid;
    if(!threadIdx.x){
        count=*proposed_count;token=*next;start=state->start;rows=state->rows;
        valid=state->ready==1&&rows>=1&&rows<=3&&*position==start&&start>=1&&
            start<=capacity-rows&&count>=0&&count<=rows&&!state->invalid&&
            !*invalid&&!*round_invalid&&(!count||(token>=0&&token<262144));
        if(!valid)atomicOr(invalid,InvalidMtp);
    }
    __syncthreads();
    if(valid&&count)for(int i=threadIdx.x;i<rows*5376;i+=256)
        if(!isfinite(features[i]))atomicOr(invalid,InvalidMtp);
    __syncthreads();
    const bool publish=valid&&!*invalid&&count>0;
    if(publish)for(int i=threadIdx.x;i<5376;i+=256)
        seed[i]=features[std::size_t(count-1)*5376+i];
    __syncthreads();
    if(!threadIdx.x){
        if(publish){*pending=token;*position=start+count;}
        state->count=publish?count:0;state->position=*position;
        state->invalid=*invalid|*round_invalid;state->ready=0;
    }
}
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
void check(cudaError_t error,const char* text){
    if(error!=cudaSuccess)throw std::runtime_error(std::string(text)+": "+cudaGetErrorString(error));
}
}
void begin_sm120_target(Sm120TargetState* state,const int* position,int rows,int capacity,
    unsigned* invalid,cudaStream_t stream){
    if(!aligned(state)||!aligned(position)||!aligned(invalid)||!stream||rows<1||rows>3||capacity<rows)
        throw std::invalid_argument("Gemma MTP begin metadata");
    begin<<<1,1,0,stream>>>(state,position,rows,capacity,invalid);
    check(cudaGetLastError(),"Gemma MTP target begin");
}
void commit_sm120_target(Sm120TargetState* state,const float* features,float* seed,
    int* pending,int* position,const int* count,const int* next,const unsigned* round_invalid,
    int capacity,unsigned* invalid,cudaStream_t stream){
    if(!aligned(state)||!aligned(features)||!aligned(seed)||!aligned(pending)||!aligned(position)||
       !aligned(count)||!aligned(next)||!aligned(round_invalid)||!aligned(invalid)||!stream||capacity<1)
        throw std::invalid_argument("Gemma MTP sampled commit metadata");
    commit<<<1,256,0,stream>>>(state,features,seed,pending,position,count,next,round_invalid,capacity,invalid);
    check(cudaGetLastError(),"Gemma MTP accepted prefix commit");
}
}
