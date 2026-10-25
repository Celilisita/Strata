#include "codec.cuh"
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

using namespace strata::qwen35::native;
namespace {
struct Statistics {double error2,norm2;unsigned maximum_error_bits,zero_scales;};
struct Result {float global_scale,maximum_error;double error2,norm2;unsigned long long zero_scales,elements;};
thread_local std::string last_error;
void checked(cudaError_t status,const char *operation){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
struct Allocation {
    void *pointer=nullptr;
    explicit Allocation(size_t bytes){checked(cudaMalloc(&pointer,bytes),"conversion bounded allocation");}
    ~Allocation(){cudaFree(pointer);}
};
__device__ double sum(double value){
    for(int d=16;d;d>>=1)value+=__shfl_xor_sync(0xffffffff,value,d);return value;
}
__global__ void decode_original(const unsigned char *raw,float *output,size_t count,Quant type,unsigned *maximum){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const auto *block=raw+(i/256)*(type==Quant::Q4K?144:210);
    const float value=block_element(block,type,int(i%256),BlockLayout::Gguf);
    output[i]=value;
    float a=fabsf(value);
    for(int d=16;d;d>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
    if(threadIdx.x%32==0)atomicMax(maximum,__float_as_uint(a));
}
__device__ float fp4(unsigned char code){
    constexpr float table[8]={0,.5f,1,1.5f,2,3,4,6};
    return (code&8)?-table[code&7]:table[code&7];
}
__device__ float fp8(unsigned char code){return __half2float(__nv_cvt_fp8_to_halfraw(code,__NV_E4M3));}
__global__ void encode(const float *values,unsigned char *output,size_t count,float global,Statistics *statistics){
    // Every eight lanes encode one block16. Pair halves match GGUF's nibble order.
    __shared__ double errors[8],norms[8];
    __shared__ unsigned zeros[8],maximums[8];
    const size_t pair=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const bool live=pair<count/2;
    const size_t group=pair/8;const int lane=threadIdx.x%8;
    const float x=live?values[group*16+lane]:0,y=live?values[group*16+lane+8]:0;
    float maximum=fmaxf(fabsf(x),fabsf(y));
    for(int d=4;d;d>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,8));
    const int center=__nv_cvt_float_to_fp8(maximum/(6.f*global),__NV_SATFINITE,__NV_E4M3);
    double best=1.e300;unsigned char best_code=0,best_scale=0;float bx=0,by=0;
    for(int choice=0;choice<3;++choice){
        const int shift=choice==0?0:choice==1?-1:1;
        const auto scale_code=static_cast<unsigned char>(max(0,min(126,center+shift)));
        const float scale=fp8(scale_code)*global,inverse=scale>0?1.f/scale:0;
        const auto code=__nv_cvt_float2_to_fp4x2(make_float2(x*inverse,y*inverse),__NV_E2M1,cudaRoundNearest);
        const float dx=fp4(code&15)*scale,dy=fp4(code>>4)*scale;
        const double ex=double(dx)-x,ey=double(dy)-y;
        double error=ex*ex+ey*ey;
        for(int d=4;d;d>>=1)error+=__shfl_xor_sync(0xffffffff,error,d,8);
        if(error<best){best=error;best_code=code;best_scale=scale_code;bx=dx;by=dy;}
    }
    if(live){
        const size_t base=(group/4)*36;
        output[base+4+(group%4)*8+lane]=best_code;
        if(!lane)output[base+group%4]=best_scale;
    }
    double error=live&&!lane?best:0,norm=live?double(x)*x+double(y)*y:0;
    unsigned zero=live&&!lane&&best_scale==0?1:0;
    float difference=live?fmaxf(fabsf(bx-x),fabsf(by-y)):0;
    error=sum(error);norm=sum(norm);
    for(int d=16;d;d>>=1){
        zero+=__shfl_xor_sync(0xffffffff,zero,d);
        difference=fmaxf(difference,__shfl_xor_sync(0xffffffff,difference,d));
    }
    const int warp=threadIdx.x/32;
    if(threadIdx.x%32==0){errors[warp]=error;norms[warp]=norm;zeros[warp]=zero;maximums[warp]=__float_as_uint(difference);}
    __syncthreads();
    if(threadIdx.x<32){
        const int i=threadIdx.x;
        error=sum(i<8?errors[i]:0);norm=sum(i<8?norms[i]:0);
        zero=i<8?zeros[i]:0;unsigned high=i<8?maximums[i]:0;
        for(int d=16;d;d>>=1){zero+=__shfl_xor_sync(0xffffffff,zero,d);high=max(high,__shfl_xor_sync(0xffffffff,high,d));}
        if(!i){atomicAdd(&statistics->error2,error);atomicAdd(&statistics->norm2,norm);
            atomicAdd(&statistics->zero_scales,zero);atomicMax(&statistics->maximum_error_bits,high);}
    }
}
}
extern "C" __declspec(dllexport) const char *sfp4_error(){return last_error.c_str();}
extern "C" __declspec(dllexport) int sfp4_convert(const void *source,size_t count,int type,void *destination,Result *result){
    try{
        if(!source||!destination||!result||!count||count%256||(type!=12&&type!=14))
            throw std::invalid_argument("original GPU conversion arguments");
        int device;checked(cudaGetDevice(&device),"conversion device");cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties,device),"conversion capabilities");
        if(properties.major!=12||properties.minor!=0)throw std::runtime_error("conversion gate SM120 only");
        const size_t raw_bytes=count/256*(type==12?144:210),output_bytes=count/64*36;
        size_t free,total;checked(cudaMemGetInfo(&free,&total),"conversion reserve check");
        const size_t required=raw_bytes+count*4+output_bytes+sizeof(Statistics)+4;
        if(required>free||free-required<size_t(4096)*1024*1024)throw std::runtime_error("conversion 4GiB headroom");
        Allocation raw(raw_bytes),decoded(count*4),packed(output_bytes),maximum(4),stats(sizeof(Statistics));
        checked(cudaMemcpy(raw.pointer,source,raw_bytes,cudaMemcpyHostToDevice),"original conversion upload");
        checked(cudaMemset(maximum.pointer,0,4),"conversion maximum reset");
        decode_original<<<unsigned((count+255)/256),256>>>(static_cast<const unsigned char *>(raw.pointer),
            static_cast<float *>(decoded.pointer),count,Quant(type),static_cast<unsigned *>(maximum.pointer));
        checked(cudaGetLastError(),"original conversion decode");
        float a;checked(cudaMemcpy(&a,maximum.pointer,4,cudaMemcpyDeviceToHost),"conversion tensor maximum");
        if(!std::isfinite(a))throw std::runtime_error("nonfinite original conversion tensor");
        result->global_scale=a>0?a/(6.f*448.f):1.f;
        checked(cudaMemset(stats.pointer,0,sizeof(Statistics)),"conversion statistics reset");
        encode<<<unsigned((count/2+255)/256),256>>>(static_cast<const float *>(decoded.pointer),
            static_cast<unsigned char *>(packed.pointer),count,result->global_scale,static_cast<Statistics *>(stats.pointer));
        checked(cudaGetLastError(),"FP4 conversion encode");
        checked(cudaMemcpy(destination,packed.pointer,output_bytes,cudaMemcpyDeviceToHost),"converted checkpoint payload");
        Statistics host{};checked(cudaMemcpy(&host,stats.pointer,sizeof(host),cudaMemcpyDeviceToHost),"conversion bounded statistics");
        result->error2=host.error2;result->norm2=host.norm2;result->zero_scales=host.zero_scales;result->elements=count;
        std::memcpy(&result->maximum_error,&host.maximum_error_bits,4);last_error.clear();return 0;
    }catch(const std::exception &error){cudaDeviceSynchronize();last_error=error.what();return 1;}
}
