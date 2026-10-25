#pragma once
// Strata174 merge<512> mathematics, kept private to234. The frozen174
// anonymous-namespace kernel cannot be called across CUs. No old segment runs.
namespace strata::gemma4::native::detail234 {
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void merge(const float* partial,float* output,int parts){
    __shared__ float weights[2048];const int lane=threadIdx.x,head=blockIdx.x,t=blockIdx.y;
    const float* p=partial+(size_t(t)*32+head)*parts*514;
    float top=-CUDART_INF_F;bool bad=false;
    for(int i=lane;i<parts;i+=32){const float z=p[i*514+1],m=p[i*514];
        bad|=!isfinite(z)||(z>0&&!isfinite(m));if(z>0)top=fmaxf(top,m);}
    const unsigned invalid=__ballot_sync(0xffffffff,bad);top=maximum(top);float denominator=0;
    for(int i=lane;i<parts;i+=32){const float z=p[i*514+1];const float w=z==0?0:expf(p[i*514]-top);
        weights[i]=w;denominator=fmaf(z,w,denominator);}
    denominator=sum(denominator);__syncwarp();
    #pragma unroll
    for(int j=0;j<16;++j){float value=0;
        for(int i=0;i<parts;++i)if(weights[i]!=0)value=fmaf(weights[i],p[i*514+2+lane+32*j],value);
        output[(size_t(t)*32+head)*512+lane+32*j]=invalid||!(denominator>0)?CUDART_NAN_F:value/denominator;}
}
}
