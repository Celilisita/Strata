// Included in ops.cu's private kernel namespace. No architecture-specific MMA.
__device__ inline unsigned pack32_code(float value,float reciprocal){
    const unsigned bits=__half_as_ushort(__float2half_rn(roundf(__fmul_rn(value,reciprocal))));
    return bits&0x7fffu?bits:0u;
}
__device__ inline float pack32_value(float x,float w,float scale,float gate,bool gated){
    float value=__fmul_rn(__fmul_rn(x,scale),w);
    if(gated)value=__fmul_rn(value,swiglu_value(gate,1.0f));
    return value;
}
__device__ inline void pack32_joint(float4 v,detail::HalfAffineInputBlock *q4,
                                  detail::HalfInputBlock *q6,int group,int token,int lane){
    float maximum=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    float sum=__fadd_rn(__fadd_rn(__fadd_rn(v.x,v.y),v.z),v.w);
    for(int offset=4;offset;offset/=2){
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,offset));
    }
    const float reciprocal=maximum?__fdividef(127.0f,maximum):0;
    const float delta=maximum?__fdividef(1.0f,reciprocal):0;
    const unsigned x=pack32_code(v.x,reciprocal),y=pack32_code(v.y,reciprocal);
    const unsigned z=pack32_code(v.z,reciprocal),w=pack32_code(v.w,reciprocal);
    const uint2 codes=make_uint2(x|(y<<16),z|(w<<16));
    const int word=group%8*16+lane*2,block=group/8;
    *reinterpret_cast<uint2 *>(q4[block].codes[token]+word)=codes;
    if(q6)*reinterpret_cast<uint2 *>(q6[block].codes[token]+word)=codes;
    if(!lane){
        q4[block].ds[group%8][token]=unsigned(__half_as_ushort(__float2half_rn(delta)))|
            (unsigned(__half_as_ushort(__float2half_rn(sum)))<<16);
        if(q6)q6[block].delta[group%8][token]=delta;
    }
}
__global__ __launch_bounds__(1024) void rms_pack32(const float *x,const float *weight,float *out,const float *gate,
                          detail::HalfAffineInputBlock *q4,detail::HalfInputBlock *q6,
                          int features,int heads,float epsilon){
    __shared__ float partial[32];
    const int head=int(blockIdx.x)%heads,token=int(blockIdx.x)/heads;
    const size_t base=size_t(blockIdx.x)*features;
    float square=0;
    for(int i=threadIdx.x;i<features;i+=blockDim.x){
        const float value=x[base+i];square=__fmaf_rn(value,value,square);
    }
    square=sum_warp(square);
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=square;
    __syncthreads();
    square=sum_warp(threadIdx.x%32<blockDim.x/32?partial[threadIdx.x%32]:0.0f);
    const float scale=rsqrtf(__fmaf_rn(square,__fdividef(1.0f,float(features)),epsilon));
    // features/32 is a multiple of four: every active encoding warp is full.
    for(int group=threadIdx.x/8;group<features/32;group+=blockDim.x/8){
        const int index=group*32+(threadIdx.x%8)*4;
        const float4 a=reinterpret_cast<const float4 *>(x+base)[index/4];
        const float4 b=reinterpret_cast<const float4 *>(weight)[index/4];
        const float4 g=gate?reinterpret_cast<const float4 *>(gate+base)[index/4]:make_float4(0,0,0,0);
        const float4 value=make_float4(pack32_value(a.x,b.x,scale,g.x,gate),
            pack32_value(a.y,b.y,scale,g.y,gate),pack32_value(a.z,b.z,scale,g.z,gate),
            pack32_value(a.w,b.w,scale,g.w,gate));
        reinterpret_cast<float4 *>(out+base)[index/4]=value;
        pack32_joint(value,q4,q6,head*(features/32)+group,token,threadIdx.x%8);
    }
}
