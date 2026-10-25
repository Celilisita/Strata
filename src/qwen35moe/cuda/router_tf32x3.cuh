// PTX ISA8.7/CUDA12.8: m16n8k8 TF32 row.col, SM80+; rn.satfinite cvt SM100+.
// https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
struct RouterTf32Tile { float ah[32][68],al[32][68],bh[32][68],bl[32][68]; };
union RouterShared { RouterTf32Tile tile;float partial[4][4]; };
static_assert(sizeof(RouterShared)==34816);
__device__ __forceinline__ unsigned router_tf32(float value) {
    unsigned result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    asm("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(result) : "f"(value));
#else
    result=__float_as_uint(value); // Only the guarded SM120 kernel calls this conversion.
#endif
    return result;
}
__device__ __forceinline__ void router_mma(float (&d)[4],const unsigned (&a)[4],const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#else
    (void)d;(void)a;(void)b;
#endif
}
__global__ __launch_bounds__(256,2)
void project_tf32x3(const float* __restrict__ weights,const float* __restrict__ input,
                   float* __restrict__ output,int tokens) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    __shared__ RouterShared shared;
    const int tid=int(threadIdx.x),lane=tid%32,warp=tid/32,first=int(blockIdx.y)*32;
    // Shared-router scalar keeps the original128-lane FMA/reduction chain.
    if(blockIdx.x==8) {
        for(int offset=0;offset<32;offset+=4) {
            float sums[4]={};
            if(tid<128)for(int k=tid;k<2048;k+=128) {
                const float w=weights[size_t(256)*2048+k];
#pragma unroll
                for(int j=0;j<4;++j)if(first+offset+j<tokens)
                    sums[j]=fmaf(w,input[size_t(first+offset+j)*2048+k],sums[j]);
            }
            if(tid<128) {
#pragma unroll
                for(int j=0;j<4;++j)
#pragma unroll
                    for(int mask=16;mask;mask>>=1)sums[j]+=__shfl_xor_sync(0xffffffffu,sums[j],mask);
                if(lane==0)for(int j=0;j<4;++j)shared.partial[j][warp]=sums[j];
            }
            __syncthreads();
            if(tid<32)for(int j=0;j<4;++j) {
                float sum=tid<4?shared.partial[j][tid]:0.f;
#pragma unroll
                for(int mask=16;mask;mask>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,mask);
                if(tid==0 && first+offset+j<tokens)output[size_t(first+offset+j)*257+256]=sum;
            }
            __syncthreads();
        }
        return; // CTA-uniform, no reader of the tensor tile.
    }
    const int row0=int(blockIdx.x)*32+(warp/4)*16,token0=first+(warp%4)*8;
    const int group=lane/4,thread=lane%4;
    // Keep small cross terms off the much larger main accumulation chain.
    float main[4]={},correction[4]={};
    for(int panel=0;panel<2048;panel+=64) {
        for(int at=tid;at<32*64;at+=256) {
            const int r=at/64,k=at%64;
            const float a=weights[size_t(int(blockIdx.x)*32+r)*2048+panel+k];
            const float b=first+r<tokens?input[size_t(first+r)*2048+panel+k]:0.f;
            const float ah=__uint_as_float(router_tf32(a)),bh=__uint_as_float(router_tf32(b));
            shared.tile.ah[r][k]=ah;shared.tile.al[r][k]=__uint_as_float(router_tf32(__fsub_rn(a,ah)));
            shared.tile.bh[r][k]=bh;shared.tile.bl[r][k]=__uint_as_float(router_tf32(__fsub_rn(b,bh)));
        }
        __syncthreads();
#pragma unroll
        for(int k=0;k<64;k+=8) {
            unsigned ah[4],al[4],bh[2],bl[2];
#pragma unroll
            for(int i=0;i<4;++i) {
                const int r=(warp/4)*16+group+(i%2)*8,c=k+thread+(i/2)*4;
                ah[i]=__float_as_uint(shared.tile.ah[r][c]);al[i]=__float_as_uint(shared.tile.al[r][c]);
            }
#pragma unroll
            for(int i=0;i<2;++i) {
                const int t=(warp%4)*8+group,c=k+thread+i*4;
                bh[i]=__float_as_uint(shared.tile.bh[t][c]);bl[i]=__float_as_uint(shared.tile.bl[t][c]);
            }
            router_mma(main,ah,bh);router_mma(correction,al,bh);router_mma(correction,ah,bl);
        }
        __syncthreads();
    }
#pragma unroll
    for(int i=0;i<4;++i) {
        const int token=token0+thread*2+(i%2),row=row0+group+(i/2)*8;
        if(token<tokens)output[size_t(token)*257+row]=__fadd_rn(main[i],correction[i]);
    }
#else
    (void)weights;(void)input;(void)output;(void)tokens;asm volatile("trap;");
#endif
}
void validate_router_spans(const float* rows,const float* input,float* output,int width,int outputs,int tokens) {
    const std::array<std::pair<uintptr_t,size_t>,3> spans{{{uintptr_t(rows),size_t(outputs)*width*4},
        {uintptr_t(input),size_t(tokens)*width*4},{uintptr_t(output),size_t(tokens)*outputs*4}}};
    for(size_t i=0;i<spans.size();++i) {
        const auto [a,n]=spans[i];
        if(!a || a%4 || a>(std::numeric_limits<uintptr_t>::max)()-n)
            throw std::invalid_argument("router span null/alignment/overflow");
        for(size_t j=0;j<i;++j)if(a<spans[j].first+spans[j].second && spans[j].first<a+n)
            throw std::invalid_argument("router weight/input/output alias");
    }
}
void prepare_router_tf32x3() {
    int device=0;auto e=cudaGetDevice(&device);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    static thread_local int prepared=-1;
    if(prepared==device)return;
    cudaDeviceProp p{};e=cudaGetDeviceProperties(&p,device);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    if(p.major!=12 || p.minor!=0)throw std::invalid_argument("router TF32x3 requires SM120");
    prepared=device;
}
