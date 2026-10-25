// Own GQA schedule: K[16,D] x Q[D,8], then V^T[16,K] x P[K,8].
// Every raw MMA row/column is useful; one warp owns one of eight Q heads.
constexpr int GqaAsyncSharedBytes = 2 * 64 * 256 * int(sizeof(half));
__device__ __forceinline__ int gqa_stage_half(int row, int column) {
    return ((row * 512 + column * 2) ^ ((row & 7) << 4)) / 2;
}
__device__ __forceinline__ unsigned gqa_pair(half a, half b) {
    return unsigned(__half_as_ushort(a)) | (unsigned(__half_as_ushort(b)) << 16);
}
__device__ __forceinline__ unsigned gqa_shared_word(const half* tile, int row, int column) {
    return *reinterpret_cast<const unsigned*>(tile + gqa_stage_half(row,column));
}
__device__ __forceinline__ void gqa_async_tile(half* tile, const half* source,
                                             int key, int width, int kv, int tid) {
#if __CUDA_ARCH__ >= 800
#pragma unroll
    for (int chunk=tid; chunk<64*32; chunk+=256) {
        const int row=chunk/32, column=(chunk%32)*8;
        auto* dst=tile+gqa_stage_half(row,column);
        if (key+row<width) {
            const auto* src=source+size_t(key+row)*CacheStride+kv*Dim+column;
            const unsigned address=unsigned(__cvta_generic_to_shared(dst));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(address),"l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst)=make_uint4(0,0,0,0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
__device__ __forceinline__ void gqa_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}
__device__ __forceinline__ void gqa_qk(float* c,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#else
    (void)c;(void)a;(void)b;
#endif
}
__device__ __forceinline__ void gqa_pv(unsigned* c,const unsigned* a,const unsigned* b) {
#if __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(c[0]),"+r"(c[1])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
#else
    (void)c;(void)a;(void)b;
#endif
}
__global__ __launch_bounds__(256,1)
void prefill_gqa_async(const float* q,const half* k,const half* v,int first,int count,float* out) {
#if __CUDA_ARCH__ >= 1200
    extern __shared__ __align__(32) unsigned char storage[];
    auto* kt=reinterpret_cast<half*>(storage); auto* vt=kt+64*Dim;
    const int tid=int(threadIdx.x),warp=tid/32,lane=tid%32,kv=int(blockIdx.x);
    const int head=kv*8+warp,qbase=int(blockIdx.y)*8,active=min(8,count-qbase);
    const int width=first+qbase+active,row=lane/4,column=2*(lane%4);
    unsigned query[16][2],accumulated[16][2]={};
#pragma unroll
    for(int d=0;d<16;++d) {
#pragma unroll
        for(int piece=0;piece<2;++piece) {
            const int feature=d*16+2*(lane%4)+piece*8;
            half a=__float2half_rn(0),b=a;
            if(row<active) {
                const size_t cell=(size_t(qbase+row)*Heads+head)*Dim+feature;
                a=__hmul(__float2half_rn(q[cell]*0.0625f),__float2half_rn(0.25f));
                b=__hmul(__float2half_rn(q[cell+1]*0.0625f),__float2half_rn(0.25f));
            }
            query[d][piece]=gqa_pair(a,b);
        }
    }
    float old_max[2]={-CUDART_INF_F,-CUDART_INF_F},old_sum[2]={},old_scale[2]={};
    gqa_async_tile(kt,k,0,width,kv,tid);gqa_async_wait();__syncthreads();
    for(int key=0;key<width;key+=64) {
        // V_current runs while QK consumes K_current; Q is already in registers.
        gqa_async_tile(vt,v,key,width,kv,tid);
        float score[4][4]={};
#pragma unroll
        for(int d=0;d<16;++d) {
#pragma unroll
            for(int tile=0;tile<4;++tile) {
                const int r=tile*16+row,c=d*16+2*(lane%4);
                const unsigned a[4]={gqa_shared_word(kt,r,c),gqa_shared_word(kt,r+8,c),
                    gqa_shared_word(kt,r,c+8),gqa_shared_word(kt,r+8,c+8)};
                gqa_qk(score[tile],a,query[d]);
            }
        }
        gqa_async_wait();__syncthreads(); // All K reads finished; V is ready.
        auto* probability=kt+warp*8*64; // Dead K, logical column-major P[key,query].
        half factor[2];
#pragma unroll
        for(int j=0;j<2;++j) {
            const bool valid=column+j<active;const int end=first+qbase+column+j+1;
            float local=-CUDART_INF_F;
#pragma unroll
            for(int tile=0;tile<4;++tile)
#pragma unroll
                for(int h=0;h<2;++h) {
                    float& s=score[tile][h*2+j];
                    s=valid&&key+tile*16+row+h*8<end?s*4.0f:-CUDART_INF_F;
                    local=fmaxf(local,s);
                }
            for(int offset=4;offset<32;offset*=2) local=fmaxf(local,__shfl_xor_sync(0xffffffffu,local,offset));
            const float maximum=valid?fmaxf(old_max[j],local):0;
            const float rescale=valid?expf(old_max[j]-maximum):0;
            float sum=0;
#pragma unroll
            for(int tile=0;tile<4;++tile)
#pragma unroll
                for(int h=0;h<2;++h) {
                    float& s=score[tile][h*2+j];s=valid?expf(s-maximum):0;sum+=s;
                }
            for(int offset=4;offset<32;offset*=2) sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
            const float denominator=valid?old_sum[j]*rescale+sum:1;
            const bool finite=!valid||(isfinite(maximum)&&isfinite(denominator)&&denominator>0&&
                (old_sum[j]==0||(old_scale[j]>0&&isfinite(old_scale[j]))));
            const float scale=valid&&finite?halfpv_numerator_scale(denominator):0;
            factor[j]=__float2half_rn(valid&&finite&&old_sum[j]>0?rescale*(scale/old_scale[j]):0);
#pragma unroll
            for(int tile=0;tile<4;++tile)
#pragma unroll
                for(int h=0;h<2;++h) probability[(column+j)*64+tile*16+row+h*8]=
                    __float2half_rn(finite?score[tile][h*2+j]*scale:0);
            old_max[j]=maximum;old_sum[j]=finite?denominator:-1;old_scale[j]=scale;
        }
        __syncthreads();
        unsigned p[4][2];
#pragma unroll
        for(int tile=0;tile<4;++tile) {
            const auto* source=probability+row*64+tile*16+2*(lane%4);
            p[tile][0]=*reinterpret_cast<const unsigned*>(source);
            p[tile][1]=*reinterpret_cast<const unsigned*>(source+8);
        }
        __syncthreads(); // All P fragments loaded before K_next overwrites K/P.
        if(key+64<width) gqa_async_tile(kt,k,key+64,width,kv,tid);
        const unsigned packed_factor=gqa_pair(factor[0],factor[1]);
#pragma unroll
        for(int feature=0;feature<16;++feature) {
#pragma unroll
            for(int h=0;h<2;++h) {
                union {unsigned u;half2 h2;} value,multiplier;
                value.u=accumulated[feature][h];multiplier.u=packed_factor;
                value.h2=__hmul2(value.h2,multiplier.h2);accumulated[feature][h]=value.u;
            }
#pragma unroll
            for(int tile=0;tile<4;++tile) {
                unsigned a[4];
#pragma unroll
                for(int word=0;word<4;++word) {
                    const int d=feature*16+row+(word%2)*8;
                    const int r=tile*16+2*(lane%4)+(word/2)*8;
                    a[word]=gqa_pair(vt[gqa_stage_half(r,d)],vt[gqa_stage_half(r+1,d)]);
                }
                gqa_pv(accumulated[feature],a,p[tile]);
            }
        }
        gqa_async_wait();__syncthreads(); // K_next ready; all V_current readers done.
    }
#pragma unroll
    for(int feature=0;feature<16;++feature)
#pragma unroll
        for(int h=0;h<2;++h)
#pragma unroll
            for(int j=0;j<2;++j) if(column+j<active) {
                const auto bits=uint16_t(accumulated[feature][h]>>(16*j));
                out[(size_t(qbase+column+j)*Heads+head)*Dim+feature*16+row+h*8]=
                    old_sum[j]>0?__half2float(__ushort_as_half(bits))/(old_sum[j]*old_scale[j]):CUDART_NAN_F;
            }
#endif
}
