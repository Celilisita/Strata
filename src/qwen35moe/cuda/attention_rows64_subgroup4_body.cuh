// Fixed Rows64/128/Key64/F32PV. Only softmax ownership/reduction/storage changes.
namespace wm=nvcuda::wmma;
constexpr int Keys=64,Dim=256,Heads=16,CacheStride=512;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
cudaDeviceProp actual_sm75(){int device=0;cudaDeviceProp p{};check(cudaGetDevice(&device));check(cudaGetDeviceProperties(&p,device));
    return p;}
__device__ float maximum(float x){for(int d=2;d;d>>=1)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,d,4));return x;}
__device__ float total(float x){for(int d=2;d;d>>=1)x+=__shfl_xor_sync(0xffffffffu,x,d,4);return x;}
__global__ void zero_tail(half* k,half* v,int width,int cells){const int i=int(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<cells){k[size_t(width)*CacheStride+i]=__float2half_rn(0);v[size_t(width)*CacheStride+i]=__float2half_rn(0);}}
template<int Rows> constexpr size_t shared_bytes(){return Rows*Dim*sizeof(half)+Rows*Keys*sizeof(float)+Rows*Keys*sizeof(half)+Rows*sizeof(float);}
static_assert(shared_bytes<64>()==57600);
template<int Rows,int Threads>
__global__ __launch_bounds__(Threads,1) void query_tile(const float* q,const half* k,const half* v,int first,int count,float* out){
#if __CUDA_ARCH__ == 750
    constexpr int Groups=4,Warps=Threads/32,Own=2;
    static_assert(Rows==64 && Threads==128 && Own==2);
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,head=blockIdx.x,kv=head/8;
    const int column_warp=warp%4,row_group=(warp/4)*Groups,qbase=blockIdx.y*Rows;
    const int active=min(Rows,count-qbase),width=first+qbase+active;
    extern __shared__ __align__(32) unsigned char storage[];
    auto* query=reinterpret_cast<half*>(storage);
    auto* scores=reinterpret_cast<float*>(query+Rows*Dim);
    auto* probability=reinterpret_cast<half*>(scores+Rows*Keys);
    auto* row_scales=reinterpret_cast<float*>(probability+Rows*Keys);
    using Acc=wm::fragment<wm::accumulator,16,16,16,float>;
    Acc running[Groups][4];
#pragma unroll
    for(int g=0;g<Groups;++g){
#pragma unroll
        for(int j=0;j<4;++j)wm::fill_fragment(running[g][j],0.0f);
    }
    float previous_max[Own],previous_sum[Own];
#pragma unroll
    for(int own=0;own<Own;++own){previous_max[own]=-CUDART_INF_F;previous_sum[own]=0;}
    for(int i=tid;i<Rows*Dim;i+=Threads){const int row=i/Dim,d=i%Dim;
        const float value=row<active?q[(size_t(qbase+row)*Heads+head)*Dim+d]:0;
        query[i]=__hmul(__float2half_rn(value*0.0625f),__float2half_rn(0.25f));}
    __syncthreads();
    for(int key=0;key<width;key+=Keys){
        {
            Acc c[Groups];
#pragma unroll
            for(int g=0;g<Groups;++g)wm::fill_fragment(c[g],0.0f);
            // Each K load serves four distinct16-query row fragments in this head.
            for(int d=0;d<Dim;d+=16){
                wm::fragment<wm::matrix_b,16,16,16,half,wm::col_major>b;
                wm::load_matrix_sync(b,k+size_t(key+column_warp*16)*CacheStride+kv*Dim+d,CacheStride);
#pragma unroll
                for(int g=0;g<Groups;++g){wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a;
                    wm::load_matrix_sync(a,query+(row_group+g)*16*Dim+d,Dim);wm::mma_sync(c[g],a,b,c[g]);}
            }
#pragma unroll
            for(int g=0;g<Groups;++g)wm::store_matrix_sync(scores+(row_group+g)*16*Keys+column_warp*16,c[g],Keys,wm::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for(int own=0;own<Own;++own){
            const int group=lane/4,sub=lane%4,row=warp*8+group+own*32,end=first+qbase+row+1;
            const bool valid=row<active;float local[16],max_local=-CUDART_INF_F;
#pragma unroll
            for(int j=0;j<16;++j){const int col=(j*4+sub+group*4)&63;
                local[j]=valid&&key+col<end?scores[row*Keys+col]*4.0f:-CUDART_INF_F;
                max_local=fmaxf(max_local,local[j]);}
            // Every lane participates, including invalid tail groups: full-mask shuffle is safe.
            const float block_max=maximum(max_local),m=valid?fmaxf(previous_max[own],block_max):0;
            float sum_local=0;
#pragma unroll
            for(int j=0;j<16;++j){const int col=(j*4+sub+group*4)&63;
                const float p=valid?expf(local[j]-m):0;sum_local+=p;
                const uint32_t bits=__half_as_ushort(__float2half_rn(p));
                const uint32_t adjacent=__shfl_xor_sync(0xffffffffu,bits,1,4);
                // Pair adjacent half bits:16 writers/warp,16 distinct shared banks at each j.
                if((sub&1)==0)reinterpret_cast<uint32_t*>(probability)[row*32+col/2]=bits|(adjacent<<16);
            }
            const float scale=valid?expf(previous_max[own]-m):0,sum=total(sum_local);
            const float denominator=valid?previous_sum[own]*scale+sum:1;
            previous_max[own]=__shfl_sync(0xffffffffu,m,0,4);
            previous_sum[own]=__shfl_sync(0xffffffffu,denominator,0,4);
            if(sub==0)row_scales[row]=scale;
        }
        __syncthreads(); // Every score reader has finished; row scales are published.
        for(int i=tid;i<Rows*16;i+=Threads)scores[i]=row_scales[i/16];
        __syncthreads();
        { // Rescale fragments die before the PV operands are created.
        Acc scales[Groups];
#pragma unroll
        for(int g=0;g<Groups;++g){wm::load_matrix_sync(scales[g],scores+(row_group+g)*16*16,16,wm::mem_row_major);
#pragma unroll
            for(int j=0;j<4;++j){
#pragma unroll
                for(int i=0;i<running[g][j].num_elements;++i)running[g][j].x[i]*=scales[g].x[i];
            }
        }
        }
        for(int d=0;d<Keys;d+=16){
            wm::fragment<wm::matrix_a,16,16,16,half,wm::row_major>a[Groups];
#pragma unroll
            for(int g=0;g<Groups;++g)wm::load_matrix_sync(a[g],probability+(row_group+g)*16*Keys+d,Keys);
#pragma unroll
            for(int j=0;j<4;++j){wm::fragment<wm::matrix_b,16,16,16,half,wm::row_major>b;
                wm::load_matrix_sync(b,v+size_t(key+d)*CacheStride+kv*Dim+column_warp*64+j*16,CacheStride);
#pragma unroll
                for(int g=0;g<Groups;++g)wm::mma_sync(running[g][j],a[g],b,running[g][j]);
            }
        }
        __syncthreads();
    }
    if((lane%4)==0){
#pragma unroll
        for(int own=0;own<Own;++own)row_scales[warp*8+lane/4+own*32]=previous_sum[own];
    }
    __syncthreads();for(int i=tid;i<Rows*16;i+=Threads)scores[i]=row_scales[i/16];__syncthreads();
#pragma unroll
    for(int g=0;g<Groups;++g){Acc denominator;wm::load_matrix_sync(denominator,scores+(row_group+g)*16*16,16,wm::mem_row_major);
#pragma unroll
        for(int j=0;j<4;++j){
#pragma unroll
            for(int i=0;i<running[g][j].num_elements;++i)running[g][j].x[i]/=denominator.x[i];
        }
    }
    if(active==Rows){
#pragma unroll
        for(int g=0;g<Groups;++g){
#pragma unroll
            for(int j=0;j<4;++j)wm::store_matrix_sync(out+(size_t(qbase+(row_group+g)*16)*Heads+head)*Dim+column_warp*64+j*16,
                running[g][j],Heads*Dim,wm::mem_row_major);
        }
    }else{
        __syncthreads(); // Protect denominator loads before score-tail staging.
#pragma unroll
        for(int j=0;j<4;++j){
#pragma unroll
            for(int g=0;g<Groups;++g)wm::store_matrix_sync(scores+(row_group+g)*16*Keys+column_warp*16,running[g][j],Keys,wm::mem_row_major);
            __syncthreads();
            for(int i=tid;i<active*64;i+=Threads){const int row=i/64,col=i%64,d=(col/16)*64+j*16+col%16;
                out[(size_t(qbase+row)*Heads+head)*Dim+d]=scores[i];}
            __syncthreads();
        }
    }
#endif
}
