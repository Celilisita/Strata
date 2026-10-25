// Four-token low-rank recurrence. No inverse decay, matrix inverse or precision change.
// Metadata keeps six K and ten causal Q/K intersections, including their relative decays.
struct GdnChunk4 {float g[4],beta[4],kk[6],qk[10];};
static_assert(sizeof(GdnChunk4)==96);
template<int N> __device__ __forceinline__ void chunk4_reduce(float (&x)[N]) {
#pragma unroll
    for(int offset=16;offset;offset>>=1)
#pragma unroll
        for(int i=0;i<N;++i)x[i]+=__shfl_xor_sync(0xffffffffu,x[i],offset);
}
__global__ __launch_bounds__(128)
void chunk4_metadata(const float* q,const float* k,const float* gate,const float* beta,
    GdnChunk4* metadata,int hk,int hv,int tokens,int stride) {
    const int lane=int(threadIdx.x)%32,work=int(blockIdx.x)*4+int(threadIdx.x)/32;
    if(work>=((tokens+3)/4)*hv)return;
    const int head=work%hv,t=(work/hv)*4,qh=head%hk,active=min(4,tokens-t);
    float qr[4][4]={},kr[4][4]={},dot[16]={};
#pragma unroll
    for(int i=0;i<4;++i)if(i<active)
#pragma unroll
        for(int r=0;r<4;++r){const int at=qh*S+r*32+lane;qr[i][r]=q[size_t(t+i)*stride+at];kr[i][r]=k[size_t(t+i)*stride+at];}
#pragma unroll
    for(int i=0;i<4;++i)
#pragma unroll
        for(int j=0;j<4;++j) {
            if(j<i) {
#pragma unroll
                for(int r=0;r<4;++r)dot[i*(i-1)/2+j]+=kr[i][r]*kr[j][r];
            }
            if(j<=i) {
#pragma unroll
                for(int r=0;r<4;++r)dot[6+i*(i+1)/2+j]+=qr[i][r]*kr[j][r];
            }
        }
    chunk4_reduce(dot);
    if(!lane) {
        auto& m=metadata[work];
#pragma unroll
        for(int i=0;i<4;++i){m.g[i]=i<active?expf(gate[size_t(t+i)*hv+head]):1.f;m.beta[i]=i<active?beta[size_t(t+i)*hv+head]:0.f;}
#pragma unroll
        for(int i=0;i<4;++i) {
            float decay=1.f;
#pragma unroll
            for(int j=i;j>=0;--j) {
                m.qk[i*(i+1)/2+j]=decay*dot[6+i*(i+1)/2+j];
                if(j<i)m.kk[i*(i-1)/2+j]=decay*dot[i*(i-1)/2+j];
                decay*=m.g[j];
            }
        }
    }
}
__global__ __launch_bounds__(128,2)
void chunk4_scan(float* state,const float* q,const float* k,const float* v,
    const GdnChunk4* metadata,float* output,int hk,int hv,int tokens,int stride) {
    constexpr int Columns=4;
    const int head=int(blockIdx.x),lane=int(threadIdx.x),qh=head%hk;
    const int col=(int(blockIdx.z)*4+int(threadIdx.y))*Columns;
    float s[Columns][4];
#pragma unroll
    for(int r=0;r<4;++r)
#pragma unroll
        for(int c=0;c<Columns;++c)s[c][r]=state[(size_t(r*32+lane)*hv+head)*S+col+c];
    const float scale=rsqrtf(float(S));
    for(int t=0;t<tokens;t+=4) {
        const int active=min(4,tokens-t);const auto m=metadata[size_t(t/4)*hv+head];
        float kr[4][4]={},qr[4][4]={},prefix[4],tail[4];
        float product=1.f;
#pragma unroll
        for(int i=0;i<4;++i){product*=m.g[i];prefix[i]=product;
            if(i<active)
#pragma unroll
                for(int r=0;r<4;++r){const int at=qh*S+r*32+lane;kr[i][r]=k[size_t(t+i)*stride+at];qr[i][r]=q[size_t(t+i)*stride+at];}
        }
        product=1.f;
#pragma unroll
        for(int i=3;i>=0;--i){tail[i]=product;product*=m.g[i];}
#pragma unroll
        for(int c=0;c<Columns;++c) {
            float dot[8]={},delta[4]={};
#pragma unroll
            for(int i=0;i<4;++i)
#pragma unroll
                for(int r=0;r<4;++r){dot[i]+=s[c][r]*kr[i][r];dot[4+i]+=s[c][r]*qr[i][r];}
            chunk4_reduce(dot);
#pragma unroll
            for(int i=0;i<4;++i)if(i<active) {
                float predicted=prefix[i]*dot[i];
#pragma unroll
                for(int j=0;j<4;++j)if(j<i) {
                    predicted+=m.kk[i*(i-1)/2+j]*delta[j];
                }
                delta[i]=(v[size_t(t+i)*stride+head*S+col+c]-predicted)*m.beta[i];
                float value=prefix[i]*dot[4+i];
#pragma unroll
                for(int j=0;j<4;++j)if(j<=i) {
                    value+=m.qk[i*(i+1)/2+j]*delta[j];
                }
                if(!lane)output[(size_t(t+i)*hv+head)*S+col+c]=value*scale;
            }
#pragma unroll
            for(int j=0;j<4;++j)delta[j]*=tail[j];
#pragma unroll
            for(int r=0;r<4;++r) {
                float updated=prefix[3]*s[c][r];
#pragma unroll
                for(int j=0;j<4;++j) {
                    updated+=kr[j][r]*delta[j];
                }
                s[c][r]=updated;
            }
        }
    }
#pragma unroll
    for(int r=0;r<4;++r)
#pragma unroll
        for(int c=0;c<Columns;++c)state[(size_t(r*32+lane)*hv+head)*S+col+c]=s[c][r];
}
