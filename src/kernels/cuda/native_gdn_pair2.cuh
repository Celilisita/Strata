// Two-token low-rank recurrence. State remains F32 and keeps Strata's layout.
// Pair products are read-only metadata, shared by all 128 state columns.
struct GdnPair2 {
    float g0, g1, b0, b1;
    float q0k0, q1k0, q1k1, k1k0;
};
__device__ __forceinline__ void pair2_reduce(float (&x)[4]) {
#pragma unroll
    for (int offset=16; offset; offset>>=1)
#pragma unroll
        for (int j=0; j<4; ++j)
            x[j]+=__shfl_xor_sync(0xffffffffu,x[j],offset);
}
__global__ __launch_bounds__(128)
void pair2_metadata(const float* q,const float* k,const float* gate,const float* beta,
                    GdnPair2* metadata,int hk,int hv,int tokens,int stride) {
    const int lane=int(threadIdx.x)%32, work=int(blockIdx.x)*4+int(threadIdx.x)/32;
    if (work>=((tokens+1)/2)*hv) return; // Warp-uniform, no CTA barriers.
    const int head=work%hv, t=(work/hv)*2, qh=head%hk;
    const bool second=t+1<tokens;
    float dot[4]={};
#pragma unroll
    for (int r=0; r<4; ++r) {
        const int i=qh*S+r*32+lane;
        const float q0=q[size_t(t)*stride+i], k0=k[size_t(t)*stride+i];
        const float q1=second?q[size_t(t+1)*stride+i]:0;
        const float k1=second?k[size_t(t+1)*stride+i]:0;
        dot[0]+=q0*k0;dot[1]+=q1*k0;dot[2]+=q1*k1;dot[3]+=k1*k0;
    }
    pair2_reduce(dot);
    if (!lane) metadata[work]={expf(gate[size_t(t)*hv+head]),
        second?expf(gate[size_t(t+1)*hv+head]):1.0f,beta[size_t(t)*hv+head],
        second?beta[size_t(t+1)*hv+head]:0.0f,dot[0],dot[1],dot[2],dot[3]};
}
__global__ __launch_bounds__(128,2)
void pair2_scan(float* state,const float* q,const float* k,const float* v,
                const GdnPair2* metadata,float* output,int hk,int hv,int tokens,int stride) {
    constexpr int Columns=8;
    const int head=int(blockIdx.x), lane=int(threadIdx.x);
    const int col=(int(blockIdx.z)*4+int(threadIdx.y))*Columns, qh=head%hk;
    float s[Columns][4];
#pragma unroll
    for (int r=0;r<4;++r)
#pragma unroll
        for (int c=0;c<Columns;++c)
            s[c][r]=state[(size_t(r*32+lane)*hv+head)*S+col+c];
    const float scale=rsqrtf(float(S));
    for (int t=0;t<tokens;t+=2) {
        const bool second=t+1<tokens;
        const auto m=metadata[size_t(t/2)*hv+head];
        float k0[4],k1[4],q0[4],q1[4];
#pragma unroll
        for (int r=0;r<4;++r) {
            const int i=qh*S+r*32+lane;
            k0[r]=k[size_t(t)*stride+i];q0[r]=q[size_t(t)*stride+i];
            k1[r]=second?k[size_t(t+1)*stride+i]:0;
            q1[r]=second?q[size_t(t+1)*stride+i]:0;
        }
#pragma unroll
        for (int c=0;c<Columns;++c) {
            float dot[4]={}; // Four independent chains hide shuffle latency.
#pragma unroll
            for (int r=0;r<4;++r) {
                dot[0]+=s[c][r]*k0[r];dot[1]+=s[c][r]*k1[r];
                dot[2]+=s[c][r]*q0[r];dot[3]+=s[c][r]*q1[r];
            }
            pair2_reduce(dot);
            const float d0=(v[size_t(t)*stride+head*S+col+c]-m.g0*dot[0])*m.b0;
            const float d1=second?(v[size_t(t+1)*stride+head*S+col+c]-
                m.g1*(m.g0*dot[1]+m.k1k0*d0))*m.b1:0;
            if (!lane) {
                output[(size_t(t)*hv+head)*S+col+c]=(m.g0*dot[2]+m.q0k0*d0)*scale;
                if(second) output[(size_t(t+1)*hv+head)*S+col+c]=
                    (m.g0*m.g1*dot[3]+m.g1*m.q1k0*d0+m.q1k1*d1)*scale;
            }
#pragma unroll
            for (int r=0;r<4;++r)
                s[c][r]=m.g0*m.g1*s[c][r]+m.g1*k0[r]*d0+k1[r]*d1;
        }
    }
#pragma unroll
    for (int r=0;r<4;++r)
#pragma unroll
        for (int c=0;c<Columns;++c)
            state[(size_t(r*32+lane)*hv+head)*S+col+c]=s[c][r];
}
