#include "strata/qwen35/native/ops.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include "small_prefill_half_input.cuh"
#include "small_prefill_half_raw.hpp"
#include <cuda_fp16.h>
#include <cmath>

namespace strata::qwen35::native {
namespace {
__device__ float sum_warp(float value) {
    for (int d = 16; d; d /= 2) value = __fadd_rn(value, __shfl_xor_sync(0xffffffff, value, d));
    return value;
}
#include "ops_fused.cuh"
#include "ops_pack32.cuh"

__global__ void convolution(const float *x, const float *history, const float *w, float *raw,
                            float *activated, float *next, int channels, int tokens) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size_t(channels) * tokens) return;
    const int c = int(index % channels), t = int(index / channels);
    float value = 0;
    for (int j = 0; j < 4; ++j) {
        const int source = t + j - 3;
        const float v = source < 0 ? history[size_t(c) * 3 + source + 3] : x[size_t(source) * channels + c];
        value = __fmaf_rn(v, w[size_t(c) * 4 + j], value);
    }
    raw[index] = value; activated[index] = swiglu_value(value, 1.0f);
    if (t == tokens - 1) {
        for (int j = 0; j < 3; ++j) {
            const int source = tokens - 3 + j;
            next[size_t(c) * 3 + j] = source < 0 ? history[size_t(c) * 3 + source + 3] : x[size_t(source) * channels + c];
        }
    }
}
__global__ void normalized_keys(const float *mixed, float *q, float *k, int dim, int heads,
                               int channels, int tokens, float epsilon) {
    const int row = int(blockIdx.x) * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= heads * tokens) return;
    const int head = row % heads, token = row / heads;
    const size_t base = size_t(token) * channels + head * dim;
    float qs = 0, ks = 0;
    for (int j = lane; j < dim; j += 32) {
        const float a = mixed[base + j], b = mixed[base + dim * heads + j];
        qs = __fmaf_rn(a, a, qs); ks = __fmaf_rn(b, b, ks);
    }
    qs = rsqrtf(fmaxf(sum_warp(qs), epsilon)); ks = rsqrtf(fmaxf(sum_warp(ks), epsilon));
    for (int j = lane; j < dim; j += 32) {
        q[size_t(row) * dim + j] = __fmul_rn(mixed[base + j], qs);
        k[size_t(row) * dim + j] = __fmul_rn(mixed[base + dim * heads + j], ks);
    }
}
__global__ void split_values(const float *mixed, float *v, int columns, int channels, int offset, size_t count) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) v[index] = mixed[(index / columns) * channels + offset + index % columns];
}
__global__ void parameters(const float *alpha, const float *br, const float *bias, const float *a,
                           float *g, float *beta, int heads, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float x = __fadd_rn(alpha[i], bias[i % heads]);
    const float softplus = x > 20 ? x : __logf(__fadd_rn(1.0f, __expf(x)));
    g[i] = __fmul_rn(softplus, a[i % heads]);
    beta[i] = __fdividef(1.0f, __fadd_rn(1.0f, __expf(-br[i])));
}

}
void rms_norm_quantized32(const float *x,const float *w,float *out,const float *gate,
                         int features,int heads,float eps,void *q4,void *q6,cudaStream_t stream){
    if(!x||!w||!out||!q4||features<128||features%128||features>32768||heads<1||heads>256||
       size_t(features)*heads>32768||features*heads%256||!(eps>0)||!std::isfinite(eps))
        throw std::invalid_argument("invalid fused RMS PP32 geometry");
    const size_t values=size_t(features)*heads*32*4;
    const size_t encoded=size_t(features)*heads/256*sizeof(detail::HalfInputBlock);
    auto overlap=[](const void *a,size_t na,const void *b,size_t nb){
        const uintptr_t aa=reinterpret_cast<uintptr_t>(a),bb=reinterpret_cast<uintptr_t>(b);
        return aa<=bb?bb-aa<na:aa-bb<nb;
    };
    if(reinterpret_cast<uintptr_t>(x)%16||reinterpret_cast<uintptr_t>(w)%16||
       reinterpret_cast<uintptr_t>(out)%16||reinterpret_cast<uintptr_t>(q4)%16||
       (gate&&reinterpret_cast<uintptr_t>(gate)%16)||(q6&&reinterpret_cast<uintptr_t>(q6)%16)||
       overlap(out,values,x,values)||overlap(q4,encoded,x,values)||overlap(q4,encoded,out,values)||
       (gate&&(overlap(q4,encoded,gate,values)||overlap(out,values,gate,values)))||
       (q6&&(overlap(q6,encoded,x,values)||overlap(q6,encoded,out,values)||
             overlap(q6,encoded,q4,encoded)||(gate&&overlap(q6,encoded,gate,values)))))
        throw std::invalid_argument("fused RMS requires distinct aligned owned storage");
    rms_pack32<<<heads*32,features>=1024?1024:256,0,stream>>>(x,w,out,gate,
        static_cast<detail::HalfAffineInputBlock *>(q4),static_cast<detail::HalfInputBlock *>(q6),features,heads,eps);
    check_cuda(cudaGetLastError(),"owned fused RMS joint Q4/Q6 preparation");
}

void causal_conv4(const float *x, const float *history, const float *w, float *raw, float *activated,
                  float *next, int channels, int tokens, cudaStream_t stream) {
    if (!x || !history || !w || !raw || !activated || !next || history == next || channels < 1 || tokens < 1 || tokens > 8192)
        throw std::invalid_argument("invalid own causal convolution");
    const size_t count = size_t(channels) * tokens;
    convolution<<<unsigned((count + 255) / 256), 256, 0, stream>>>(x, history, w, raw, activated, next, channels, tokens);
    check_cuda(cudaGetLastError(), "native causal convolution");
}
void gdn_split(const float *mixed, float *q, float *k, float *v, int dim, int hq, int hv, int tokens,
               float eps, cudaStream_t stream) {
    if (!mixed || !q || !k || !v || (dim != 64 && dim != 128) || hq < 1 || hv < hq || hv > 256 || hv % hq ||
        tokens < 1 || tokens > 8192 || !(eps > 0) || !std::isfinite(eps))
        throw std::invalid_argument("invalid own normalized GDN split");
    const int channels = (2 * hq + hv) * dim;
    normalized_keys<<<(hq * tokens + 3) / 4, 128, 0, stream>>>(mixed, q, k, dim, hq, channels, tokens, eps);
    check_cuda(cudaGetLastError(), "native normalized GDN keys");
    const size_t count = size_t(hv) * dim * tokens;
    split_values<<<unsigned((count + 255) / 256), 256, 0, stream>>>(mixed, v, hv * dim, channels, 2 * hq * dim, count);
    check_cuda(cudaGetLastError(), "native GDN value split");
}
void gdn_parameters(const float *alpha, const float *br, const float *bias, const float *a,
                    float *g, float *beta, int heads, int tokens, cudaStream_t stream) {
    if (!alpha || !br || !bias || !a || !g || !beta || heads < 1 || tokens < 1)
        throw std::invalid_argument("invalid own GDN parameters");
    const size_t count = size_t(heads) * tokens;
    parameters<<<unsigned((count + 255) / 256), 256, 0, stream>>>(alpha, br, bias, a, g, beta, heads, count);
    check_cuda(cudaGetLastError(), "native GDN decay/beta");
}

void add_rms_norm(const float *a,const float *b,const float *w,float *residual_out,float *out,
                  int features,int rows,float eps,cudaStream_t stream) {
    if(!a||!b||!w||!residual_out||!out||a==residual_out||b==residual_out||residual_out==out||
       features<32||features>32768||rows<1||rows>2097152||!(eps>0)||!std::isfinite(eps))
        throw std::invalid_argument("invalid own fused residual normalization");
    residual_rms<<<rows,features>=1024?1024:256,0,stream>>>(a,b,w,residual_out,out,features,eps);
    check_cuda(cudaGetLastError(),"native fused residual normalization");
}
void gdn_prepare(const float *mixed,const float *alpha,const float *br,const float *bias,const float *a,
                 float *q,float *k,float *v,float *g,float *beta,int dim,int hq,int hv,int tokens,
                 float eps,cudaStream_t stream) {
    if(!mixed||!alpha||!br||!bias||!a||!q||!k||!v||!g||!beta||(dim!=64&&dim!=128)||
       hq<1||hv<hq||hv>256||hv%hq||tokens<1||tokens>8192||!(eps>0)||!std::isfinite(eps))
        throw std::invalid_argument("invalid own fused GDN preparation");
    prepared_gdn<<<(hq*tokens+3)/4,128,0,stream>>>(mixed,alpha,br,bias,a,q,k,v,g,beta,dim,hq,hv,tokens,eps);
    check_cuda(cudaGetLastError(),"native fused GDN preparation");
}

}
