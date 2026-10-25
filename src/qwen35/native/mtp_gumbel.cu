#include "strata/qwen35/native/mtp_gumbel.hpp"
#include "mtp_gumbel_rng.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <climits>
#include <cmath>
#include <math_constants.h>
#include <cstdlib>
namespace strata::qwen35::native {namespace {
struct Candidate {double score;int id;unsigned bad;};
__device__ Candidate merge(Candidate a,Candidate b){
    const unsigned bad=a.bad|b.bad;
    if(b.score>a.score||(b.score==a.score&&b.id<a.id))a=b;
    a.bad=bad;return a;
}
__device__ Candidate warp_max(Candidate x){
    for(int d=16;d;d/=2){Candidate other{__shfl_down_sync(0xffffffff,x.score,d),
        __shfl_down_sync(0xffffffff,x.id,d),__shfl_down_sync(0xffffffff,x.bad,d)};x=merge(x,other);}
    return x;
}
__global__ void pick(const int *ids,const double *prefix,const int *kept,
    const strata::kernels::SamplePick *source,int vocabulary,int limit,uint64_t seed,
    const uint64_t *counter,unsigned offset,int *out,unsigned *invalid){
    __shared__ double scores[4];__shared__ int labels[4];__shared__ unsigned bads[4];
    const int rank=threadIdx.x,lane=rank%32,warp=rank/32,count=*kept;
    Candidate best{-CUDART_INF,INT_MAX,unsigned(source->first_bad!=-1||count<1||count>limit||count>vocabulary)};
    if(rank<count&&!best.bad){
        const int token=ids[rank];const double delta=prefix[rank]-(rank?prefix[rank-1]:0.);
        best.bad=unsigned(token<0||token>=vocabulary||!isfinite(delta));
        if(!best.bad&&delta>0){
            const double u=mtp_gumbel_detail::uniform(seed,*counter+uint64_t(offset)+blockIdx.x,token);
            best.score=log(delta)-log(-log(u));best.id=token;
        }
    }
    best=warp_max(best);
    if(!lane){scores[warp]=best.score;labels[warp]=best.id;bads[warp]=best.bad;}
    __syncthreads();
    if(!warp){
        best=lane<4?Candidate{scores[lane],labels[lane],bads[lane]}:Candidate{-CUDART_INF,INT_MAX,0};
        best=warp_max(best);
        if(!lane){const bool bad=best.bad||best.id<0||best.id>=vocabulary||!isfinite(best.score);
            out[blockIdx.x]=bad?0:best.id;invalid[blockIdx.x]=unsigned(bad);}
    }
}
}
bool mtp_gumbel_supported(){
    const char *flag=std::getenv("STRATA_DENSE_MTP_GUMBEL");if(flag&&flag[0]=='0')return false;
    int device=0;check_cuda(cudaGetDevice(&device),"MTP coupling device");const auto caps=capabilities(device);
    return caps.major==12&&caps.minor==0;
}
void mtp_gumbel_pick(const strata::kernels::RadixSampler &sampler,int vocabulary,
    const strata::kernels::SamplerParams &p,const uint64_t *counter,unsigned offset,int *out,unsigned *bad,cudaStream_t stream,int draws){
    if(vocabulary<1||p.top_k<1||p.top_k>64||!counter||offset>3||!out||!bad||draws<1||draws>65536)
        throw std::invalid_argument("MTP Gumbel sampler geometry/counter");
    pick<<<draws,128,0,stream>>>(sampler.sorted_ids(),sampler.cumulative(),sampler.kept_count(),
        sampler.result(),vocabulary,p.top_k,p.seed,counter,offset,out,bad);
    check_cuda(cudaGetLastError(),"MTP token-ID coupled pick");
}
}
