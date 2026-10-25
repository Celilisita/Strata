#include "strata/qwen35/native/mtp_prob.hpp"
#include "mtp_gumbel_rng.cuh"
#include "strata/qwen35/native/ffn.hpp"
#include <cmath>
#include <climits>
namespace strata::qwen35::native {namespace {
__device__ bool cursor_ok(MtpProbCursor c,unsigned offset){return c.counter<=UINT64_MAX-uint64_t(offset)-uint64_t(blockIdx.x);}
__device__ double random(uint64_t seed,uint64_t counter,int domain){return mtp_gumbel_detail::uniform(seed,counter,domain);}
__device__ bool distribution_bad(const strata::kernels::SamplePick *pick,int count,int limit,int vocabulary,double total){
    return pick->first_bad!=-1||count<1||count>limit||count>vocabulary||!(total>0)||!isfinite(total);
}
__global__ void draft(const int *ids,const double *cdf,const int *kept,const strata::kernels::SamplePick *source,
    int vocabulary,int limit,uint64_t seed,const MtpProbCursor *cursor,unsigned offset,MtpProbRow *rows,int *out,unsigned *invalid){
    const int tid=threadIdx.x,count=*kept;const auto c=*cursor;auto &row=rows[blockIdx.x];
    const double total=count>=1&&count<=limit&&count<=vocabulary?cdf[count-1]:0.;
    __shared__ unsigned bad;
    if(!tid){bad=unsigned(distribution_bad(source,count,limit,vocabulary,total)||!cursor_ok(c,offset));
        row.counter=c.counter+offset+uint64_t(blockIdx.x);row.epoch=c.epoch;row.seed=seed;row.vocabulary=vocabulary;row.kept=count;row.id=0;}
    __syncthreads();
    if(tid<count&&count<=limit&&count<=vocabulary){
        const double delta=cdf[tid]-(tid?cdf[tid-1]:0.);const int id=ids[tid];
        row.ids[tid]=id;row.mass[tid]=delta/total;
        if(id<0||id>=vocabulary||delta<0||!isfinite(delta)||!isfinite(row.mass[tid]))atomicOr(&bad,1u);
    }
    __syncthreads();
    if(!tid){
        if(!bad){const double u=random(seed,row.counter,0)*total;int at=0;
            while(at+1<count&&cdf[at]<u)++at;
            row.id=ids[at];if(!(row.mass[at]>0))bad=1;}
        row.bad=bad;out[blockIdx.x]=bad?0:row.id;invalid[blockIdx.x]=bad;
    }
}
template<bool HasDraft>__global__ void target(const int *ids,const double *cdf,const int *kept,
    const strata::kernels::SamplePick *source,int vocabulary,int limit,uint64_t seed,const MtpProbCursor *cursor,
    unsigned offset,const MtpProbRow *rows,const int *expected,int *out,unsigned *invalid){
    const int tid=threadIdx.x,count=*kept;const auto c=*cursor;
    const uint64_t position=c.counter+offset+uint64_t(blockIdx.x);
    const double total=count>=1&&count<=limit&&count<=vocabulary?cdf[count-1]:0.;
    __shared__ unsigned bad;__shared__ double residual[64],p_at_q,q_at_q;
    if(!tid){bad=unsigned(distribution_bad(source,count,limit,vocabulary,total)||!cursor_ok(c,offset));p_at_q=0;q_at_q=0;
        if constexpr(HasDraft){const auto &q=rows[blockIdx.x];
            if(q.bad||q.counter!=position||q.epoch!=c.epoch||q.seed!=seed||q.vocabulary!=vocabulary||q.kept<1||q.kept>64||q.id!=expected[blockIdx.x])bad=1;
            if(!bad){double sum=0;int matches=0;
                for(int j=0;j<q.kept;++j){const double mass=q.mass[j];const int id=q.ids[j];
                    if(id<0||id>=vocabulary||mass<0||!isfinite(mass)){bad=1;break;}
                    for(int i=0;i<j;++i)if(q.ids[i]==id)bad=1;
                    sum+=mass;if(id==q.id){q_at_q=mass;++matches;}}
                if(matches!=1||!(q_at_q>0)||fabs(sum-1.)>1e-12)bad=1;
            }
        }
    }
    __syncthreads();
    const bool ready=bad==0;__syncthreads();
    if(ready&&tid<count){
        const double delta=cdf[tid]-(tid?cdf[tid-1]:0.);const int id=ids[tid];const double mass=delta/total;
        if(id<0||id>=vocabulary||delta<0||!isfinite(mass))atomicOr(&bad,1u);
        if constexpr(HasDraft){const auto &q=rows[blockIdx.x];double qm=0;
            for(int j=0;j<q.kept;++j)if(q.ids[j]==id)qm=q.mass[j];
            residual[tid]=fmax(0.,mass-qm);if(id==q.id)p_at_q=mass;
        }
    }
    __syncthreads();
    if(!tid){int chosen=0;
        if(!bad){
            if constexpr(HasDraft){const auto &q=rows[blockIdx.x];
                if(random(seed,position,1)<fmin(1.,p_at_q/q_at_q))chosen=q.id;
                else{double sum=0;for(int i=0;i<count;++i)sum+=residual[i];
                    if(!(sum>0)||!isfinite(sum))bad=1;
                    else{const double u=random(seed,position,2)*sum;double acc=0;int at=0;
                        for(;at+1<count;++at){acc+=residual[at];if(acc>=u)break;}
                        chosen=ids[at];if(!(residual[at]>0))bad=1;}
                }
            }else{const double u=random(seed,position,3)*total;int at=0;while(at+1<count&&cdf[at]<u)++at;chosen=ids[at];}
        }
        out[blockIdx.x]=bad?0:chosen;invalid[blockIdx.x]=bad;
    }
}
void validate(int vocabulary,const strata::kernels::SamplerParams &p,const MtpProbCursor *cursor,unsigned offset,
    const int *out,const unsigned *bad,int draws){
    if(vocabulary<1||!mtp_prob_params(p)||!cursor||offset>3||!out||!bad||draws<1||draws>65536)
        throw std::invalid_argument("probability coupling geometry/cursor");
}
}
void mtp_prob_draft(const strata::kernels::RadixSampler &s,int vocabulary,const strata::kernels::SamplerParams &p,
    const MtpProbCursor *cursor,unsigned offset,MtpProbRow *rows,int *out,unsigned *bad,cudaStream_t stream,int draws){
    validate(vocabulary,p,cursor,offset,out,bad,draws);if(!rows)throw std::invalid_argument("missing draft probability owner");
    draft<<<draws,128,0,stream>>>(s.sorted_ids(),s.cumulative(),s.kept_count(),s.result(),vocabulary,p.top_k,p.seed,cursor,offset,rows,out,bad);
    check_cuda(cudaGetLastError(),"retain draft probability row");
}
void mtp_prob_target(const strata::kernels::RadixSampler &s,int vocabulary,const strata::kernels::SamplerParams &p,
    const MtpProbCursor *cursor,unsigned offset,const MtpProbRow *rows,const int *expected,int *out,unsigned *bad,cudaStream_t stream,bool has_draft,int draws){
    validate(vocabulary,p,cursor,offset,out,bad,draws);if(has_draft&&(!rows||!expected))throw std::invalid_argument("missing causal draft row/ID");
    if(has_draft)target<true><<<draws,128,0,stream>>>(s.sorted_ids(),s.cumulative(),s.kept_count(),s.result(),vocabulary,p.top_k,p.seed,cursor,offset,rows,expected,out,bad);
    else target<false><<<draws,128,0,stream>>>(s.sorted_ids(),s.cumulative(),s.kept_count(),s.result(),vocabulary,p.top_k,p.seed,cursor,offset,nullptr,nullptr,out,bad);
    check_cuda(cudaGetLastError(),"probability-ratio acceptance/residual pick");
}
}
