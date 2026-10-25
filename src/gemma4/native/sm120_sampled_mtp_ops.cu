#include "sm120_sampled_mtp.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <cmath>
#include <stdexcept>
namespace strata::gemma4::native::sampled_mtp {
namespace {
__global__ void prepare(Round* s,const int* position,const unsigned* invalid,int local,int global,int context){
    const int p=*position;s->start=p;s->invalid=*invalid;s->response={};s->next=0;
    s->drafts[0]=s->drafts[1]=0;s->bounds={};
    if(p<1||p>=context||p>global){s->invalid|=32u;return;}
    const int origin=p>local?p-local:0;
    s->bounds={p,p-1,(p-1025>origin?p-1025:origin),origin,p-origin,0,p};
}
__global__ void save(Round* s,const int* proposal,const unsigned* invalid,int slot){
    const int id=*proposal;s->drafts[slot]=id>=0&&id<262144?id:0;
    s->invalid|=*invalid|(id>=0&&id<262144?0u:32u);
}
__global__ void inputs(Round* s,const int* pending,int drafts){
    s->inputs[0]=*pending;s->inputs[1]=drafts?s->drafts[0]:0;s->inputs[2]=drafts==2?s->drafts[1]:0;
    if(*pending<0||*pending>=262144)s->invalid|=32u;
}
__global__ void histories(Round* s,const int* history,int* rows,int length,int drafts,const unsigned* invalid){
    if(threadIdx.x==0&&blockIdx.x==0)s->invalid|=*invalid;
    const int i=int(blockIdx.x)*256+threadIdx.x;
    for(int row=0;row<=drafts;++row)if(i<length){
        // History is already the target row0 prefix, including pending once.
        // Future row t has that suffix followed by exactly drafts[0..t-1].
        const int at=i+row;
        rows[row*length+i]=at<length?history[at]:s->drafts[at-length];
    }
}
__global__ void decision(Round* s,int drafts,int capacity,const int* eog,int neog,const unsigned* invalid){
    Response r{};r.proposed=drafts;r.position=s->start;s->invalid|=*invalid;
    if(s->invalid||s->remaining<1||s->start<1||s->start>=capacity){
        s->invalid|=64u;r.invalid=s->invalid;r.reason=int(Stop::Invalid);r.finished=1;s->response=r;return;
    }
    const int limit=min(min(drafts+1,s->remaining),capacity-s->start);
    for(int i=0;i<limit;++i){const auto pick=s->picks[i];
        if(pick.first_bad!=-1||pick.id<0||pick.id>=262144||!isfinite(pick.logit)||!isfinite(pick.probability)){
            s->invalid|=128u;r={};r.proposed=drafts;r.position=s->start;r.invalid=s->invalid;r.finished=1;r.reason=int(Stop::Invalid);s->response=r;return;}
        r.ids[r.count++]=pick.id;const bool match=i<drafts&&pick.id==s->drafts[i];r.matched+=int(match);
        bool stop=false;for(int j=0;j<neog;++j)stop|=pick.id==eog[j];
        if(stop){r.finished=1;r.reason=int(Stop::Eog);break;}
        if(r.count==s->remaining){r.finished=1;r.reason=int(Stop::Budget);break;}
        if(!match)break;
    }
    // accepted counts extra emitted tokens beyond one ordinary target draw;
    // matched separately preserves an EOS proposal match without double count.
    r.accepted=r.count-1;r.position=s->start+r.count;s->next=r.ids[r.count-1];s->response=r;
}
__global__ void finish(Round* s,const int* position,const unsigned* invalid){
    s->invalid|=*invalid;s->response.invalid=s->invalid;s->response.position=*position;
    if(s->invalid||*position!=s->start+s->response.count){
        s->response.invalid|=256u;s->response.reason=int(Stop::Invalid);s->response.finished=1;
    }
}
void checked(const void* a,cudaStream_t stream){if(!a||!stream)throw std::invalid_argument("sampled MTP device lease");}
}
void prepare_round(Round* s,const int* p,const unsigned* bad,int local,int global,int context,cudaStream_t stream){
    checked(s,stream);checked(p,stream);checked(bad,stream);if(local<1||global<1||context<2)throw std::invalid_argument("sampled MTP bounds");
    prepare<<<1,1,0,stream>>>(s,p,bad,local,global,context);primitive::check_cuda(cudaGetLastError(),"sampled MTP bounds");}
void save_draft(Round* s,const int* p,const unsigned* bad,int slot,cudaStream_t stream){
    checked(s,stream);checked(p,stream);checked(bad,stream);if(slot<0||slot>1)throw std::invalid_argument("sampled MTP proposal slot");
    save<<<1,1,0,stream>>>(s,p,bad,slot);primitive::check_cuda(cudaGetLastError(),"sampled MTP proposal");}
void form_inputs(Round* s,const int* pending,int drafts,cudaStream_t stream){
    checked(s,stream);checked(pending,stream);if(drafts<0||drafts>2)throw std::invalid_argument("sampled MTP rows");
    inputs<<<1,1,0,stream>>>(s,pending,drafts);primitive::check_cuda(cudaGetLastError(),"sampled MTP inputs");}
void prepare_histories(Round* s,const int* history,int* rows,int length,int drafts,const unsigned* bad,cudaStream_t stream){
    checked(s,stream);checked(bad,stream);if(length<0||(length&&(!history||!rows))||drafts<0||drafts>2)throw std::invalid_argument("sampled MTP histories");
    histories<<<(length+255)/256?(length+255)/256:1,256,0,stream>>>(s,history,rows,length,drafts,bad);
    primitive::check_cuda(cudaGetLastError(),"sampled MTP row histories");}
void decide(Round* s,int drafts,int capacity,const int* eog,int count,const unsigned* bad,cudaStream_t stream){
    checked(s,stream);checked(bad,stream);if(drafts<0||drafts>2||count<0||count>16||(count&&!eog))throw std::invalid_argument("sampled MTP decision policy");
    decision<<<1,1,0,stream>>>(s,drafts,capacity,eog,count,bad);primitive::check_cuda(cudaGetLastError(),"sampled MTP sample-and-match prefix");}
void finalize(Round* s,const int* position,const unsigned* invalid,cudaStream_t stream){
    checked(s,stream);checked(position,stream);checked(invalid,stream);
    finish<<<1,1,0,stream>>>(s,position,invalid);primitive::check_cuda(cudaGetLastError(),"sampled MTP final commit response");}
}
