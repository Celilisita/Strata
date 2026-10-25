#include "attention_swa_detail.hpp"
#include <limits>

namespace strata::gemma4::native::swa403 {
namespace {
constexpr std::size_t Rows=Heads*QueryTile;
constexpr std::size_t Scores=Rows*KeyTile,Output=Rows*Dim,Kv=KvHeads*KeyTile*Dim;
struct Span {const void* data;std::size_t bytes,alignment;};
void disjoint(const Span* spans,std::size_t count){
    for(std::size_t i=0;i<count;++i){
        const auto s=spans[i];const auto p=reinterpret_cast<std::uintptr_t>(s.data);
        if(!s.data||!s.bytes||p%s.alignment||s.bytes>(std::numeric_limits<std::uintptr_t>::max)()-p)
            throw std::invalid_argument("Gemma403 span extent/alignment/overflow");
        for(std::size_t j=0;j<i;++j){
            const auto q=reinterpret_cast<std::uintptr_t>(spans[j].data);
            if(p<q+spans[j].bytes&&q<p+s.bytes)
                throw std::invalid_argument("Gemma403 overlapping live spans");
        }
    }
}
struct Empty {
    void query(detail403::Tile,int)const{}
    void qk(detail403::Tile,int,int,int)const{}
    void probability(detail403::Tile,int,int,int)const{}
    void pv(detail403::Tile,int,int,int)const{}
    void finish(detail403::Tile,const float*,int)const{}
};
}
std::size_t workspace_bytes(){
    return (Scores+2*Output+4*Rows)*sizeof(float)+(Output+2*Kv+2*Scores)*sizeof(half);
}
namespace detail403 {
Tile bind(void* data,std::size_t bytes){
    if(!data||bytes<workspace_bytes())throw std::invalid_argument("Gemma403 workspace extent");
    auto* f=static_cast<float*>(data);
    auto take=[&](std::size_t n){auto* p=f;f+=n;return p;};
    native::detail::AttentionTile state{nullptr,nullptr,take(Scores),take(Output),take(Output),
        take(Rows),take(Rows),take(Rows),take(Rows)};
    auto* h=reinterpret_cast<half*>(f);
    auto half_take=[&](std::size_t n){auto* p=h;h+=n;return p;};
    return {state,half_take(Output),half_take(Kv),half_take(Kv),half_take(Scores),half_take(Scores)};
}
void validate(const float* q,float* out,CacheView c,const int* p,int past,int heads,
    int tokens,Lease lease,Binding live){
    const auto b=lease.binding;const auto w=lease.workspace;
    const std::int64_t end=std::int64_t(past)+tokens;
    const std::int64_t required=(std::min)(end,std::int64_t(Window)+tokens-1);
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||
       b.stream!=live.stream||heads!=Heads||c.heads!=KvHeads||c.dim!=Dim||c.window!=Window||
       (tokens!=8192&&tokens!=6144)||past<0||end>262144||c.capacity<required||c.capacity>32768||
       !w.handle||w.bytes<workspace_bytes()||!w.blas_bytes)
        throw std::invalid_argument("Gemma403 SWA live lease/shape/history contract");
    const auto values=std::size_t(tokens)*Heads*Dim;
    const auto cachebytes=std::size_t(c.capacity)*KvHeads*(Dim/32)*34;
    const Span spans[]={{q,values*sizeof(float),16},{out,values*sizeof(float),16},
        {c.key,cachebytes,2},{c.value,cachebytes,2},{p,sizeof(int),alignof(int)},
        {w.data,w.bytes,16},{w.blas_data,w.blas_bytes,256}};
    disjoint(spans,sizeof(spans)/sizeof(spans[0]));
}
}
void attention(const float* q,float* out,CacheView c,const int* position,
    int past,int heads,int tokens,Lease lease,Binding live){
    Empty observer;detail403::execute(q,out,c,position,past,heads,tokens,lease,live,observer);
    record_enqueue(live);
}
}
