#include "attention_gqa_t2_ops.hpp"
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native::gqa371 {
Layout layout(int capacity){
    if(capacity!=10240&&capacity!=32768)throw std::invalid_argument("371 only reserved8K/30K T2 capacities");
    static_assert(10240%PvKeys==0&&32768%PvKeys==0,"PV tiles cover both fixed capacities");
    Layout l{};l.capacity=capacity;l.parts=capacity/PvKeys;l.kv_bytes=std::size_t(KvHeads)*capacity*Dim*sizeof(half);
    std::size_t at=0;
    auto take=[&](std::size_t bytes){const auto first=at;at=(at+bytes+255)/256*256;return first;};
    constexpr std::size_t output_elements=std::size_t(KvHeads)*Rows*Dim;
    const std::size_t scores=std::size_t(KvHeads)*Rows*capacity;
    l.query=take(output_elements*sizeof(half));l.score=take(scores*sizeof(float));
    l.probability=take(scores*sizeof(half));l.partial=take(output_elements*sizeof(float));
    l.maximum=take(KvHeads*Rows*sizeof(float));l.denominator=take(KvHeads*Rows*sizeof(float));
    l.query_bad=take(KvHeads*Rows*sizeof(unsigned));l.first_bad_key=take(KvHeads*sizeof(int));
    l.split_partial=take(std::size_t(KvHeads)*l.parts*Rows*Dim*sizeof(float));l.arena_bytes=at;
    l.kv_stride=static_cast<long long>(capacity)*Dim;l.query_stride=Rows*Dim;
    l.score_stride=static_cast<long long>(capacity)*Rows;l.output_stride=Rows*Dim;
    l.pv_kv_stride=static_cast<long long>(PvKeys)*Dim;
    l.pv_probability_stride=Rows*PvKeys;l.pv_partial_stride=Rows*Dim;
    if(l.kv_bytes>RawLimit||l.arena_bytes>ArenaLimit)throw std::invalid_argument("371 borrowed budget");
    return l;
}
namespace detail {
namespace {
struct Span {const void* data;std::size_t bytes,alignment;};
void spans(const Span* values,std::size_t count){
    for(std::size_t i=0;i<count;++i){
        const auto& s=values[i];const auto p=reinterpret_cast<std::uintptr_t>(s.data);
        if(!s.data||!s.bytes||p%s.alignment||s.bytes>(std::numeric_limits<std::uintptr_t>::max)()-p)
            throw std::invalid_argument("371 full advertised span/alignment/overflow");
        for(std::size_t j=0;j<i;++j){const auto q=reinterpret_cast<std::uintptr_t>(values[j].data);
            if(p<q+values[j].bytes&&q<p+s.bytes)throw std::invalid_argument("371 full advertised spans overlap");}
    }
}
}
void validate(const float* q,float* out,CacheView c,const int* position,int heads,int tokens,
    AttentionWorkspace w,Lease lease,Binding live,const unsigned* invalid,const Layout& l){
    const auto b=lease.binding;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream||
       heads!=Heads||tokens!=Queries||c.heads!=KvHeads||c.dim!=Dim||c.window!=0||c.capacity!=l.capacity||
       lease.dead_k_bytes!=RawLimit||lease.dead_v_bytes!=RawLimit||!w.handle||
       w.bytes<l.arena_bytes||w.bytes>ArenaLimit||w.blas_bytes!=BlasBytes)
        throw std::invalid_argument("371 physical global T2/live borrowed lease");
    constexpr std::size_t io=std::size_t(Queries)*Heads*Dim*sizeof(float);
    const std::size_t kv=std::size_t(c.capacity)*KvHeads*(Dim/32)*34;
    const Span all[]={{q,io,4},{out,io,4},{c.key,kv,2},{c.value,kv,2},{position,4,4},
        {w.data,w.bytes,16},{w.blas_data,w.blas_bytes,256},
        {lease.dead_k,lease.dead_k_bytes,16},{lease.dead_v,lease.dead_v_bytes,16},
        {invalid,invalid?sizeof(unsigned):0,alignof(unsigned)}};
    spans(all,sizeof(all)/sizeof(all[0])-(invalid?0:1));
}
}
}
