#include "attention_hh_only_detail.hpp"
namespace strata::gemma4::native {
namespace {
struct EmptyObserver {
    template<class... T> void query_tile(T&&...)const noexcept{}
    template<class... T> void qk(T&&...)const noexcept{}
    template<class... T> void probability(T&&...)const noexcept{}
    template<class... T> void pv(T&&...)const noexcept{}
    template<class... T> void finish(T&&...)const noexcept{}
};
}
void hh_only_physical_pp_349_attention(const float* q,float* out,CacheView cache,
    const int* position,int past,int heads,int tokens,AttentionWorkspace work,
    CompensatedPpK1024Lease lease,CompensatedPpK1024Binding live){
    EmptyObserver observer;
    detail349::execute(q,out,cache,position,past,heads,tokens,work,lease,live,observer);
}
}
