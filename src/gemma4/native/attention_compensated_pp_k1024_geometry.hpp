#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native {
constexpr int CompensatedPpK1024Queries=512,CompensatedPpK1024Keys=1024;
constexpr int CompensatedPpK1024Heads=32,CompensatedPpK1024KvHeads=4,CompensatedPpK1024Dim=512;
constexpr std::size_t CompensatedPpK1024ArenaLimit=462422016,CompensatedPpK1024DeadLimit=134217728;
struct CompensatedPpK1024Layout {
    int end;
    std::size_t k_bytes,dead_bytes,q,p,work,acc,maximum,sum,a,b,tables,arena_bytes;
};
inline CompensatedPpK1024Layout compensated_pp_k1024_layout(int past,int tokens){
    const bool eligible=(tokens==8192&&(past==0||past==8192||past==16384))||
        (tokens==6144&&past==24576);
    if(!eligible)throw std::invalid_argument("Gemma324 exact30720 PP chunk");
    static_assert(sizeof(void*)==8,"Gemma324 device pointers require64bit");
    CompensatedPpK1024Layout l{};l.end=past+tokens;
    l.k_bytes=std::size_t(4)*l.end*512*4;l.dead_bytes=l.k_bytes/2;
    constexpr std::size_t query=std::size_t(CompensatedPpK1024Heads)*CompensatedPpK1024Queries*CompensatedPpK1024Dim*4;
    constexpr std::size_t score=std::size_t(CompensatedPpK1024Heads)*CompensatedPpK1024Queries*CompensatedPpK1024Keys*4;
    constexpr std::size_t rows=std::size_t(CompensatedPpK1024Heads)*CompensatedPpK1024Queries*4;
    std::size_t at=l.k_bytes;
    auto take=[&](std::size_t bytes){auto first=at;at+=bytes;return first;};
    l.q=take(query);l.p=take(score);l.work=take(score>query?score:query);l.acc=take(query);
    l.maximum=take(rows);l.sum=take(rows);l.a=take(rows);l.b=take(rows);
    l.tables=take(10*4*sizeof(void*));l.arena_bytes=at;
    if(l.arena_bytes>CompensatedPpK1024ArenaLimit||l.dead_bytes>CompensatedPpK1024DeadLimit)
        throw std::invalid_argument("Gemma324 borrowed memory budget");
    return l;
}
}
