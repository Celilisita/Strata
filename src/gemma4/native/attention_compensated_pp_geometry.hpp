#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native {
constexpr int CompensatedPpQueries=512,CompensatedPpKeys=512;
constexpr int CompensatedPpHeads=32,CompensatedPpKvHeads=4,CompensatedPpDim=512;
constexpr std::size_t CompensatedPpArenaLimit=462422016,CompensatedPpDeadLimit=134217728;
struct CompensatedPpLayout {
    int end;
    std::size_t k_bytes,dead_bytes,q,p,work,acc,maximum,sum,a,b,tables,arena_bytes;
};
inline CompensatedPpLayout compensated_pp_layout(int past,int tokens){
    const bool eligible=(tokens==8192&&(past==0||past==8192||past==16384))||
        (tokens==6144&&past==24576);
    if(!eligible)throw std::invalid_argument("Gemma286 exact30720 PP chunk");
    static_assert(sizeof(void*)==8,"Gemma286 device pointers require64bit");
    CompensatedPpLayout l{};l.end=past+tokens;
    l.k_bytes=std::size_t(4)*l.end*512*4;l.dead_bytes=l.k_bytes/2;
    constexpr std::size_t matrix=std::size_t(32)*512*512*4,rows=std::size_t(32)*512*4;
    std::size_t at=l.k_bytes;
    auto take=[&](std::size_t bytes){auto first=at;at+=bytes;return first;};
    l.q=take(matrix);l.p=take(matrix);l.work=take(matrix);l.acc=take(matrix);
    l.maximum=take(rows);l.sum=take(rows);l.a=take(rows);l.b=take(rows);
    l.tables=take(9*4*sizeof(void*));l.arena_bytes=at;
    if(l.arena_bytes>CompensatedPpArenaLimit||l.dead_bytes>CompensatedPpDeadLimit)
        throw std::invalid_argument("Gemma286 borrowed memory budget");
    return l;
}
}
