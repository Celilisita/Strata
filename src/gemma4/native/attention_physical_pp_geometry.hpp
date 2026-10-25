#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::gemma4::native {
constexpr int PhysicalPpQueries=1024,PhysicalPpKeys=512;
constexpr int PhysicalPpHeads=32,PhysicalPpKvHeads=4,PhysicalPpDim=512;
constexpr std::size_t PhysicalPpArenaLimit=462422016;
constexpr std::size_t PhysicalPpDeadLimit=134217728;

struct PhysicalPpLayout {
    int end;
    std::size_t k_bytes,dead_bytes;
    std::size_t scores,partial,acc,maximum,sum,a,b;
    std::size_t key_table,value_table,query_table,score_table,partial_table;
    std::size_t arena_bytes;
};

inline PhysicalPpLayout physical_pp_layout(int past,int tokens){
    const bool chunk=(tokens==8192&&(past==0||past==8192||past==16384))||
        (tokens==6144&&past==24576);
    if(!chunk)throw std::invalid_argument("Gemma241 requires one real30720 PP chunk");
    static_assert(sizeof(void*)==8,"Gemma241 device pointer tables require64bit");
    PhysicalPpLayout l{};
    l.end=past+tokens;
    l.k_bytes=std::size_t(PhysicalPpKvHeads)*l.end*PhysicalPpDim*sizeof(float);
    l.dead_bytes=l.k_bytes/2;
    const std::size_t matrix=std::size_t(PhysicalPpHeads)*PhysicalPpQueries*
        PhysicalPpDim*sizeof(float);
    const std::size_t rows=std::size_t(PhysicalPpHeads)*PhysicalPpQueries*sizeof(float);
    const std::size_t table=PhysicalPpHeads*sizeof(float*);
    std::size_t at=l.k_bytes;
    auto take=[&](std::size_t bytes){const auto first=at;at+=bytes;return first;};
    l.scores=take(matrix);l.partial=take(matrix);l.acc=take(matrix);
    l.maximum=take(rows);l.sum=take(rows);l.a=take(rows);l.b=take(rows);
    l.key_table=take(table);l.value_table=take(table);l.query_table=take(table);
    l.score_table=take(table);l.partial_table=take(table);l.arena_bytes=at;
    if(l.arena_bytes>PhysicalPpArenaLimit||l.dead_bytes>PhysicalPpDeadLimit)
        throw std::invalid_argument("Gemma241 borrowed memory budget");
    return l;
}

// The actual operator uses this same host-only range check before enqueuing.
// Validate full advertised leases, including their untouched canary suffixes.
struct PhysicalPpSpan {const void* data;std::size_t bytes,alignment;};
inline void physical_pp_validate_spans(const PhysicalPpSpan* spans,std::size_t count){
    if(!spans||!count)throw std::invalid_argument("Gemma241 missing borrowed spans");
    for(std::size_t i=0;i<count;++i){
        const auto& s=spans[i];
        const auto first=reinterpret_cast<std::uintptr_t>(s.data);
        if(!s.data||!s.bytes||!s.alignment||(s.alignment&(s.alignment-1))||
           first%s.alignment||s.bytes>std::numeric_limits<std::uintptr_t>::max()-first)
            throw std::invalid_argument("Gemma241 pointer extent/alignment");
        for(std::size_t j=0;j<i;++j){
            const auto other=reinterpret_cast<std::uintptr_t>(spans[j].data);
            if(first<other+spans[j].bytes&&other<first+s.bytes)
                throw std::invalid_argument("Gemma241 overlapping borrowed spans");
        }
    }
}

struct PhysicalPpHeadOffsets {std::size_t k,v,q,score,partial;int value_bank;};
#if defined(__CUDACC__)
#define STRATA241_HD __host__ __device__
#else
#define STRATA241_HD
#endif
// Float offsets consumed by the GPU pointer-table kernel, also inspectable
// without CUDA in the bounded host geometry check. Call with validated shapes.
STRATA241_HD inline PhysicalPpHeadOffsets physical_pp_head_offsets(
    int end,int head,int query_first,int key_first){
    const int kh=head/(PhysicalPpHeads/PhysicalPpKvHeads);
    const std::size_t matrix=std::size_t(PhysicalPpQueries)*PhysicalPpDim;
    return {(std::size_t(kh)*end+key_first)*PhysicalPpDim,
        (std::size_t(kh%2)*end+key_first)*PhysicalPpDim,
        (std::size_t(query_first)*PhysicalPpHeads+head)*PhysicalPpDim,
        std::size_t(head)*matrix,std::size_t(head)*matrix,kh/2};
}
#undef STRATA241_HD
}
