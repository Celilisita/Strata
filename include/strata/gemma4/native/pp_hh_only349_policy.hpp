#pragma once
#include "strata/gemma4/native/model.hpp"
#include <type_traits>

namespace strata::gemma4::native::pp349 {
// Named extension of the existing fixed-int enum, without redefining Model or
// PpKeyTile. Old owners keep the exact frozen339 class/enum declaration.
static_assert(std::is_same_v<std::underlying_type_t<PpKeyTile>,int>);
static_assert(int(PpKeyTile::Keys512)==0&&int(PpKeyTile::Keys1024)==1);
inline constexpr PpKeyTile Keys1024HH349=static_cast<PpKeyTile>(2);
inline bool is_hh(PpKeyTile mode){return mode==Keys1024HH349;}
inline bool is_k1024(PpKeyTile mode){return mode==PpKeyTile::Keys1024||is_hh(mode);}
inline bool supported_chunk(int capacity,int tokens,int past){
    return (capacity==10240&&tokens==8192&&past==0)||
        (capacity==32768&&((tokens==8192&&(past==0||past==8192||past==16384))||
                          (tokens==6144&&past==24576)));
}
// Existing selector clears logical/333 counts when mode changes; reset also
// clears counts while preserving selection. No request can mix HH/other leaves.
inline size_t hh_leaf_enqueues(const Model& m){
    return is_hh(m.pp_key_tile())?m.pp_k1024_enqueues():0;
}
inline size_t four_product_leaf_enqueues(const Model& m){
    return is_hh(m.pp_key_tile())?0:m.pp_four_product324_enqueues();
}
inline bool hh_request(const Model& m){
    return is_hh(m.pp_key_tile())&&m.compensated_physical_global_pp_request();
}
// Future driver must verify immutable per-length unit/Root capability before
// calling this cold selector. Compiled availability does not grant admission.
// Do not emit the legacy pp_four_product324_enqueues() getter for HH mode: its
// old logical-minus333 meaning predates HH. Use the explicit helper above.
}
