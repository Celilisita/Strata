#pragma once
#include "half_row_pair.hpp"

namespace strata::gemma4::native {
// Unit-only fixed K1792x3 alternative to the retained K896x6 path.
// Same RN16 operands, global scale, F32 interleaved output and owner arenas.
// No Model/Ffn dispatch selects this function; all other shapes are rejected.
void half_row_pair_three_panel_project(primitive::Matrix gate, primitive::Matrix up,
    const HalfRowPairLease&);
}
