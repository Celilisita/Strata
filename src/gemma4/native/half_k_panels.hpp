#pragma once
#include "half_row_pair.hpp"

namespace strata::gemma4::native {
// Fixed direct-encoded operand panels: W[6][2*21504][896], X[6][1280][896].
// Output stays the197 typed interleaved [token][Gate;Up] layout. The caller
// uses the existing197 GELU and191 fullDown after this function consumes X.
// No half weight cache survives this projection or borrows an extra arena.
void half_row_pair_panel_project(primitive::Matrix gate,primitive::Matrix up,
    const HalfRowPairLease&);
}
