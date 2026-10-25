#pragma once
#include "half_row_pair.hpp"
#include <cstdint>
namespace strata::gemma4::native {
struct Q4PpPanelBinding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct Q4PpPanelLease {HalfRowPairLease operands;Q4PpPanelBinding binding;};
struct Q4PpPanelResources {int registers,local_bytes,shared_bytes,max_threads;};
Q4PpPanelResources q4_pp_panel_resources(); // Cold loaded-kernel attributes only.
void q4_pp_panel_project(primitive::Matrix,primitive::Matrix,const Q4PpPanelLease&,Q4PpPanelBinding);
namespace detail298 { // Bounded diagnostics: caller validates before any phase.
void validate(primitive::Matrix,primitive::Matrix,const Q4PpPanelLease&,Q4PpPanelBinding);
void scale(const HalfRowPairLease&);
void weights(primitive::Matrix,primitive::Matrix,const HalfRowPairLease&,bool candidate);
void input(const HalfRowPairLease&);
void products(const HalfRowPairLease&);
}
}
