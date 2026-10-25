#pragma once
#include "strata/prefill/native_grouped_mmq.hpp"
namespace strata::prefill::mmq {
struct NativeOrdinaryGroupedMmqResources {
    int cc=0,highest_arch=0,mma_layout=0,I=0,J=0,threads=0,K_vram=0;
    int registers=0,binary=0,active_blocks=0,reason_mask=0;
    size_t shared=0,local=0;
};
// Q5 only: strict default-off flag plus actual resource eligibility, before any plan/lease write.
bool native_q5_down_ordinary_eligible();
// Internal, read-only eligibility before the original plan/lease/packed mutation.
NativeOrdinaryGroupedMmqResources native_ordinary_grouped_resources(int type,bool active);
bool native_ordinary_grouped_supported(int type,int cols,int outputs,int parts,bool active);
void ordinary_grouped_repack(const void*,void*,int,int,int,int,const int32_t*,void*);
void ordinary_grouped_tile(int,const NativeMmqMatrix*,int,int,const int32_t*,const void*,float*,
    int,int,int,int,void*,const NativeMmqTile*,int);
}
