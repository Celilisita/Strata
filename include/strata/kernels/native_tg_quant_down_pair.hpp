#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::kernels {
// Original buffers and the existing eight-entry directory. shared_q8 is the
// first576B of Runtime's input_q8, after its norm activation has become dead.
struct NativeTgQuantDownPair {
    const float* expert_hidden;
    const float* shared_hidden;
    const unsigned long long* down_addresses;
    const int32_t* down_rows;
    void* expert_q8;
    void* shared_q8;
    const void* shared_down;
    float* expert_parts;
    float* shared_output;
};
struct NativeTgQuantDownPairResources {
    int device,major,minor,sms,binary,registers,threads,max_threads,active_blocks,reason_mask;
    size_t shared,local;
};
extern "C" int strata_tg_quant_down_pair_resources(int type,int stage,int* values);
inline NativeTgQuantDownPairResources native_tg_quant_down_pair_resources(int type,int stage){
    int v[12]{};if(strata_tg_quant_down_pair_resources(type,stage,v)!=0)
        throw std::runtime_error("TG paired quant/Down resource query failed");
    return {v[0],v[1],v[2],v[3],v[4],v[5],stage?128:256,v[6],v[7],v[8],size_t(v[9]),size_t(v[10])};
}
// Resources for both kernels are checked before either output can be mutated.
// False means untouched unsupported-resource fallback; malformed arguments throw.
bool native_tg_quant_down_pair_launch(int type,const float*,const float*,const unsigned long long*,
    const int32_t*,void*,void*,const void*,float*,float*,void*);
inline bool native_tg_quant_down_pair(int type,const NativeTgQuantDownPair& a,void* stream){
    return native_tg_quant_down_pair_launch(type,a.expert_hidden,a.shared_hidden,a.down_addresses,
        a.down_rows,a.expert_q8,a.shared_q8,a.shared_down,a.expert_parts,a.shared_output,stream);
}
// Counts successful host submissions, including captures, but never graph replays.
unsigned long long native_tg_quant_down_pair_submissions();
}
