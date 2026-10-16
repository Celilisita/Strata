#pragma once
#include "small_prefill_mma.cuh"
#include "strata/qwen35/native/small_prefill.hpp"
#include <limits>

namespace strata::qwen35::native::detail {
constexpr int DirectRows=64,DirectTokens=32,DirectSplits=3;
template<Quant Type,bool PackedCodes=false,bool CompactInput=false>
constexpr int direct_shared_bytes(int words){
    const int bytes=Type==Quant::Q4K?144:(PackedCodes?(words==1?288:32):210);
    const int stride=Type==Quant::Q6K && !PackedCodes && CompactInput && words==2?240:(bytes+15)&~15;
    const int weight_words=words==1?32*DirectRows:detail::HalfOperandPitch*DirectRows;
    const int input_words=words==1?32*DirectTokens:(CompactInput?detail::CompactHalfOperandPitch:detail::HalfOperandPitch)*DirectTokens;
    return DirectRows*stride+(weight_words+input_words)*4+8*DirectTokens*(Type==Quant::Q6K?4:8);
}
inline bool split_overlap(const void *one,size_t one_bytes,const void *two,size_t two_bytes){
    const auto a=reinterpret_cast<uintptr_t>(one),b=reinterpret_cast<uintptr_t>(two);
    return a<=b?b-a<one_bytes:a-b<two_bytes;
}
template<bool HalfWeight>
size_t direct_codes_split_bytes(Matrix weight,int tokens){
    if constexpr(!HalfWeight)matrix_bytes(weight,true);
    const int limit=(std::numeric_limits<int>::max)();
    if(weight.quant!=Quant::Q6K||weight.layout!=(HalfWeight?BlockLayout::Gguf:BlockLayout::Q6Codes)||
       reinterpret_cast<uintptr_t>(weight.data)%16||tokens!=DirectTokens||weight.rows%DirectRows||
       weight.columns/256<DirectSplits||weight.rows>limit-127||
       int64_t((weight.rows+127)/128)*(weight.columns/256)>limit)
        throw std::invalid_argument("split Q6 requires aligned codes, complete PP32 row tiles and three K ranges");
    return size_t(DirectSplits)*size_t(tokens)*size_t(weight.rows)*sizeof(float);
}
}
