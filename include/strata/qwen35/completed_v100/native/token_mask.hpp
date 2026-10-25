#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace strata::completed_v100::qwen35::native {
// Request sampling policy, copied into captured kernel arguments. Logits stay
// unchanged for numerical diagnostics; nonfinite logits are still rejected.
struct TokenMask {int ids[16]{};int count=0,first=0,last=-1;};
inline TokenMask make_token_mask(std::vector<int> ids,int vocabulary){
    if(ids.size()>16||vocabulary<1)throw std::invalid_argument("invalid token mask extent");
    std::sort(ids.begin(),ids.end());
    if(std::adjacent_find(ids.begin(),ids.end())!=ids.end()||ids.size()>=size_t(vocabulary))
        throw std::invalid_argument("duplicate or exhaustive token mask");
    TokenMask out;
    for(int id:ids){
        if(id<0||id>=vocabulary)throw std::out_of_range("masked token outside vocabulary");
        out.ids[out.count++]=id;
    }
    if(out.count){out.first=ids.front();out.last=ids.back();}return out;
}
}
