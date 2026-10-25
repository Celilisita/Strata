#pragma once
namespace strata::qwen35::native {
inline bool mtp_prob_default(int resident_bound){
    constexpr unsigned mask=2u; // MEASURED_PROB_RESOURCE_MASK
    return resident_bound>0&&(mask&(resident_bound<=8192?1u:2u));
}
}
