#include "strata/qwen35moe/runtime.hpp"
#include "strata/prefill/native_blackwell_dense_rows.hpp"
#include "strata/qwen35moe/nvfp4_prefix_rows.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe {
bool Runtime::batch_project_blackwell(const GpuWeight& weight, float* output,
                                     int count, int n_in, int n_out) {
    const char* enabled = std::getenv("STRATA_BENCH_DENSE_GROUP_F16");
    if (enabled && std::strcmp(enabled,"0") && std::strcmp(enabled,"1"))
        throw std::invalid_argument("dense group F16 requires0 or1");
    const char* setting = std::getenv("STRATA_BENCH_DENSE_GROUP_MIN_ROWS");
    const int minimum = setting ? std::stoi(setting) : 1024;
    if (minimum != 256 && minimum != 1024)
        throw std::invalid_argument("dense group minimum requires256 or1024");
    const bool q5 = weight.type==13 && prefill::mmq::blackwell_q5_tc_enabled();
    const bool nv_small=nvfp4_&&(weight.type==12||weight.type==13)&&
        nvfp4_prefix_detail::small(count);
    if (decode_phase_ || (!nv_small&&count < minimum) || (weight.type != 12 && weight.type != 14 && !q5) ||
        !enabled || std::strcmp(enabled,"1")) return false;
    batch_.half_width = 0;
    batch_.half_count = 0;
    static const unsigned prefix_formats=[] {
        const char* value=std::getenv("STRATA_QWEN35_DENSE_PREFIX_I64");
        if(!value||!std::strcmp(value,"0"))return 0u;
        if(!std::strcmp(value,"12"))return 1u<<12;
        if(!std::strcmp(value,"13"))return 1u<<13;
        if(!std::strcmp(value,"12,13"))return (1u<<12)|(1u<<13);
        throw std::invalid_argument("dense prefix I64 requires0/12/13/12,13");
    }();
    const bool prefix=(count==256||nv_small)&&(weight.type==12||weight.type==13)&&
        (prefix_formats&(1u<<weight.type))&&!checkpoint_sink_&&!paired_session_&&!sessions_&&
        !mtp_enabled_&&!verify_state_&&!verify_history_&&
        ((weight.type==12&&((n_in==2048&&(n_out==4096||n_out==8192))||
                           (n_in==4096&&n_out==2048)))||
         (weight.type==13&&n_in==2048&&n_out==8192));
    if(prefix) {
        prefill::mmq::run_blackwell_dense_wide_int8_tile(weight.type,weight.ptr(),batch_.input_q8.data(),
            batch_.input_half.data(),batch_.input_half.bytes(),batch_.input_low.data(),batch_.input_low.bytes(),
            output,n_in,count,n_out,stream_,64);
        dense_profile_.launches(2); // Existing explicit encoder and product, no extra work.
        return true;
    }
    const auto launches = prefill::mmq::run_blackwell_dense_rows(weight.type, weight.ptr(),
        batch_.input_q8.data(), batch_.input_q8.bytes(),
        batch_.input_half.data(), batch_.input_half.bytes(),
        batch_.input_low.data(), batch_.input_low.bytes(),
        output, size_t(count) * size_t(n_out) * sizeof(float), n_in, count, n_out, stream_);
    dense_profile_.launches(launches);
    return true;
}
} // namespace strata::qwen35moe
