#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/prefill/precise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::qwen35moe {
bool Runtime::batch_project_precise_dense(const GpuWeight& weight, float* output,
                                          int count, int n_in, int n_out) {
    const unsigned formats = prefill_config_.precise_dense_types;
    // Called only by batch_project_mmq: covers explicit GDN QKV/out and
    // !allow_tensor projections; shared allow_tensor=true does not enter here.
    if (!formats || decode_phase_ || checkpoint_sink_ || count < 1024 || !prefill_tc_ || !prefill_scan_ ||
        (weight.type != 12 && weight.type != 13 && weight.type != 14) || !(formats & (1u << weight.type)) ||
        (n_in != 2048 && n_in != 4096) || n_out < 2048 || n_out > 8192 || n_out % 16) return false;
    if (dense_profile_.active())
        throw std::invalid_argument("precise dense TC launch profile is uncalibrated; disable DENSE_EVENTS");
    auto& b = batch_;
    const size_t values = size_t(count)*n_in;
    const size_t weight_bytes = kernels::native_mmvq_weight_bytes(weight.type,n_in,n_out);
    if (count > b.capacity || !weight.ptr() || !output || weight.data.bytes() < weight_bytes ||
        b.input_q8.bytes() < size_t(count)*kernels::native_q8_1_bytes(n_in) ||
        b.input_half.bytes() < values*2 || b.input_low.bytes() < values*2 ||
        !b.scan_activated.data() || !b.gemm.scratch() || b.gemm.stream() != stream_)
        throw std::invalid_argument("unprepared precise dense phase workspace");
    // Identical capacity/alignment, Q8 reconstruction and three-term call order
    // as the independent full-cost gate. No persistent dequantized weight cache.
    const int64_t elements = (std::min)(int64_t(b.scan_activated.bytes()/6), b.gemm.scratch_elems())/n_in*n_in;
    if (elements < n_in) throw std::invalid_argument("unreserved precise dense weight chunk");
    const std::array<std::pair<uintptr_t,size_t>,7> spans{{
        {uintptr_t(weight.ptr()),weight_bytes},
        {uintptr_t(b.input_q8.data()),size_t(count)*kernels::native_q8_1_bytes(n_in)},
        {uintptr_t(b.input_half.data()),values*2}, {uintptr_t(b.input_low.data()),values*2},
        {uintptr_t(b.scan_activated.data()),b.scan_activated.bytes()},
        {uintptr_t(b.gemm.scratch()),size_t(b.gemm.scratch_elems())*2},
        {uintptr_t(output),size_t(count)*n_out*4}}};
    for (size_t i=0;i<spans.size();++i) {
        const auto [begin,bytes]=spans[i];
        if (!begin || begin%4 || begin>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("invalid precise dense span/alignment");
        for (size_t j=0;j<i;++j)
            if (begin<spans[j].first+spans[j].second && spans[j].first<begin+bytes)
                throw std::invalid_argument("precise dense source/output/phase scratch aliases");
    }
    b.half_width=0; b.half_count=0;
    q8_1_to_f16(b.input_q8.data(),b.input_half.as<uint16_t>(),count*n_in,stream_,b.input_low.as<uint16_t>());
    prefill::precise_native(b.gemm,b.input_half.as<uint16_t>(),b.input_low.as<uint16_t>(),weight.type,
        weight.ptr(),output,count,n_out,n_in,b.scan_activated.as<float>(),
        reinterpret_cast<uint16_t*>(b.scan_activated.as<float>()+elements),elements);
    return true;
}
} // namespace strata::qwen35moe
