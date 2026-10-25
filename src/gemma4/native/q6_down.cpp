#include "strata/runtime/capacity_storage.hpp"
#include "strata/gemma4/native/q6_down.hpp"
#include "strata/gemma4/native/q6_aligned.hpp"
namespace strata::gemma4::native {
primitive::Matrix replace_owned_q6_down(primitive::Matrix raw,
    strata::runtime::CapacityStorage& owner,cudaStream_t stream){
    if(!stream||raw.quant!=primitive::Quant::Q6K||raw.layout!=primitive::BlockLayout::Gguf||
       raw.rows!=5376||raw.columns!=21504)
        throw std::invalid_argument("Gemma unique cold Q6 Down replacement");
    const size_t bytes=aligned_q6_bytes(raw);
    void* destination=owner.allocate(bytes);
    const auto packed=pack_aligned_q6(raw,destination,bytes,stream);
    primitive::check_cuda(cudaStreamSynchronize(stream),"Gemma cold Q6 Down pack lifetime");
    // No stale raw pointer escapes. Owner retains both allocations on pack/join
    // failure; release is attempted only after the destination is complete.
    owner.release(const_cast<void*>(raw.data));
    return {packed.data,raw.quant,raw.rows,raw.columns,primitive::BlockLayout::Q6Aligned224};
}
}
