#include "original_quant_projection.hpp"
#include "mmq.cuh"
#include <set>
#include <stdexcept>
namespace strata::gemma4::native {
std::size_t original_projection_pool_bound_bytes(){
    const auto& device=ggml_cuda_info().devices[ggml_cuda_get_device()];
    if(device.nsm<1)throw std::runtime_error("Gemma original MMQ device SM count");
    std::set<std::size_t> sizes;
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q6_K})for(int j=8;j<=128;j+=8){
#ifdef STRATA_DEEPSEEK_V4_GGML_API
        const auto config=ggml_cuda_mmq_get_config(type,j,false,device.cc,GGML_PREC_Q8);
#else
        const auto config=ggml_cuda_mmq_get_config(type,j,false,device.cc);
#endif
        if(config.type==GGML_TYPE_COUNT||!config.stream_k||mmq_get_nbytes_shared(config,device.cc)>device.smpbo)continue;
        if(config.I<1||config.J<1)throw std::runtime_error("Gemma original MMQ tile geometry");
        // Selected mmq.cuh allocates only when tile_count%grid !=0. The
        // grid is tile_count (then no allocation) or nsm (then this size).
        sizes.insert(std::size_t(device.nsm)*config.I*config.J*sizeof(float));
    }
    // Strata's CachingPool first-fits any idle buffer >= requested bytes.
    // One ordered caller can retain at most one buffer per increasing maximum;
    // summing every distinct legal request size bounds every possible history.
    std::size_t bound=0;for(const auto n:sizes)bound+=n;return bound;
}
}
