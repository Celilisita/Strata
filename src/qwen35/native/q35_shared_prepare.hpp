#pragma once
#include "pp_prefetch.hpp"
#include "strata/qwen35/native/nvfp4_prefill_gdn.hpp"
#include <cstdint>
namespace strata::qwen35::native {
// This ticket is invocation-scoped, never a cache of encoded activations.
struct Q35SharedPrepareLease {
    const void* owner=nullptr;
    const void* producer=nullptr;
    const float* input=nullptr;
    void* arena=nullptr;
    size_t bytes=0;
    cudaStream_t stream=nullptr;
    uint64_t epoch=0;
};
// Parent layouts, codec kernels, weights and arenas stay unchanged.
class Q35SharedNvfp4 final:public PrefetchNvfp4 {
    bool shared_=false;
    bool geometry_=false;
    cudaStream_t stream_=nullptr;
    uint64_t epoch_=0;
    unsigned long long groups_=0;
public:
    Q35SharedNvfp4(size_t weights,int width,int tokens):PrefetchNvfp4(weights,width,tokens),
        geometry_(width>=5120&&tokens>=3){}
    // Caller releases/drains its graphs before policy changes, then recaptures.
    void shared_prepare(bool,cudaStream_t);
    bool shared_prepare()const{return shared_;}
    uint64_t shared_epoch()const{return epoch_;}
    unsigned long long shared_groups()const{return groups_;}
    unsigned long long calls()const override{return OriginalNvfp4Gemm::calls()+groups_*4;}
    Q35SharedPrepareLease lease(const void*,const float*,void*,size_t,cudaStream_t)const;
    bool group(const Q35SharedPrepareLease&,const void*,const GdnPrefillProjection*,
               const float*,int,cudaStream_t);
};
bool try_q35_shared_prepare(DenseGemm*,const void* producer,const GdnPrefillProjection*,
    const float*,void* arena,size_t arena_bytes,int tokens,cudaStream_t);
}
