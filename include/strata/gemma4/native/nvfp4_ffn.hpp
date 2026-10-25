#pragma once
#include "strata/gemma4/native/nvfp4_projection.hpp"
namespace strata::gemma4::native {
struct Nvfp4FfnWeights {
    qwen35::native::Matrix gate,up,down;
    const float *pre_norm=nullptr,*post_norm=nullptr,*layer_scale=nullptr;
};
// Gemma math owner; the borrowed projection service owns every NVFP4 GEMM.
class Nvfp4Ffn {
    Nvfp4ProjectionService& projections_;
    qwen35::native::DeviceStorage storage_;
    float *normalized_,*gate_,*up_,*activation_,*down_,*post_;
    int maximum_;
    float epsilon_;
public:
    Nvfp4Ffn(Nvfp4ProjectionService&,int maximum_batch,float epsilon);
    ~Nvfp4Ffn();
    void run(const Nvfp4FfnWeights&,const float*,float*,int,cudaStream_t);
    // Exclusive between Main forward calls, on the same ordered stream.
    // A following GELU overwrites this full slab before Down consumes it.
    void* mtp_decode_workspace()const{return activation_;}
    std::size_t mtp_decode_workspace_bytes()const{return std::size_t(maximum_)*21504*4;}
    std::size_t bytes()const{return storage_.bytes();}
};
}
