#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include "strata/gemma4/native/sm120_projection_plan.hpp"
namespace strata::gemma4::native {
// Model owns one bounded FP32 attention service, shared by all sixty layers.
class Sm120AttentionService {
    primitive::DeviceStorage storage_;
    cudaStream_t stream_;
    cublasHandle_t handle_=nullptr;
    AttentionWorkspace workspace_{};
    int reserved_,batch_;
    bool pp_tc_=false,global_t1_merge_=false,local270_=false;
public:
    Sm120AttentionService(int reserved,int batch,cudaStream_t);
    ~Sm120AttentionService();
    void enqueue(const float*,float*,CacheView,const int*,int tokens,int past,cudaStream_t);
    std::size_t bytes()const{return storage_.bytes();}
};
}
