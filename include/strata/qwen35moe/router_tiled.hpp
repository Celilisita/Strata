#pragma once
#include <cstddef>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe {
inline bool router_tf32x3_policy(const char* raw) {
    if(!raw || !std::strcmp(raw,"0"))return false;
    if(!std::strcmp(raw,"1"))return true;
    throw std::invalid_argument("router TF32x3 requires0 or1");
}
// F32 router with canonical lane/K/reduction order, four independent tokens.
void route_project_tiled(const float* rows, const float* input, float* logits,
    int width, int outputs, int tokens, void* stream);
// Explicit controls for real-weight operator gates; no new workspace or weight pack.
void route_project_tiled_f32(const float* rows,const float* input,float* logits,
    int width,int outputs,int tokens,void* stream);
void route_project_tf32x3(const float* rows,const float* input,float* logits,
    int width,int outputs,int tokens,void* stream);
struct RouterProjectResources {
    int registers=0,threads=0,active_blocks=0;size_t shared_bytes=0,local_bytes=0;
};
RouterProjectResources route_project_resources(bool tensor);
}
