#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::kernels {
struct NativeGdnConvTime32Resources {
    int major=0,minor=0,sms=0,conv_regs=0,conv_local=0,conv_shared=0,conv_blocks=0,conv_binary=0;
    int commit_regs=0,commit_local=0,commit_shared=0,commit_blocks=0,commit_binary=0;
    bool eligible=false;
};
// The Runtime caller owns phase/session/MTP eligibility. Small operators may
// call the same low-level kernel for T=3..8192; that is not Runtime PP coverage.
inline bool native_gdn_conv_time32_selected(bool prefill,bool single,bool mtp,int channels,int tokens,bool prefix) {
    const char* p=std::getenv("STRATA_QWEN35_PP_CONV_TIME32");
    if(!p||std::strcmp(p,"0")==0)return false;
    if(std::strcmp(p,"1")!=0)throw std::invalid_argument("PP_CONV_TIME32 must be0 or1");
    return prefill&&single&&!mtp&&!prefix&&channels==8192&&tokens>=1024&&tokens<=8192;
}
NativeGdnConvTime32Resources native_gdn_conv_time32_resources();
// F32 token-major input/output, weights[8192,4], history[8192,3] oldest first.
// No allocation/synchronization. Return false before writes when unqualified.
// The separate commit kernel follows convolution on the same caller stream.
bool native_gdn_conv_time32(float* history,const float* input,const float* weights,
                           float* output,int channels,int tokens,void* stream);
}
