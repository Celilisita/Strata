#pragma once
#include <cstdint>
namespace strata::qwen35::native {
struct TinyParameterPairRequest {
    bool enabled=false;
    uint64_t submissions=0;
};
inline TinyParameterPairRequest &tiny_parameter_pair_request(){
    static thread_local TinyParameterPairRequest state;return state;
}
// PREPARE only, after prior request graphs are released and before eager/capture.
// The caller owns environment parsing and hardware admission. No hot getenv.
inline void configure_tiny_parameter_pair_request(bool enabled){
    auto &state=tiny_parameter_pair_request();state.enabled=enabled;state.submissions=0;
}
// Current-request host eager/capture submissions, not GPU graph replays.
inline uint64_t tiny_parameter_pair_submissions(){return tiny_parameter_pair_request().submissions;}
}
