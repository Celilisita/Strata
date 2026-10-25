#pragma once
#include <cstdint>
#include "verify_activity.hpp"
namespace strata::glm5_next {
struct KernelActivityStats {uint64_t kernels=0,dropped=0,errors=0;bool enabled=false;CausalActivityReport causal{};};
class KernelActivity {
    bool active_=false;
public:
    explicit KernelActivity(bool requested=false);
    ~KernelActivity();
    KernelActivity(const KernelActivity&)=delete;
    KernelActivity& operator=(const KernelActivity&)=delete;
    bool active()const{return active_;}
    KernelActivityStats snapshot();
    KernelActivityStats finish();
};
bool kernel_activity_requested();
}
