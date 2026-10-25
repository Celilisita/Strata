#pragma once
#include <vector>
#include "runtime.hpp"
#include "serve_metrics.hpp"
#include "kernel_activity.hpp"
#include "../deepseek_v4/serve_request.hpp"
#include <atomic>
#include <chrono>
#include <memory>
namespace strata::glm5_next {
int serve_mtp(deepseek_v4::Runtime&,const deepseek_v4::ServeRequest&,
    const std::shared_ptr<std::atomic<bool>>&,std::chrono::steady_clock::time_point,
    const deepseek_v4::ServeMetrics&,KernelActivity&);
}
