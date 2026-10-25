#pragma once
#include <algorithm>
#include <stdexcept>
#include <vector>
#include "../deepseek_v4/cpu_topology_win.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace strata::glm5_next {
struct CpuPlacement {
    bool enabled = false;
    deepseek_v4::CpuCcd ccd;
    uint64_t process_mask = 0;
    std::vector<deepseek_v4::CpuCoreBinding> logical;
    std::vector<std::vector<deepseek_v4::CpuCoreBinding>> pools;
};

inline bool single_ccd_requested() {
    const char* value = std::getenv("STRATA_GLM53_CPU_SINGLE_CCD");
    if (value && std::strcmp(value, "0") && std::strcmp(value, "1"))
        throw std::runtime_error("GLM single CCD flag must be 0 or 1");
    return value && !std::strcmp(value, "1");
}

inline const CpuPlacement& cpu_placement() {
    static const CpuPlacement placement = [] {
        CpuPlacement result;
        if (!single_ccd_requested()) return result;
#ifdef _WIN32
        DWORD_PTR allowed = 0, system = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system))
            throw std::runtime_error("Cannot read GLM process affinity");
        // L3 domains include both SMT siblings. The whole process stays in
        // one complete domain, including GPU submission and CPU helpers.
        for (const auto& ccd : deepseek_v4::cpu_ccd_topology()) {
            if (ccd.group != 0 || ccd.cores.empty()) continue;
            const uint64_t mask = ccd.mask;
            if ((mask & uint64_t(allowed)) != mask) continue;
            result.enabled = true;
            result.ccd = ccd;
            result.process_mask = mask;
            for (int logical=0;logical<64;++logical)if(mask&(1ull<<logical))
                result.logical.push_back({ccd.group,uint8_t(logical),1ull<<logical});
            // Smaller explicit pools retain one thread per physical core
            // first. Full-CCD pools include every SMT sibling exactly once.
            std::vector<deepseek_v4::CpuCoreBinding> order=ccd.cores;
            for(const auto& binding:result.logical) {
                bool primary=false;for(const auto& core:ccd.cores)primary|=core.mask==binding.mask;
                if(!primary)order.push_back(binding);
            }
            result.pools.resize(order.size()+1);
            for(size_t count=1;count<=order.size();++count) {
                auto& pool=result.pools[count];pool.assign(order.begin(),order.begin()+count);
                std::sort(pool.begin(),pool.end(),[](const auto& a,const auto& b){return a.logical<b.logical;});
            }
            return result;
        }
        throw std::runtime_error("No complete single CCD fits the GLM process affinity");
#else
        throw std::runtime_error("GLM single CCD placement currently requires Windows topology");
#endif
    }();
    return placement;
}

inline void configure_cpu_process() {
    const auto& placement = cpu_placement();
    if (!placement.enabled) return;
    static std::once_flag configured;
    std::call_once(configured, [&] {
#ifdef _WIN32
        if (!SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(placement.process_mask)))
            throw std::runtime_error("Cannot set GLM single CCD process affinity");
        DWORD_PTR actual = 0, system = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &actual, &system) ||
            uint64_t(actual) != placement.process_mask)
            throw std::runtime_error("GLM single CCD process affinity verification failed");
        const auto count = std::to_string(placement.logical.size());
        _putenv_s("OMP_NUM_THREADS", count.c_str());
        _putenv_s("OMP_THREAD_LIMIT", count.c_str());
        _putenv_s("OMP_DYNAMIC", "FALSE");
        std::fprintf(stderr, "GLM CPU: one CCD, L3=%u MiB, physical cores=%zu, logical threads=%zu, mask=0x%llx\n",
            placement.ccd.cache_bytes >> 20, placement.ccd.cores.size(),
            placement.logical.size(),static_cast<unsigned long long>(placement.process_mask));
#endif
    });
}

inline void validate_cpu_threads(int threads) {
    const auto& placement = cpu_placement();
    if (placement.enabled && (threads < 1 || size_t(threads) > placement.logical.size()))
        throw std::runtime_error("GLM CPU thread count exceeds one CCD's logical processors");
}

inline void configure_cpu_pool(ggml_threadpool_params& options) {
    configure_cpu_process();
    validate_cpu_threads(options.n_threads);
    const auto& placement = cpu_placement();
    if (!placement.enabled) return;
    std::memset(options.cpumask, 0, sizeof(options.cpumask));
    for (int index = 0; index < options.n_threads; ++index)
        options.cpumask[placement.pools.at(size_t(options.n_threads)).at(size_t(index)).logical] = true;
    options.strict_cpu = true;
    options.poll = 0; // Park between real CPU jobs; GPU submission shares this CCD.
}
inline deepseek_v4::CpuCoreBinding save_cpu_affinity() {
    deepseek_v4::CpuCoreBinding result;
#ifdef _WIN32
    if(cpu_placement().enabled) {
        GROUP_AFFINITY current{};
        if(!GetThreadGroupAffinity(GetCurrentThread(),&current))throw std::runtime_error("Cannot save CPU pool creator affinity");
        result.group=current.Group;result.mask=uint64_t(current.Mask);
    }
#endif
    return result;
}
inline void restore_cpu_affinity(const deepseek_v4::CpuCoreBinding& saved) {
    if(saved.mask)deepseek_v4::cpu_bind_current(saved);
}
inline bool cpu_backend_openmp() {
    static const bool enabled=[] {
        const auto reg=ggml_backend_cpu_reg();
        const auto get=reinterpret_cast<ggml_backend_get_features_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_get_features"));
        if(!get)throw std::runtime_error("CPU backend feature contract unavailable");
        for(const auto* item=get(reg);item&&item->name;++item)
            if(!std::strcmp(item->name,"OPENMP"))return true;
        return false;
    }();
    return enabled;
}
inline size_t cpu_worker_core(int worker,int threads) {
    // GGML native workers1..N-1 take the first cores; its caller0 takes
    // the last. OpenMP assigns masks in worker-index order instead.
    return size_t(cpu_backend_openmp()?worker:worker?worker-1:threads-1);
}
template<class F> inline void with_cpu_caller(int threads,F&& compute) {
    const auto saved=save_cpu_affinity();
    const auto& placement=cpu_placement();
    const auto target=placement.enabled?placement.pools.at(size_t(threads)).at(cpu_worker_core(0,threads)):deepseek_v4::CpuCoreBinding{};
    const bool changed=placement.enabled && (saved.group!=target.group || saved.mask!=target.mask);
    if(changed)deepseek_v4::cpu_bind_current(target);
    try { compute(); }
    catch(...) { if(changed)restore_cpu_affinity(saved);throw; }
    if(changed)restore_cpu_affinity(saved);
}
}
