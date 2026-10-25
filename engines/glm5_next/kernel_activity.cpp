#include "kernel_activity.hpp"
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <mutex>
#include "verify_activity_timeline.hpp"
#include "surface_activity.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <cupti_activity.h>
#include <cupti_version.h>
namespace strata::glm5_next {
namespace {
// Each selected SDK documents this exact record for both kernel activity kinds.
#if CUPTI_API_VERSION == 130000
using ActivityKernelRecord = CUpti_ActivityKernel10;
constexpr uint32_t activity_record_version = 10;
#elif CUPTI_API_VERSION == 26
using ActivityKernelRecord = CUpti_ActivityKernel9;
constexpr uint32_t activity_record_version = 9;
#else
#error Unsupported CUPTI API version: verify its kernel activity record before enabling this collector
#endif
struct Activity {
    HMODULE library=nullptr;
    decltype(&cuptiActivityRegisterCallbacks) callbacks=nullptr;
    decltype(&cuptiActivityEnable) enable=nullptr;
    decltype(&cuptiActivityDisable) disable=nullptr;
    decltype(&cuptiActivityFlushAll) flush=nullptr;
    decltype(&cuptiActivityGetNextRecord) next=nullptr;
    decltype(&cuptiActivityGetNumDroppedRecords) dropped=nullptr;
    decltype(&cuptiGetTimestamp) timestamp=nullptr;
    decltype(&cuptiGetVersion) version=nullptr;
    std::atomic<uint64_t> kernels{0},lost{0},errors{0};
    std::atomic<uint64_t> buffers{0},buffer_peak{0};
    bool busy=false,initialized=false;
    std::atomic<bool> causal{false};
    std::atomic<bool> surface{false};
    SurfaceActivityReport surface_report{};
    bool collecting=false,prepared=false;
    uint64_t last_marker=0,open_window=0;
    std::mutex timeline_mutex;
    CausalActivityReport report{};
    std::vector<VerifyKernelRecord> records;
};
std::atomic<Activity*> causal_actor{nullptr};
// CUPTI may finish callbacks during CUDA teardown. Keep the state and DLL alive
// until process termination; every activity buffer is freed by its callback.
Activity& state(){static auto* value=new Activity;return *value;}
void check(CUptiResult value,const char* action){
    if(value!=CUPTI_SUCCESS)throw std::runtime_error(std::string("CUPTI ")+action+" failed: "+std::to_string(int(value)));
}
#include "surface_activity_collect.inc"
void CUPTIAPI request_buffer(uint8_t** buffer,size_t* bytes,size_t* maximum){
    auto& s=state();const auto claimed=s.buffers.fetch_add(1)+1;
    if((s.causal.load(std::memory_order_acquire)||s.surface.load(std::memory_order_acquire))&&claimed>8){--s.buffers;++s.errors;*buffer=nullptr;*bytes=0;*maximum=0;return;}
    auto peak=s.buffer_peak.load();while(claimed>peak&&!s.buffer_peak.compare_exchange_weak(peak,claimed)){}
    *bytes=1u<<20;*maximum=0;
    *buffer=static_cast<uint8_t*>(_aligned_malloc(*bytes,8));
    if(!*buffer){*bytes=0;++s.errors;--s.buffers;}
}
void CUPTIAPI complete_buffer(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid){
    auto& s=state();CUpti_Activity* record=nullptr;
    while(buffer&&valid){
        const auto result=s.next(buffer,valid,&record);
        if(result==CUPTI_ERROR_MAX_LIMIT_REACHED)break;
        if(result!=CUPTI_SUCCESS){++s.errors;break;}
        if(record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||record->kind==CUPTI_ACTIVITY_KIND_KERNEL)++s.kernels;
        if(s.surface.load(std::memory_order_acquire)&&
            (record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||record->kind==CUPTI_ACTIVITY_KIND_KERNEL))
            surface_record(s,*reinterpret_cast<const ActivityKernelRecord*>(record));
        if(s.causal.load(std::memory_order_acquire)&&
           (record->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL||record->kind==CUPTI_ACTIVITY_KIND_KERNEL)) {
            // Parse the exact kernel record documented by the selected SDK.
            const auto* kernel=reinterpret_cast<const ActivityKernelRecord*>(record);
            std::lock_guard<std::mutex> lock(s.timeline_mutex);
            if(!s.causal.load(std::memory_order_acquire))continue;
            if(!kernel->start||!kernel->end||kernel->end<kernel->start)++s.report.timestamp_errors;
            else if(!s.report.windows[0].begin||kernel->end<=s.report.windows[0].begin||
                    (s.report.windows[7].end&&kernel->start>=s.report.windows[7].end)){}
            else if(s.records.size()==verify_activity_capacity)++s.report.overflow;
            else s.records.push_back({kernel->start,kernel->end,kernel->graphNodeId,
                kernel->deviceId,kernel->contextId,kernel->streamId,kernel->graphId,verify_kernel_class(kernel->name),
                kernel->name&&std::strstr(kernel->name,"hybrid_weight_kernel")?1u:
                kernel->name&&std::strstr(kernel->name,"moe_weighted_reduction_f32")?2u:0u});
        }
    }
    size_t lost=0;
    if(s.dropped(context,stream,&lost)==CUPTI_SUCCESS)s.lost+=lost;else ++s.errors;
    if(buffer){_aligned_free(buffer);--s.buffers;}
}
void initialize(){
    auto& s=state();if(s.initialized)return;
    const char* path=std::getenv("STRATA_CUPTI_DLL");
    if(!path||!*path)throw std::runtime_error("kernel activity needs an explicit installed CUPTI DLL path");
    s.library=LoadLibraryA(path);if(!s.library)throw std::runtime_error("cannot load installed CUPTI DLL");
#define LOAD(field,name) s.field=reinterpret_cast<decltype(s.field)>(GetProcAddress(s.library,name));if(!s.field)throw std::runtime_error("CUPTI API absent: " name)
    LOAD(callbacks,"cuptiActivityRegisterCallbacks");LOAD(enable,"cuptiActivityEnable");
    LOAD(disable,"cuptiActivityDisable");LOAD(flush,"cuptiActivityFlushAll");
    LOAD(next,"cuptiActivityGetNextRecord");LOAD(dropped,"cuptiActivityGetNumDroppedRecords");
    LOAD(version,"cuptiGetVersion");
#undef LOAD
    uint32_t version=0;check(s.version(&version),"get API version");
    if(version!=CUPTI_API_VERSION)throw std::runtime_error("installed CUPTI API differs from the selected SDK");
    check(s.callbacks(request_buffer,complete_buffer),"register callbacks");
    s.initialized=true;
}
#include "verify_activity_collect.inc"
}
bool kernel_activity_requested(){const char* flag=std::getenv("STRATA_GLM53_KERNEL_ACTIVITY");return flag&&std::string(flag)=="1";}
#include "surface_activity_methods.inc"
KernelActivity::KernelActivity(bool requested){
    if(causal_activity_requested()) {
        auto& s=state();if(s.busy)throw std::runtime_error("only one serial request may own kernel activity");
        {std::lock_guard<std::mutex> lock(s.timeline_mutex);
            s.report={};s.report.requested=true;s.report.record_version=activity_record_version;s.records.clear();}
        s.prepared=false;s.collecting=false;s.last_marker=0;s.open_window=0;
        s.kernels=0;s.lost=0;s.errors=0;s.buffer_peak=s.buffers.load();s.busy=true;s.causal=true;active_=true;causal_actor.store(&s,std::memory_order_release);
        try{initialize();prepare_causal_activity(s);}catch(...){++s.report.api_errors;}
        return;
    }
    if(!requested)return;initialize();auto& s=state();
    if(s.busy)throw std::runtime_error("only one serial request may own kernel activity");
    check(s.flush(1),"flush before start");s.kernels=0;s.lost=0;s.errors=0;
    check(s.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"enable concurrent kernels");
    s.busy=true;active_=true;
}
KernelActivityStats KernelActivity::snapshot(){
    if(!active_)return {};auto& s=state();
    if(s.causal.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(s.timeline_mutex);
        return {s.kernels.load(),s.lost.load(),s.errors.load(),true,s.report};
    }
    check(s.flush(1),"flush completed kernels");
    return {s.kernels.load(),s.lost.load(),s.errors.load(),true};
}
KernelActivityStats KernelActivity::finish(){
    if(!active_)return {};auto& s=state();
    if(s.causal.load(std::memory_order_acquire)) {
        if(s.collecting){causal_api(s.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),s);s.collecting=false;}
        if(s.prepared)causal_api(s.flush(1),s);
        KernelActivityStats result;
        {std::lock_guard<std::mutex> lock(s.timeline_mutex);
            causal_actor.store(nullptr,std::memory_order_release);s.causal=false;
            if(s.open_window)++s.report.marker_errors;
            s.report.dropped=s.lost.load();s.report.buffer_errors=s.errors.load();s.report.callback_buffers_peak=s.buffer_peak.load();
            verify_activity_aggregate(s.report,s.records);
            result={s.kernels.load(),s.lost.load(),s.errors.load(),true,s.report};s.records.clear();}
        s.busy=false;active_=false;return result;
    }
    check(s.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"disable concurrent kernels");
    auto result=snapshot();active_=false;s.busy=false;return result;
}
KernelActivity::~KernelActivity(){if(active_)try{finish();}catch(const std::exception& e){std::fprintf(stderr,"kernel_activity_cleanup %s\n",e.what());}}
void kernel_activity_verify_begin(uint64_t ordinal){if(auto* actor=causal_actor.load(std::memory_order_acquire))causal_verify_begin(*actor,ordinal);}
void kernel_activity_verify_end(uint64_t ordinal){if(auto* actor=causal_actor.load(std::memory_order_acquire))causal_verify_end(*actor,ordinal);}
}
#else
namespace strata::glm5_next {
bool kernel_activity_requested(){return false;}
KernelActivity::KernelActivity(bool requested){if(requested)throw std::runtime_error("installed CUDA kernel activity is required");}
KernelActivity::~KernelActivity()=default;
KernelActivityStats KernelActivity::snapshot(){return {};}
KernelActivityStats KernelActivity::finish(){return {};}
void kernel_activity_verify_begin(uint64_t){}
void kernel_activity_verify_end(uint64_t){}
KernelActivitySurface::KernelActivitySurface(bool enabled){if(enabled)throw std::runtime_error("installed CUDA CUPTI surface is required");}
KernelActivitySurface::~KernelActivitySurface()=default;
void KernelActivitySurface::begin_call(unsigned){}
void KernelActivitySurface::end_call(unsigned){}
SurfaceActivityReport KernelActivitySurface::finish(){return {};}
}
#endif
