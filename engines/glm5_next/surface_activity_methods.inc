// Included in namespace strata::glm5_next, outside the anonymous namespace.
KernelActivitySurface::KernelActivitySurface(bool enabled){
    if(!enabled)return;auto& s=state();if(s.busy)throw std::runtime_error("CUPTI surface requires exclusive diagnostic ownership");
    {std::lock_guard<std::mutex> lock(s.timeline_mutex);s.surface_report={};s.surface_report.requested=true;
        s.surface_report.records.reserve(surface_activity_capacity);}
    s.surface=true;s.busy=true;active_=true;s.prepared=false;s.collecting=false;s.open_window=0;s.kernels=0;s.lost=0;s.errors=0;
    try{initialize();s.timestamp=reinterpret_cast<decltype(s.timestamp)>(GetProcAddress(s.library,"cuptiGetTimestamp"));
        s.version=reinterpret_cast<decltype(s.version)>(GetProcAddress(s.library,"cuptiGetVersion"));
        if(!s.timestamp||!s.version)throw std::runtime_error("CUPTI surface timestamp/version absent");
        uint32_t version=0;check(s.version(&version),"surface API version");s.surface_report.api_version=version;
        if(version!=CUPTI_API_VERSION)throw std::runtime_error("CUPTI surface API mismatch");
        check(s.flush(1),"surface prior flush");s.lost=0;s.errors=0;
        check(s.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"surface enable");s.collecting=true;s.prepared=true;
    }catch(...){++s.surface_report.api_errors;}
}
void KernelActivitySurface::begin_call(unsigned call){
    if(!active_)return;auto& s=state();
    if(call<1||call>3||s.open_window){++s.surface_report.marker_errors;return;}
    uint64_t timestamp=0;if(!s.prepared||s.timestamp(&timestamp)!=CUPTI_SUCCESS){++s.surface_report.api_errors;return;}
    std::lock_guard<std::mutex> lock(s.timeline_mutex);auto& c=s.surface_report.calls[call-1];
    if(c.begin)++s.surface_report.marker_errors;c.begin=timestamp;s.open_window=call;
}
void KernelActivitySurface::end_call(unsigned call){
    if(!active_)return;auto& s=state();if(call<1||call>3||s.open_window!=call){++s.surface_report.marker_errors;return;}
    uint64_t timestamp=0;if(!s.prepared||s.timestamp(&timestamp)!=CUPTI_SUCCESS){++s.surface_report.api_errors;return;}
    std::lock_guard<std::mutex> lock(s.timeline_mutex);s.surface_report.calls[call-1].end=timestamp;s.open_window=0;
}
SurfaceActivityReport KernelActivitySurface::finish(){
    if(!active_)return {};auto& s=state();
    if(s.collecting){if(s.disable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)!=CUPTI_SUCCESS)++s.surface_report.api_errors;s.collecting=false;}
    if(s.prepared&&s.flush(1)!=CUPTI_SUCCESS)++s.surface_report.api_errors;
    SurfaceActivityReport out;
    {std::lock_guard<std::mutex> lock(s.timeline_mutex);s.surface=false;
        if(s.open_window)++s.surface_report.marker_errors;s.surface_report.dropped=s.lost.load();s.surface_report.buffer_errors=s.errors.load();
        surface_aggregate(s.surface_report);out=std::move(s.surface_report);}
    s.busy=false;active_=false;return out;
}
KernelActivitySurface::~KernelActivitySurface(){if(active_)try{finish();}catch(...){}}
