// Included inside the existing Windows collector's anonymous namespace.
bool causal_activity_requested(){
    const char* value=std::getenv("STRATA_GLM53_VERIFY_ACTIVITY");
    if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))throw std::runtime_error("verify activity must be0or1");
    return value&&!std::strcmp(value,"1");
}
bool causal_api(CUptiResult result,Activity& s){
    if(result==CUPTI_SUCCESS)return true;
    std::lock_guard<std::mutex> lock(s.timeline_mutex);++s.report.api_errors;return false;
}
void prepare_causal_activity(Activity& s){
    // These extra API lookups are diagnostic-only and outside timed windows.
    s.timestamp=reinterpret_cast<decltype(s.timestamp)>(GetProcAddress(s.library,"cuptiGetTimestamp"));
    s.version=reinterpret_cast<decltype(s.version)>(GetProcAddress(s.library,"cuptiGetVersion"));
    if(!s.timestamp||!s.version){++s.report.api_errors;return;}
    uint32_t version=0;if(!causal_api(s.version(&version),s))return;
    s.report.cupti_api_version=version;
    if(version!=CUPTI_API_VERSION){++s.report.api_errors;return;}
    {std::lock_guard<std::mutex> lock(s.timeline_mutex);s.records.reserve(verify_activity_capacity);}
    if(!causal_api(s.flush(1),s))return;
    s.kernels=0;s.lost=0;s.errors=0;
    {std::lock_guard<std::mutex> lock(s.timeline_mutex);s.records.clear();}
    // Enable before PP, as the original request-wide collector does. Avoid a
    // first-enable/graph instrumentation transition inside measured verify17.
    if(!causal_api(s.enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),s))return;
    s.collecting=true;s.prepared=true;
}
void causal_verify_begin(Activity& s,uint64_t ordinal){
    if(!s.causal.load(std::memory_order_acquire))return;
    if(ordinal!=s.last_marker+1||s.open_window){++s.report.marker_errors;return;}
    s.last_marker=ordinal;
    if(ordinal<verify_activity_first||ordinal>verify_activity_last||!s.prepared)return;
    if(!s.collecting){++s.report.marker_errors;return;}
    uint64_t timestamp=0;if(!causal_api(s.timestamp(&timestamp),s))return;
    std::lock_guard<std::mutex> lock(s.timeline_mutex);
    auto& window=s.report.windows[size_t(ordinal-verify_activity_first)];
    window.ordinal=ordinal;window.begin=timestamp;++s.report.windows_started;s.open_window=ordinal;
}
void causal_verify_end(Activity& s,uint64_t ordinal){
    if(!s.causal.load(std::memory_order_acquire)||ordinal<verify_activity_first||ordinal>verify_activity_last||!s.prepared)return;
    if(s.open_window!=ordinal){++s.report.marker_errors;return;}
    uint64_t timestamp=0;const bool obtained=causal_api(s.timestamp(&timestamp),s);
    {std::lock_guard<std::mutex> lock(s.timeline_mutex);
        auto& window=s.report.windows[size_t(ordinal-verify_activity_first)];window.end=timestamp;
        if(obtained&&timestamp>window.begin)++s.report.windows_completed;else ++s.report.timestamp_errors;
        s.open_window=0;}
}
