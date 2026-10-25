// Included inside the existing Windows CUPTI collector anonymous namespace.
void surface_record(Activity& s,const ActivityKernelRecord& kernel){
    std::lock_guard<std::mutex> lock(s.timeline_mutex);
    if(!s.surface.load(std::memory_order_acquire))return;
    auto& report=s.surface_report;
    if(!kernel.start||!kernel.end||kernel.end<kernel.start){++report.timestamp_errors;return;}
    bool selected=false;
    for(const auto& c:report.calls)selected|=c.begin&&kernel.end>c.begin&&(!c.end||kernel.start<c.end);
    if(!selected)return;
    if(report.records.size()==surface_activity_capacity){++report.overflow;return;}
    SurfaceKernelRecord raw;raw.kernel={kernel.start,kernel.end,kernel.graphNodeId,kernel.deviceId,kernel.contextId,
        kernel.streamId,kernel.graphId,verify_kernel_class(kernel.name),kernel.name&&std::strstr(kernel.name,"hybrid_weight_kernel")?1u:
        kernel.name&&std::strstr(kernel.name,"moe_weighted_reduction_f32")?2u:0u};
    size_t bytes=0;if(kernel.name){for(;bytes+1<raw.name.size()&&kernel.name[bytes];++bytes)raw.name[bytes]=kernel.name[bytes];
        raw.name_truncated=kernel.name[bytes]!=0;}
    report.records.push_back(raw);
}
void surface_aggregate(SurfaceActivityReport& report){
    auto& records=report.records;std::sort(records.begin(),records.end(),[](const auto& a,const auto& b){return verify_record_less(a.kernel,b.kernel);});
    bool complete=records.size()>0;
    for(size_t call=0;call<report.calls.size();++call){auto& c=report.calls[call];
        if(!c.begin||c.end<=c.begin){complete=false;continue;}
        for(const auto& raw:records){const auto& r=raw.kernel;if(r.start<c.begin||r.end>c.end)continue;
            ++c.kernels;c.lookups+=r.kind==VerifyKernel::Lookup;c.copies+=r.kind==VerifyKernel::Copy2||r.kind==VerifyKernel::Copy4;c.joins+=r.kind==VerifyKernel::Join;
            if(r.graph&&r.node){++c.graph_nodes;c.graph_lookups+=r.kind==VerifyKernel::Lookup;
                c.graph_copies+=r.kind==VerifyKernel::Copy2||r.kind==VerifyKernel::Copy4;c.graph_joins+=r.kind==VerifyKernel::Join;}
            else ++c.zero_graph_nodes;
            for(const auto& other:records){const auto& q=other.kernel;if(&q==&r)break;
                if(q.start>=c.begin&&q.end<=c.end&&verify_same_stream(q,r)&&r.start<q.end){
                    ++c.overlap_pairs;c.overlap_ns+=std::min(r.end,q.end)-r.start;}}
            if(r.kind!=VerifyKernel::Copy2&&r.kind!=VerifyKernel::Copy4)continue;
            const VerifyKernelRecord *lookup=nullptr,*join=nullptr,*prior=nullptr;
            for(const auto& candidate:records){const auto& q=candidate.kernel;
                if(q.start<c.begin||q.start>=r.start||!verify_same_stream(r,q))continue;
                prior=&q;if(q.kind==VerifyKernel::Lookup)lookup=&q;}
            if(!lookup){++report.missing_lookup;for(const auto& candidate:records){const auto& q=candidate.kernel;
                if(q.start>=c.begin&&q.start<r.start&&q.kind==VerifyKernel::Lookup&&!verify_same_stream(q,r)){++report.stream_mismatch;break;}}}
            if(!prior||prior->kind!=VerifyKernel::Lookup)++report.nonadjacent_lookup;
            for(const auto& candidate:records){const auto& q=candidate.kernel;
                if(q.start<=r.start||q.end>c.end||!verify_same_stream(r,q))continue;
                if(q.kind==VerifyKernel::Join){join=&q;break;}
                if(q.kind==VerifyKernel::Other&&q.start<r.end)++report.early_other_overlap;
            }
            if(!join){++report.missing_join;for(const auto& candidate:records){const auto& q=candidate.kernel;
                if(q.start>r.start&&q.end<=c.end&&q.kind==VerifyKernel::Join&&!verify_same_stream(q,r)){++report.stream_mismatch;break;}}}
            if(lookup&&join){if(!lookup->node||!r.node||!join->node||!lookup->graph||!r.graph||!join->graph)++report.zero_node;
                if(lookup->graph!=r.graph||r.graph!=join->graph)++report.graph_mismatch;
            }
        }
        complete&=c.lookups==1&&c.copies==1&&c.joins==1;
        // Eager is retained for contrast, but graph records are mandatory for
        // capture/replay. Do not treat direct-only timestamps as graph proof.
        if(call)complete&=c.graph_lookups==1&&c.graph_copies==1&&c.graph_joins==1;
    }
    report.passed=complete&&!report.overflow&&!report.timestamp_errors&&!report.api_errors&&!report.marker_errors&&!report.dropped&&!report.buffer_errors;
}
