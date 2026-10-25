#pragma once
#include "verify_activity.hpp"
#include <algorithm>
#include <cstring>
#include <vector>
namespace strata::glm5_next {
inline VerifyKernel verify_kernel_class(const char* name) {
    if(!name)return VerifyKernel::Other;
    if(std::strstr(name,"cold_gpu_copy_rows4"))return VerifyKernel::Copy4;
    if(std::strstr(name,"cold_gpu_copy_rows"))return VerifyKernel::Copy2;
    if(std::strstr(name,"lookup_kernel"))return VerifyKernel::Lookup;
    if(std::strstr(name,"hybrid_weight_kernel")||std::strstr(name,"moe_weighted_reduction_f32"))return VerifyKernel::Join;
    return VerifyKernel::Other;
}
inline bool verify_same_stream(const VerifyKernelRecord& a,const VerifyKernelRecord& b) {
    return a.device==b.device&&a.context==b.context&&a.stream==b.stream;
}
inline bool verify_record_less(const VerifyKernelRecord& a,const VerifyKernelRecord& b) {
    if(a.start!=b.start)return a.start<b.start;
    if(a.end!=b.end)return a.end<b.end;
    return uint32_t(a.kind)<uint32_t(b.kind);
}
inline uint64_t verify_union_add(uint64_t begin,uint64_t end,uint64_t& frontier) {
    const uint64_t contribution=end>std::max(begin,frontier)?end-std::max(begin,frontier):0;
    frontier=std::max(frontier,end);return contribution;
}
// Postprocess only after activity disable+flush, while callbacks cannot append.
// Retain no raw trace in the report. All differences use one CUPTI time domain.
inline void verify_activity_aggregate(CausalActivityReport& report,std::vector<VerifyKernelRecord>& records) {
    std::sort(records.begin(),records.end(),verify_record_less);report.records=records.size();
    bool complete=report.windows_started==8&&report.windows_completed==8;
    report.insufficient=report.windows_started<8||report.windows_completed<8;
    for(auto& window:report.windows) {
        if(!window.begin||window.end<=window.begin){complete=false;continue;}
        const auto first=records.begin();
        const auto last=std::lower_bound(first,records.end(),window.end,
            [](const auto& record,uint64_t t){return record.start<t;});
        uint64_t union_end=0;const VerifyKernelRecord* stream=nullptr;
        for(auto it=first;it!=last;++it) {
            if(it->end<=window.begin)continue;
            ++window.kernels;const auto clipped_start=std::max(it->start,window.begin),clipped_end=std::min(it->end,window.end);
            window.kernel_sum_ns+=clipped_end-clipped_start;
            window.kernel_union_ns+=verify_union_add(clipped_start,clipped_end,union_end);
            if(it->kind==VerifyKernel::Copy2||it->kind==VerifyKernel::Copy4) {
                if(!stream)stream=&*it;
                if(!verify_same_stream(*stream,*it))++window.identity_errors;
                ++window.copies;
            }
        }
        if(!stream||window.copies!=39){++window.order_errors;complete=false;continue;}
        uint64_t compute_frontier=0,outside_frontier=0;
        for(auto it=first;it!=last;++it) {
            if(it->end<=window.begin)continue;
            const auto clipped_start=std::max(it->start,window.begin),clipped_end=std::min(it->end,window.end);
            if(!verify_same_stream(*stream,*it)){++window.outside_stream_kernels;
                window.outside_stream_kernel_union_ns+=verify_union_add(clipped_start,clipped_end,outside_frontier);}
            else window.compute_stream_kernel_union_ns+=verify_union_add(clipped_start,clipped_end,compute_frontier);
        }
        size_t layer_index=0;
        for(auto copy=first;copy!=last;++copy) {
            if(copy->end<=window.begin)continue;
            if(copy->kind!=VerifyKernel::Copy2&&copy->kind!=VerifyKernel::Copy4)continue;
            if(layer_index>=39){++window.order_errors;break;}
            auto& block=window.blocks[layer_index];int layer=3+int(layer_index);if(layer>=11)layer+=2;
            block.layer=layer;++layer_index;block.copy_capacity=copy->kind==VerifyKernel::Copy4?4:2;
            block.device=copy->device;block.context=copy->context;block.stream=copy->stream;block.graph=copy->graph;
            const VerifyKernelRecord* lookup=nullptr;bool adjacency=true;
            for(auto it=copy;it!=first;) {
                --it;if(!verify_same_stream(*copy,*it))continue;
                if(it->kind==VerifyKernel::Lookup)lookup=&*it;
                else adjacency=false;
                break;
            }
            const VerifyKernelRecord* join=nullptr;uint64_t branch_frontier=0;
            for(auto it=copy+1;it!=last;++it) {
                if(!verify_same_stream(*copy,*it))continue;
                if(it->kind==VerifyKernel::Join){join=&*it;break;}
                if(it->kind==VerifyKernel::Lookup||it->kind==VerifyKernel::Copy2||it->kind==VerifyKernel::Copy4){adjacency=false;break;}
                if(it->start<copy->end||it->end>window.end){adjacency=false;break;}
                ++block.kernel_count;block.branch_kernel_sum_ns+=it->end-it->start;
                block.branch_kernel_union_ns+=verify_union_add(it->start,it->end,branch_frontier);
            }
            const bool identity=lookup&&join&&lookup->graph&&lookup->node&&copy->graph&&copy->node&&join->graph&&join->node&&
                lookup->graph==copy->graph&&copy->graph==join->graph&&(join->padding==1||join->padding==2);
            if(!identity){++window.identity_errors;continue;}
            const bool order=adjacency&&lookup->start>=window.begin&&lookup->end<=copy->start&&copy->end<=join->start&&join->end<=window.end;
            if(!order||!block.kernel_count||block.branch_kernel_union_ns>join->start-copy->end){++window.order_errors;continue;}
            block.lookup_ns=lookup->end-lookup->start;block.lookup_to_copy_gap_ns=copy->start-lookup->end;
            block.copy_ns=copy->end-copy->start;block.branch_ns=join->start-copy->end;
            block.join_ns=join->end-join->start;block.total_ns=join->end-lookup->start;block.join_variant=join->padding;
            block.passed=true;++window.paired;
        }
        window.passed=window.copies==39&&window.paired==39&&!window.order_errors&&!window.identity_errors;
        complete&=window.passed;
    }
    for(const auto& record:records){bool included=false;
        for(const auto& window:report.windows)included|=window.begin&&window.end>window.begin&&record.start<window.end&&record.end>window.begin;
        if(!included)++report.records_outside_windows;}
    report.passed=complete&&!report.overflow&&!report.timestamp_errors&&!report.marker_errors&&!report.api_errors&&!report.dropped&&!report.buffer_errors;
}
}
