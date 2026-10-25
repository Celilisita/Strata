#pragma once
#include "ggml-backend.h"
#include "expert_source_pin_backend.hpp"
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace strata::deepseek_v4 {
struct SourcePinStreamCall {const char* operation;uint32_t primary,pending;};
struct SourcePinStreamProbe {
    uintptr_t backend,main_stream,copy_stream,current_context,main_context,copy_context;
    int32_t backend_device,physical_device,current_device,stream_index;
    uint32_t reason,count;
    SourcePinStreamCall calls[7];
};
using SourcePinStreamProbeFn=bool(*)(ggml_backend_t,SourcePinStreamProbe*);
bool source_pin_stream_probe(ggml_backend_t,SourcePinStreamProbe*);

inline bool fence_source_pin_streams(ggml_backend_t backend,const char* point,uint64_t bank,SourcePinApiResult& result) {
    result={};
    const auto device=ggml_backend_get_device(backend);
    const auto reg=ggml_backend_dev_backend_reg(device);
    const auto probe=reinterpret_cast<SourcePinStreamProbeFn>(
        ggml_backend_reg_get_proc_address(reg,"strata_source_pin_stream_probe"));
    if(!probe)throw std::runtime_error("source pin actual-stream diagnostic proc unavailable");
    SourcePinStreamProbe state{};const bool good=probe(backend,&state);
    std::fprintf(stderr,"strata_source_pin_stream_probe point=%s bank=%llu success=%d reason=%u backend=%p backend_device=%d physical_device=%d current_device=%d stream_index=%d main=%p copy=%p context=%p main_context=%p copy_context=%p calls=[",
        point,(unsigned long long)bank,int(good),state.reason,reinterpret_cast<void*>(state.backend),state.backend_device,
        state.physical_device,state.current_device,state.stream_index,reinterpret_cast<void*>(state.main_stream),
        reinterpret_cast<void*>(state.copy_stream),reinterpret_cast<void*>(state.current_context),
        reinterpret_cast<void*>(state.main_context),reinterpret_cast<void*>(state.copy_context));
    for(uint32_t i=0;i<state.count&&i<7;++i)std::fprintf(stderr,"%s%s:%u/%u",i?",":"",
        state.calls[i].operation,state.calls[i].primary,state.calls[i].pending);
    std::fprintf(stderr,"]\n");std::fflush(stderr);
    if(good)return true;
    // Only a fresh resource failure of one of these actual runtime fences may
    // enter startup allocation rollback. Identity/query/unknown pending errors
    // are never consumed, including errors left by earlier work.
    result.stage="backend_stream_identity";result.primary=state.reason?state.reason:1;
    const char* expected[]{"cudaGetDevice","cuCtxGetCurrent","cuStreamGetCtx_main",
        "cuStreamGetCtx_copy","cudaDeviceSynchronize","cudaStreamSynchronize_main","cudaStreamSynchronize_copy"};
    if(state.reason!=4||state.count<5||state.count>7||
       state.backend!=reinterpret_cast<uintptr_t>(backend)||!state.main_stream||!state.copy_stream||
       state.current_device!=state.physical_device||!state.current_context||
       state.current_context!=state.main_context||state.current_context!=state.copy_context)return false;
    for(uint32_t i=0;i<state.count;++i) {
        const auto& call=state.calls[i];
        if(!call.operation||std::strcmp(call.operation,expected[i]))return false;
        if(i+1<state.count&&(call.primary||call.pending))return false;
    }
    const auto& failed=state.calls[state.count-1];
    result.primary=failed.primary;result.pending=failed.pending;
    if(failed.primary==2&&(failed.pending==0||failed.pending==2)) {
        result.stage=failed.operation; // Classified by the same startup transaction.
    } else result.stage="backend_stream_unclassified";
    return false;
}
}
