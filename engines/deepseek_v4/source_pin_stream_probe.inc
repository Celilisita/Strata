// Included inside strata::deepseek_v4, after the private Pipeline registry.
// Cold source-owner diagnostics only; borrow existing handles without creating any.
bool source_pin_stream_probe(ggml_backend_t backend,SourcePinStreamProbe* out) {
    if(!out)return false;
    *out={};out->backend=reinterpret_cast<uintptr_t>(backend);
    out->backend_device=out->physical_device=out->current_device=out->stream_index=-1;
    if(!backend||!ggml_backend_is_cuda(backend)){out->reason=1;return false;}
    auto* ctx=static_cast<ggml_backend_cuda_context*>(backend->context);
    if(!ctx){out->reason=1;return false;}
    std::lock_guard<std::mutex> guard(registry_mutex);
    out->backend_device=ctx->device;out->stream_index=ctx->curr_stream_no;
    if(ctx->device<0||ctx->device>=GGML_CUDA_MAX_DEVICES||ctx->curr_stream_no<0||ctx->curr_stream_no>=GGML_CUDA_MAX_STREAMS){out->reason=2;return false;}
    const auto found=registry.find(ctx);
    const auto main=ctx->streams[ctx->device][ctx->curr_stream_no];
    const auto copy=found==registry.end()?nullptr:found->second->copy;
    out->main_stream=reinterpret_cast<uintptr_t>(main);out->copy_stream=reinterpret_cast<uintptr_t>(copy);
    if(!main||!copy){out->reason=3;return false;}
    const auto& info=ggml_cuda_info();
    if(ctx->device>=info.device_count){out->reason=2;return false;}
    out->physical_device=info.devices[ctx->device].physical_device;
    auto record=[&](const char* operation,uint32_t primary) {
        const uint32_t pending=uint32_t(cudaPeekAtLastError());
        out->calls[out->count++]={operation,primary,pending};
        if(primary||pending){out->reason=4;return false;}
        return true;
    };
    int current=-1;const auto device_result=cudaGetDevice(&current);out->current_device=current;
    if(!record("cudaGetDevice",uint32_t(device_result)))return false;
    if(current!=out->physical_device){out->reason=5;return false;}
    CUcontext current_context=nullptr;
    const auto context_result=cuCtxGetCurrent(&current_context);
    out->current_context=reinterpret_cast<uintptr_t>(current_context);
    if(!record("cuCtxGetCurrent",uint32_t(context_result)))return false;
    CUcontext main_context=nullptr;
    const auto main_result=cuStreamGetCtx(reinterpret_cast<CUstream>(main),&main_context);
    out->main_context=reinterpret_cast<uintptr_t>(main_context);
    if(!record("cuStreamGetCtx_main",uint32_t(main_result)))return false;
    CUcontext copy_context=nullptr;
    const auto copy_result=cuStreamGetCtx(reinterpret_cast<CUstream>(copy),&copy_context);
    out->copy_context=reinterpret_cast<uintptr_t>(copy_context);
    if(!record("cuStreamGetCtx_copy",uint32_t(copy_result)))return false;
    if(!current_context||current_context!=main_context||current_context!=copy_context){out->reason=6;return false;}
    if(!record("cudaDeviceSynchronize",uint32_t(cudaDeviceSynchronize())))return false;
    if(!record("cudaStreamSynchronize_main",uint32_t(cudaStreamSynchronize(main))))return false;
    if(!record("cudaStreamSynchronize_copy",uint32_t(cudaStreamSynchronize(copy))))return false;
    return true;
}
