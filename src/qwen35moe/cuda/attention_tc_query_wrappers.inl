void attention_tc_prefill_query_rows(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream,int query_rows) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity) ||
       (query_rows!=16 && query_rows!=32 && query_rows!=64))
        throw std::invalid_argument("unsupported attention query-row geometry");
    const int width=first+count,padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    const dim3 grid(Heads,(count+query_rows-1)/query_rows);
    if(query_rows==64) {
        prepare_query64();
        prefill_query64<<<grid,256,Query64SharedBytes,s>>>(q,reinterpret_cast<const half*>(k),
            reinterpret_cast<const half*>(v),first,count,out);
    } else if(query_rows==32) prefill_query32<<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    else prefill_register<<<grid,128,0,s>>>(q,reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void attention_tc_prefill_half_pv(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity)||first+count>30720)
        throw std::invalid_argument("half PV requires bounded valid attention geometry");
    const int width=first+count,padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    prefill_halfpv_query32<<<dim3(Heads,(count+31)/32),128,0,s>>>(q,
        reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
AttentionTcQueryResources attention_tc_half_pv_resources() {
    cudaFuncAttributes attr{};auto error=cudaFuncGetAttributes(&attr,prefill_halfpv_query32);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    int blocks=0;error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prefill_halfpv_query32,128,0);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,blocks,attr.sharedSizeBytes,0,128};
}
AttentionTcQueryResources attention_tc_query_resources(int query_rows) {
    if(query_rows!=16 && query_rows!=32 && query_rows!=64)
        throw std::invalid_argument("attention resource rows require16,32 or64");
    if(query_rows==64) prepare_query64();
    const auto kernel=query_rows==64 ? prefill_query64 : query_rows==32 ? prefill_query32 : prefill_register;
    const int threads=query_rows==64 ? 256 : 128;
    const size_t dynamic=query_rows==64 ? Query64SharedBytes : 0;
    cudaFuncAttributes attr{};
    auto error=cudaFuncGetAttributes(&attr,kernel);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    int blocks=0;
    error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,threads,dynamic);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes+dynamic,attr.localSizeBytes,blocks,attr.sharedSizeBytes,dynamic,threads};
}
AttentionTcResources attention_tc_resources(bool compact,bool reuse) {
    if(reuse && !compact && register_output_enabled()) return attention_register_resources();
    cudaFuncAttributes attr{};
    const auto error=reuse?cudaFuncGetAttributes(&attr,prefill<false,true>):
        compact?cudaFuncGetAttributes(&attr,prefill_register):cudaFuncGetAttributes(&attr,prefill<false,false>);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes};
}
void attention_tc_prefill_gqa_async(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity)||first+count>30720)
        throw std::invalid_argument("GQA async requires bounded causal attention views");
    const std::array<std::pair<uintptr_t,size_t>,4> spans{{
        {uintptr_t(q),size_t(count)*Heads*Dim*sizeof(float)},
        {uintptr_t(k),size_t(capacity)*CacheStride*sizeof(uint16_t)},
        {uintptr_t(v),size_t(capacity)*CacheStride*sizeof(uint16_t)},
        {uintptr_t(out),size_t(count)*Heads*Dim*sizeof(float)}}};
    for(size_t i=0;i<spans.size();++i) {
        const auto [begin,bytes]=spans[i];
        if(begin%16 || begin>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("GQA async alignment/span overflow");
        for(size_t j=0;j<i;++j)
            if(begin<spans[j].first+spans[j].second && spans[j].first<begin+bytes)
                throw std::invalid_argument("GQA async query/cache/output alias");
    }
    prepare_gqa_async();
    const int width=first+count,padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    prefill_gqa_async<<<dim3(2,(count+7)/8),256,GqaAsyncSharedBytes,s>>>(q,
        reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
AttentionTcQueryResources attention_tc_gqa_async_resources() {
    prepare_gqa_async();cudaFuncAttributes attr{};
    auto error=cudaFuncGetAttributes(&attr,prefill_gqa_async);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    int blocks=0;error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prefill_gqa_async,256,GqaAsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes+GqaAsyncSharedBytes,attr.localSizeBytes,
        blocks,attr.sharedSizeBytes,GqaAsyncSharedBytes,256};
}
void attention_tc_prefill_gqa32_async(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream) {
    if(!q||!k||!v||!out||!stream||!attention_tc_fits(first,count,capacity)||first+count>30720)
        throw std::invalid_argument("GQA async requires bounded causal attention views");
    const std::array<std::pair<uintptr_t,size_t>,4> spans{{
        {uintptr_t(q),size_t(count)*Heads*Dim*sizeof(float)},
        {uintptr_t(k),size_t(capacity)*CacheStride*sizeof(uint16_t)},
        {uintptr_t(v),size_t(capacity)*CacheStride*sizeof(uint16_t)},
        {uintptr_t(out),size_t(count)*Heads*Dim*sizeof(float)}}};
    for(size_t i=0;i<spans.size();++i) {
        const auto [begin,bytes]=spans[i];
        if(begin%16 || begin>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("GQA async alignment/span overflow");
        for(size_t j=0;j<i;++j)
            if(begin<spans[j].first+spans[j].second && spans[j].first<begin+bytes)
                throw std::invalid_argument("GQA async query/cache/output alias");
    }
    prepare_gqa32_async();
    const int width=first+count,padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    prefill_gqa32_async<<<dim3(4,(count+7)/8),128,Gqa32AsyncSharedBytes,s>>>(q,
        reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
AttentionTcQueryResources attention_tc_gqa32_async_resources() {
    prepare_gqa32_async();cudaFuncAttributes attr{};
    auto error=cudaFuncGetAttributes(&attr,prefill_gqa32_async);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    int blocks=0;error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prefill_gqa32_async,128,Gqa32AsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes+Gqa32AsyncSharedBytes,attr.localSizeBytes,
        blocks,attr.sharedSizeBytes,Gqa32AsyncSharedBytes,128};
}
#include "attention_tc_gqa_key64_host.inl"
