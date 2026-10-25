// Included after the original public API definitions; no change to their typed dispatch.
namespace {
void prepare_gqa_key64() {
    int device=0;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    static thread_local int prepared=-1;
    if(prepared==device) return;
    cudaDeviceProp properties{};error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if(properties.major!=12 || properties.minor!=0) throw std::invalid_argument("GQA key64 requires SM120");
    error=cudaFuncSetAttribute(prefill_gqa_key64_async,cudaFuncAttributeMaxDynamicSharedMemorySize,GqaKey64AsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    prepared=device;
}
}
void attention_tc_prefill_gqa_key64(const float* q,uint16_t* k,uint16_t* v,
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
    prepare_gqa_key64();
    const int width=first+count,padded=((width+63)/64)*64;
    auto s=static_cast<cudaStream_t>(stream);
    if(padded>width) zero_tail<<<((padded-width)*CacheStride+255)/256,256,0,s>>>(
        reinterpret_cast<half*>(k),reinterpret_cast<half*>(v),width,(padded-width)*CacheStride);
    prefill_gqa_key64_async<<<dim3(4,(count+7)/8),128,GqaKey64AsyncSharedBytes,s>>>(q,
        reinterpret_cast<const half*>(k),reinterpret_cast<const half*>(v),first,count,out);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
AttentionTcQueryResources attention_tc_gqa_key64_resources() {
    prepare_gqa_key64();cudaFuncAttributes attr{};
    auto error=cudaFuncGetAttributes(&attr,prefill_gqa_key64_async);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    int blocks=0;error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,prefill_gqa_key64_async,128,GqaKey64AsyncSharedBytes);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return {attr.numRegs,attr.sharedSizeBytes+GqaKey64AsyncSharedBytes,attr.localSizeBytes,
        blocks,attr.sharedSizeBytes,GqaKey64AsyncSharedBytes,128};
}
