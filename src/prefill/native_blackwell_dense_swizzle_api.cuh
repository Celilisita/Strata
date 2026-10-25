namespace {
#include "native_blackwell_dense_swizzle_int8.cuh"
}
// Included after validate_dense_group_views; same encoder/workspaces/stream as I64.
template<class F> void launch_dense_swizzle(const void* weights,const void* input,
    void* codes,void* meta,float* output,int width,int rows,int outputs,void* stream) {
    const int cells=rows*(width/32);
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(codes),
        static_cast<GroupInputMeta*>(meta),width,rows);
    NativeMmqEncodedInput encoded{codes,meta,nullptr,rows,128};
    dense_swizzle_group_int8<F><<<dim3(outputs/64,(rows+127)/128),256,0,
        static_cast<cudaStream_t>(stream)>>>(static_cast<const typename F::Block*>(weights),
        output,width,rows,outputs,encoded);
}
void run_blackwell_dense_swizzle_int8_tile(int type,const void* weights,const void* input,
    void* codes,size_t code_bytes,void* meta,size_t meta_bytes,float* output,
    int width,int rows,int outputs,void* stream) {
    if((type!=12&&type!=13)||(width!=2048&&width!=4096)||outputs<64||outputs%64)
        throw std::invalid_argument("dense swizzle requires Q4/Q5, K2048/4096, Nmultiple64");
    validate_dense_group_views(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,128);
    if(type==12)launch_dense_swizzle<GroupQ4>(weights,input,codes,meta,output,width,rows,outputs,stream);
    else launch_dense_swizzle<GroupQ5>(weights,input,codes,meta,output,width,rows,outputs,stream);
    CUDA_CHECK(cudaGetLastError());
}
bool try_blackwell_dense_swizzle(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,
    void* stream,int control_features) {
    static const unsigned mask=blackwell_dense_int8_mask(std::getenv("STRATA_BENCH_DENSE_INT8_SWIZZLE"));
    if(control_features!=64||!blackwell_dense_int8_selected(mask,type,rows,width,outputs))return false;
    run_blackwell_dense_swizzle_int8_tile(type,weights,input,codes,code_bytes,meta,meta_bytes,
        output,width,rows,outputs,stream);
    return true;
}
template<class F> BlackwellDenseResources dense_swizzle_resources() {
    cudaFuncAttributes attr{};int active=0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_swizzle_group_int8<F>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_swizzle_group_int8<F>,256,0));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,attr.maxThreadsPerBlock,active};
}
BlackwellDenseResources blackwell_dense_swizzle_int8_resources(int type,int width) {
    check_device();
    if((type!=12&&type!=13)||(width!=2048&&width!=4096))
        throw std::invalid_argument("invalid dense swizzle resource query");
    return type==12?dense_swizzle_resources<GroupQ4>():dense_swizzle_resources<GroupQ5>();
}
