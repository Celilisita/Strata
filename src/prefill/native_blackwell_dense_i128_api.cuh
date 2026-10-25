template<class F> constexpr int dense_i128_shared_bytes() {
    return 128*(int(sizeof(typename F::Block))+4)+128*8*12*4;
}
template<class F> void configure_dense_i128() {
    static const bool configured=[] {
        CUDA_CHECK(cudaFuncSetAttribute(dense_i128_group_int8<F>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,dense_i128_shared_bytes<F>()));
        return true;
    }();
    (void)configured;
}
template<class F>
void launch_dense_i128(const void* weights,const void* input,void* codes,void* meta,
    float* output,int width,int rows,int outputs,void* stream) {
    configure_dense_i128<F>(); // Before the encoder; failure cannot leave a partial input submission.
    const int cells=rows*(width/32);
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(codes),
        static_cast<GroupInputMeta*>(meta),width,rows);
    NativeMmqEncodedInput encoded{codes,meta,nullptr,rows,128};
    dense_i128_group_int8<F><<<dim3(outputs/128,(rows+127)/128),256,dense_i128_shared_bytes<F>(),
        static_cast<cudaStream_t>(stream)>>>(static_cast<const typename F::Block*>(weights),
        output,width,rows,outputs,encoded);
}
void run_blackwell_dense_i128_int8_tile(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,void* stream) {
    if((type!=12 && type!=13) || (width!=2048 && width!=4096) || outputs<128 || outputs%128)
        throw std::invalid_argument("dense I128 requires Q4/Q5, K2048/4096, Nmultiple128");
    validate_dense_group_views(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,128);
    if(type==12) launch_dense_i128<GroupQ4>(weights,input,codes,meta,output,width,rows,outputs,stream);
    else launch_dense_i128<GroupQ5>(weights,input,codes,meta,output,width,rows,outputs,stream);
    CUDA_CHECK(cudaGetLastError());
}
bool try_blackwell_dense_i128(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,
    void* stream,int control_features) {
    static const unsigned mask=blackwell_dense_int8_mask(std::getenv("STRATA_BENCH_DENSE_INT8_I128"));
    if(control_features!=64 || !blackwell_dense_int8_selected(mask,type,rows,width,outputs)) return false;
    run_blackwell_dense_i128_int8_tile(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream);
    return true;
}
template<class F> BlackwellDenseResources dense_i128_resources() {
    configure_dense_i128<F>();
    cudaFuncAttributes attr{};int active=0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_i128_group_int8<F>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_i128_group_int8<F>,256,
        dense_i128_shared_bytes<F>()));
    return {attr.numRegs,attr.sharedSizeBytes+dense_i128_shared_bytes<F>(),attr.localSizeBytes,
        attr.maxThreadsPerBlock,active};
}
BlackwellDenseResources blackwell_dense_i128_int8_resources(int type,int width) {
    check_device();
    if((type!=12 && type!=13) || (width!=2048 && width!=4096))
        throw std::invalid_argument("invalid dense I128 resource query");
    return type==12 ? dense_i128_resources<GroupQ4>() : dense_i128_resources<GroupQ5>();
}
