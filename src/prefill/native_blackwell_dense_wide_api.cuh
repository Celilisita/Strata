// Included after validate_dense_group_views; no allocation, stream or extra launch.
template<class F, int Features>
void launch_dense_wide(const void* weights,const void* input,void* codes,void* meta,
    float* output,int width,int rows,int outputs,void* stream) {
    const int cells = rows*(width/32);
    encode_dense_group_int8<<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(codes),
        static_cast<GroupInputMeta*>(meta),width,rows);
    NativeMmqEncodedInput encoded{codes,meta,nullptr,rows,128};
    dense_wide_group_int8<F,Features><<<dim3(outputs/Features,(rows+127)/128),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const typename F::Block*>(weights),output,width,rows,outputs,encoded);
}
void run_blackwell_dense_wide_int8_tile(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,
    void* stream,int output_rows) {
    if((type!=12 && type!=13) || (output_rows!=16 && output_rows!=32 && output_rows!=64) ||
        (width!=2048 && width!=4096) || outputs<64 || outputs%64)
        throw std::invalid_argument("dense wide INT8 requires Q4/Q5, K2048/4096, Nmultiple64, I16/32/64");
    validate_dense_group_views(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,128);
    if(output_rows==16) {
        if(type==12) launch_dense_group_int8<GroupQ4,128>(weights,input,codes,meta,output,width,rows,outputs,stream);
        else launch_dense_group_int8<GroupQ5,128>(weights,input,codes,meta,output,width,rows,outputs,stream);
    } else if(type==12) {
        if(output_rows==32) launch_dense_wide<GroupQ4,32>(weights,input,codes,meta,output,width,rows,outputs,stream);
        else launch_dense_wide<GroupQ4,64>(weights,input,codes,meta,output,width,rows,outputs,stream);
    } else {
        if(output_rows==32) launch_dense_wide<GroupQ5,32>(weights,input,codes,meta,output,width,rows,outputs,stream);
        else launch_dense_wide<GroupQ5,64>(weights,input,codes,meta,output,width,rows,outputs,stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
template<class F,int Features> BlackwellDenseResources dense_wide_resources() {
    cudaFuncAttributes attr{}; int active=0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_wide_group_int8<F,Features>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_wide_group_int8<F,Features>,256,0));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,attr.maxThreadsPerBlock,active};
}
BlackwellDenseResources blackwell_dense_wide_int8_resources(int type,int width,int output_rows) {
    check_device();
    if((type!=12 && type!=13) || (width!=2048 && width!=4096) ||
        (output_rows!=16 && output_rows!=32 && output_rows!=64))
        throw std::invalid_argument("invalid dense wide resource query");
    if(output_rows==16) return blackwell_dense_group_int8_resources(type,width,128);
    if(type==12) return output_rows==32 ? dense_wide_resources<GroupQ4,32>() : dense_wide_resources<GroupQ4,64>();
    return output_rows==32 ? dense_wide_resources<GroupQ5,32>() : dense_wide_resources<GroupQ5,64>();
}
