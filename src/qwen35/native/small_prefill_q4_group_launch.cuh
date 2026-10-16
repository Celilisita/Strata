#pragma once
void small_prefill_q4_group_pair(Q4GroupCodes gate,Q4GroupCodes up,const void *input,float *gout,float *uout,float *activation,cudaStream_t stream,bool materialize,void *q6_codes){
    const size_t gb=q4_group_codes_bytes(gate),ub=q4_group_codes_bytes(up);
    const size_t ib=small_prefill_half_input_bytes(gate.columns,32),ob=size_t(gate.rows)*32*4;
    if(!configured||gate.rows!=up.rows||gate.columns!=up.columns||!input||!gout||!uout||!activation)
        throw std::invalid_argument("group pair requires configured equal PP32 views and buffers");
    const void *pointers[]={gate.data,up.data,input,gout,uout,activation,q6_codes};
    const size_t extents[]={gb,ub,ib,ob,ob,ob,q6_codes?small_prefill_half_input_bytes(gate.rows,32):0};
    for(int i=0;i<7;++i){
        if(pointers[i]&&reinterpret_cast<uintptr_t>(pointers[i])%16)throw std::invalid_argument("unaligned group Q4 buffer");
        for(int j=0;j<i;++j)if(extents[i]&&extents[j]&&detail::split_overlap(pointers[i],extents[i],pointers[j],extents[j]))
            throw std::invalid_argument("group Q4 live buffers overlap");
    }
    paired_q4_group_register<<<unsigned(gate.rows/64),128,0,stream>>>(gate,up,static_cast<const detail::HalfAffineInputBlock *>(input),gout,uout,activation,materialize,static_cast<detail::HalfInputBlock *>(q6_codes));
    check_cuda(cudaGetLastError(),"owned group register Q4 pair");
}
