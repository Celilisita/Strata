#pragma once
// Included inside the owned namespace after the raw Q4 launch functions.
namespace {
__global__ void reduce_small_pair(const float *parts,float *first,float *second,size_t vectors,int cells) {
    const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;
    if(i>=vectors)return;
    const auto *p=reinterpret_cast<const float4 *>(parts)+size_t(blockIdx.y)*cells*vectors;
    float4 preceding=make_float4(0,0,0,0);
    for(int cell=1;cell<cells;++cell) {
        const float4 v=p[size_t(cell)*vectors+i];
        preceding=make_float4(__fadd_rn(preceding.x,v.x),__fadd_rn(preceding.y,v.y),
                             __fadd_rn(preceding.z,v.z),__fadd_rn(preceding.w,v.w));
    }
    const float4 tail=p[i];
    auto *out=reinterpret_cast<float4 *>(blockIdx.y?second:first);
    out[i]=make_float4(__fadd_rn(tail.x,preceding.x),__fadd_rn(tail.y,preceding.y),
                      __fadd_rn(tail.z,preceding.z),__fadd_rn(tail.w,preceding.w));
}
}
void small_prefill_q4_small_pair_linear(Matrix first,Matrix second,const void *input,float *a,float *b,
                                       cudaStream_t stream,void *parts,size_t bytes) {
    const size_t ab=matrix_bytes(first),bb=matrix_bytes(second),ib=small_prefill_half_input_bytes(first.columns,32);
    const size_t ob=size_t(first.rows)*32*4,required=2*size_t(first.columns/256)*ob;
    if(!configured||first.quant!=Quant::Q4K||second.quant!=Quant::Q4K||first.layout!=BlockLayout::Q4Pairs||
       second.layout!=BlockLayout::Q4Pairs||first.rows!=second.rows||first.columns!=second.columns||
       first.rows>=64||first.rows%4||first.columns/256>32||!input||!a||!b||!parts||bytes<required)
        throw std::invalid_argument("small Q4 pair requires equal bounded raw geometry and sufficient arena");
    const struct {const void *p;size_t bytes;} operands[]={{first.data,ab},{second.data,bb},{input,ib},{a,ob},{b,ob},{parts,required}};
    for(int i=0;i<6;++i) {
        if(reinterpret_cast<uintptr_t>(operands[i].p)%16)throw std::invalid_argument("small Q4 pair alignment");
        for(int j=i+1;j<6;++j)if(!(i==0&&j==1)&&detail::split_overlap(operands[i].p,operands[i].bytes,operands[j].p,operands[j].bytes))
            throw std::invalid_argument("small Q4 pair overlapping live operands");
    }
    const int cells=first.columns/256;
    register_q4<0,true><<<dim3(1,2,cells),128,0,stream>>>(first,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts),second);
    check_cuda(cudaGetLastError(),"owned small Q4 pair projection");
    const size_t vectors=ob/sizeof(float4);
    reduce_small_pair<<<dim3(unsigned((vectors+Threads-1)/Threads),2),Threads,0,stream>>>(static_cast<const float *>(parts),a,b,vectors,cells);
    check_cuda(cudaGetLastError(),"owned small Q4 pair reduction");
}
