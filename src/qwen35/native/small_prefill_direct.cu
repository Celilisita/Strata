#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include "small_prefill_mma.cuh"
#include "small_prefill_blocked.cuh"
#include "small_prefill_half_input.cuh"
#include "small_prefill_half_codes.hpp"
#include "small_prefill_fused_reduce.cuh"
#include "small_prefill_direct_limits.cuh"
#include "small_prefill_half_raw.hpp"
#include "small_prefill_direct_kernel.cuh"
#include "small_prefill_q6_register.cuh"
#include "small_prefill_q6_prefetch.cuh"
#include "small_prefill_q6_planes.hpp"
#include <limits>

namespace strata::qwen35::native {
namespace {
constexpr int Rows=detail::DirectRows,Tokens=detail::DirectTokens,Threads=256;
constexpr int DirectSplits=detail::DirectSplits;
thread_local int code_words=0;
thread_local bool split_sm70=false;
using detail::direct::direct_project;
// Independent pairwise reduction; all partial writes precede this kernel on the stream.
__global__ void direct_split_reduce(const float *partials,float *output,size_t vectors){
    const size_t index=size_t(blockIdx.x)*Threads+threadIdx.x;
    if(index>=vectors)return;
    const auto *source=reinterpret_cast<const float4 *>(partials);
    const float4 a=source[index],b=source[vectors+index],c=source[2*vectors+index];
    const float4 result=make_float4(
        __fadd_rn(__fadd_rn(a.x,b.x),c.x),
        __fadd_rn(__fadd_rn(a.y,b.y),c.y),
        __fadd_rn(__fadd_rn(a.z,b.z),c.z),
        __fadd_rn(__fadd_rn(a.w,b.w),c.w));
    reinterpret_cast<float4 *>(output)[index]=result;
}

__global__ void direct_split2_reduce(const float *parts,float *output,size_t vectors){
    const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;if(i>=vectors)return;
    const auto *p=reinterpret_cast<const float4 *>(parts);const float4 a=p[i],b=p[vectors+i];
    reinterpret_cast<float4 *>(output)[i]=make_float4(__fadd_rn(a.x,b.x),__fadd_rn(a.y,b.y),__fadd_rn(a.z,b.z),__fadd_rn(a.w,b.w));
}
}
template<bool Blocked,bool Complete>
void configure_direct_instances(){
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,Blocked,Complete>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K>(code_words)),"direct Q6 shared setup");
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q4K,Blocked,Complete>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q4K>(code_words)),"direct Q4 shared setup");
}
template<bool Complete>
void configure_half_raw_instances(){
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q4K,true,Complete,false,false,true,false,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q4K,false,true>(2)),"half raw Q4 shared setup");
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,Complete,false,false,true,false,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,false,true>(2)),"half raw Q6 shared setup");
}
void configure_small_prefill_direct(){
    int device=0;check_cuda(cudaGetDevice(&device),"direct MMA setup device");
    const auto caps=capabilities(device);code_words=caps.major>=8?1:2;
    split_sm70=caps.major==7&&caps.minor==0;
    if(split_sm70)check_cuda(cudaFuncSetAttribute(detail::prefetch_q6_project<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"prefetch Q6 shared policy");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(detail::prefetch_q6_project<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"raw prefetch Q6 shared policy");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(detail::prefetch_q6_project<true,1,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"plane prefetch Q6 shared policy");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(detail::prefetch_q6_project<true,1,true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"byte plane Q6 shared policy");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(detail::register_q6_project,cudaFuncAttributePreferredSharedMemoryCarveout,100),"register Q6 shared policy");
    if(split_sm70){configure_half_raw_instances<true>();configure_half_raw_instances<false>();
        check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,false,true,true,false,false,true,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       detail::direct_shared_bytes<Quant::Q6K,false,true>(2)),"raw Q6 two-part shared setup");
        check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,true,true,false,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       detail::direct_shared_bytes<Quant::Q6K,true,true>(2)),"byte-code Q6 three-part shared setup");
        check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,false,true,false,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       detail::direct_shared_bytes<Quant::Q6K,true,true>(2)),"byte-code Q6 complete shared setup");
    }
    if(caps.shared_per_block<detail::direct_shared_bytes<Quant::Q6K,true>(code_words))
        throw std::runtime_error("direct MMA shared capacity insufficient");
    configure_direct_instances<false,false>();configure_direct_instances<true,false>();
    configure_direct_instances<false,true>();configure_direct_instances<true,true>();
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true>(code_words)),"prepared direct Q6 segmented shared setup");
    check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true>(code_words)),"prepared direct Q6 complete shared setup");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true>(code_words)),"prepared direct Q6 split shared setup");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,true,true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true>(code_words)),"prepared half-code Q6 split shared setup");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,true,true,true,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true>(code_words)),"prepared half-code fused Q6 shared setup");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,true,true,true,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true,true>(code_words)),"prepared half-code compact input shared setup");
    if(split_sm70)check_cuda(cudaFuncSetAttribute(direct_project<Quant::Q6K,true,true,true,false,true,true,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   detail::direct_shared_bytes<Quant::Q6K,true,true>(code_words)),"prepared half-code complete input shared setup");
}
template<bool Blocked,bool PackedCodes=false>
void direct_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    matrix_bytes(weight,PackedCodes);
    if constexpr(PackedCodes)
        if(weight.layout!=BlockLayout::Q6Codes||weight.quant!=Quant::Q6K||reinterpret_cast<uintptr_t>(weight.data)%16)
            throw std::invalid_argument("prepared direct requires aligned Q6 code planes");
    if(tokens<2||tokens>=64||processors<=0||!code_words)throw std::invalid_argument("direct MMA shape/setup");
    const int tiles=(weight.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    const dim3 grid((weight.rows+Rows-1)/Rows,(tokens+Tokens-1)/Tokens);
    if(weight.quant==Quant::Q6K){
        if(units==tiles)direct_project<Quant::Q6K,Blocked,true,PackedCodes><<<grid,Threads,detail::direct_shared_bytes<Quant::Q6K,PackedCodes>(code_words),stream>>>(
            weight,input,output,tokens,units);
        else direct_project<Quant::Q6K,Blocked,false,PackedCodes><<<grid,Threads,detail::direct_shared_bytes<Quant::Q6K,PackedCodes>(code_words),stream>>>(
            weight,input,output,tokens,units);
    }else{
        if(units==tiles)direct_project<Quant::Q4K,Blocked,true><<<grid,Threads,detail::direct_shared_bytes<Quant::Q4K>(code_words),stream>>>(
            weight,input,output,tokens,units);
        else direct_project<Quant::Q4K,Blocked,false><<<grid,Threads,detail::direct_shared_bytes<Quant::Q4K>(code_words),stream>>>(
            weight,input,output,tokens,units);
    }
    check_cuda(cudaGetLastError(),"owned register MMA projection");
}
void small_prefill_direct_linear(Matrix weight,const void *input,float *output,int tokens,
                                 cudaStream_t stream,int processors){
    direct_linear<false>(weight,input,output,tokens,stream,processors);
}
void small_prefill_direct_codes_linear(Matrix weight,const void *input,float *output,int tokens,
                                       cudaStream_t stream,int processors){
    direct_linear<true,true>(weight,input,output,tokens,stream,processors);
}
template<bool HalfWeight,bool Fused=false,bool CompactInput=false>
void direct_codes_split_linear(Matrix weight,const void *input,float *output,int tokens,
                                             void *scratch,size_t scratch_bytes,cudaStream_t stream,int processors,
                                             unsigned *counters=nullptr){
    const size_t required=detail::direct_codes_split_bytes<HalfWeight>(weight,tokens);
    const size_t weight_bytes=HalfWeight?size_t(weight.rows)*size_t(weight.columns/256)*HalfCodeBlockBytes:
                                       matrix_bytes(weight,true);
    const size_t input_bytes=small_prefill_half_input_bytes(weight.columns,tokens);
    const size_t output_bytes=required/DirectSplits;
    if(!split_sm70||code_words!=2||processors<=0||!input||!output||!scratch||scratch_bytes<required||
       reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       reinterpret_cast<uintptr_t>(scratch)%16||
       detail::split_overlap(scratch,required,weight.data,weight_bytes)||
       detail::split_overlap(scratch,required,input,input_bytes)||detail::split_overlap(scratch,required,output,output_bytes)||
       detail::split_overlap(output,output_bytes,weight.data,weight_bytes)||detail::split_overlap(output,output_bytes,input,input_bytes))
        throw std::invalid_argument("split Q6 requires setup and aligned nonoverlapping caller scratch extent");
    const dim3 grid(weight.rows/Rows,1,DirectSplits);
    direct_project<Quant::Q6K,true,true,true,true,true,HalfWeight,Fused,CompactInput><<<grid,Threads,detail::direct_shared_bytes<Quant::Q6K,true,CompactInput>(code_words),stream>>>(
        weight,input,static_cast<float *>(scratch),tokens,(weight.rows+127)/128,Fused?output:nullptr,counters);
    check_cuda(cudaGetLastError(),"owned split Q6 projection");
    if constexpr(Fused)return;
    const size_t vectors=output_bytes/sizeof(float4);
    direct_split_reduce<<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(scratch),output,vectors);
    check_cuda(cudaGetLastError(),"owned split Q6 reduction");
}
size_t small_prefill_direct_codes_split_bytes(Matrix weight,int tokens){
    return detail::direct_codes_split_bytes<false>(weight,tokens);
}
size_t small_prefill_half_codes_split_bytes(HalfCodeMatrix weight,int tokens){
    half_code_matrix_bytes(weight);
    return detail::direct_codes_split_bytes<true>({weight.data,Quant::Q6K,weight.rows,weight.columns},tokens);
}
void small_prefill_direct_codes_split_linear(Matrix weight,const void *input,float *output,int tokens,
                                            void *scratch,size_t bytes,cudaStream_t stream,int processors){
    direct_codes_split_linear<false>(weight,input,output,tokens,scratch,bytes,stream,processors);
}
void small_prefill_half_codes_split_linear(HalfCodeMatrix weight,const void *input,float *output,int tokens,
                                         void *scratch,size_t bytes,cudaStream_t stream,int processors){
    half_code_matrix_bytes(weight);
    // The private kernel descriptor uses dimensions/pointer only; its template
    // selects the prepared layout. This descriptor never escapes this operator.
    direct_codes_split_linear<true>({weight.data,Quant::Q6K,weight.rows,weight.columns},
                                   input,output,tokens,scratch,bytes,stream,processors);
}
void small_prefill_half_codes_fused_linear(HalfCodeMatrix weight,const void *input,float *output,int tokens,
                                         HalfCodeWorkspace workspace,cudaStream_t stream,int processors){
    const size_t required=half_code_workspace_bytes(weight);
    if(!workspace.ready||workspace.rows!=weight.rows||workspace.stream!=stream||
       !workspace.data||workspace.bytes<required)
        throw std::invalid_argument("fused projection requires initialized matching stream scratch");
    const size_t partial_bytes=size_t(3)*32*weight.rows*4;
    auto *counters=reinterpret_cast<unsigned *>(static_cast<unsigned char *>(workspace.data)+partial_bytes);
    // Include ticket bytes in all overlap guards, even though producers write
    // only partials into the first portion of the declared scratch lease.
    const size_t in_bytes=small_prefill_half_input_bytes(weight.columns,tokens),out_bytes=partial_bytes/3;
    if(detail::split_overlap(workspace.data,required,input,in_bytes)||detail::split_overlap(workspace.data,required,output,out_bytes)||
       detail::split_overlap(workspace.data,required,weight.data,weight.bytes))
        throw std::invalid_argument("fused ticket extent overlaps input/output/weights");
    direct_codes_split_linear<true,true>({weight.data,Quant::Q6K,weight.rows,weight.columns},
        input,output,tokens,workspace.data,workspace.bytes,stream,processors,counters);
}
void small_prefill_half_codes_compact_linear(HalfCodeMatrix weight,const void *input,float *output,int tokens,
                                            void *scratch,size_t bytes,cudaStream_t stream,int processors){
    half_code_matrix_bytes(weight);
    direct_codes_split_linear<true,false,true>({weight.data,Quant::Q6K,weight.rows,weight.columns},
                                             input,output,tokens,scratch,bytes,stream,processors);
}
void small_prefill_half_codes_complete_linear(HalfCodeMatrix weight,const void *input,float *output,int tokens,
                                             cudaStream_t stream,int processors){
    half_code_matrix_bytes(weight);
    const Matrix view{weight.data,Quant::Q6K,weight.rows,weight.columns};
    const size_t output_bytes=detail::direct_codes_split_bytes<true>(view,tokens)/DirectSplits;
    const size_t input_bytes=small_prefill_half_input_bytes(weight.columns,tokens);
    if(!split_sm70||code_words!=2||processors<=0||!input||!output||
       reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(output,output_bytes,input,input_bytes)||detail::split_overlap(output,output_bytes,weight.data,weight.bytes)||
       detail::split_overlap(input,input_bytes,weight.data,weight.bytes))
        throw std::invalid_argument("complete half-code projection requires nonoverlapping aligned SM70 buffers");
    direct_project<Quant::Q6K,true,true,true,false,true,true,false,true><<<weight.rows/Rows,Threads,detail::direct_shared_bytes<Quant::Q6K,true,true>(code_words),stream>>>(
        view,input,output,tokens,(weight.rows+127)/128);
    check_cuda(cudaGetLastError(),"owned complete Q6 half-code projection");
}
void small_prefill_direct_blocked_linear(Matrix weight,const void *input,float *output,int tokens,
                                         cudaStream_t stream,int processors){
    direct_linear<true>(weight,input,output,tokens,stream,processors);
}
void small_prefill_half_raw_linear(Matrix w,const void *input,float *output,int tokens,cudaStream_t stream,int processors,RawHalfScratch scratch,bool allow_split){
    const size_t weight_bytes=matrix_bytes(w,true),input_bytes=small_prefill_half_input_bytes(w.columns,tokens);
    const size_t output_bytes=size_t(w.rows)*tokens*4;
    if(!split_sm70||code_words!=2||tokens!=32||processors<=0||!input||!output||
       reinterpret_cast<uintptr_t>(w.data)%16||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(input,input_bytes,w.data,weight_bytes)||detail::split_overlap(output,output_bytes,w.data,weight_bytes)||
       detail::split_overlap(output,output_bytes,input,input_bytes))
        throw std::invalid_argument("raw half-code projection requires aligned nonoverlapping SM70 PP32 buffers");
    if(w.layout==BlockLayout::Q6Codes){
        if(allow_split&&half_raw_slices(w,processors)==3){
            const size_t required=3*output_bytes;
            if(!scratch.data||scratch.bytes<required||reinterpret_cast<uintptr_t>(scratch.data)%16||
               detail::split_overlap(scratch.data,required,input,input_bytes)||detail::split_overlap(scratch.data,required,output,output_bytes)||
               detail::split_overlap(scratch.data,required,w.data,weight_bytes))
                throw std::invalid_argument("Q6 byte-code three-part service requires distinct scratch");
            direct_project<Quant::Q6K,true,true,true,true,true,false,false,true><<<dim3((w.rows+Rows-1)/Rows,1,3),Threads,detail::direct_shared_bytes<Quant::Q6K,true,true>(2),stream>>>(w,input,static_cast<float *>(scratch.data),32,1);
            check_cuda(cudaGetLastError(),"owned byte-code Q6 three-part projection");
            const size_t vectors=output_bytes/sizeof(float4);
            direct_split_reduce<<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(scratch.data),output,vectors);
            check_cuda(cudaGetLastError(),"owned byte-code Q6 three-part reduction");return;
        }
        direct_project<Quant::Q6K,true,true,true,false,true,false,false,true><<<dim3((w.rows+Rows-1)/Rows),Threads,detail::direct_shared_bytes<Quant::Q6K,true,true>(2),stream>>>(w,input,output,32,1);
        check_cuda(cudaGetLastError(),"owned complete byte-code Q6 projection");return;
    }
    if(allow_split&&half_raw_slices(w,processors)==2){
        const size_t required=2*output_bytes;
        if(!scratch.data||scratch.bytes<required||reinterpret_cast<uintptr_t>(scratch.data)%16||
           detail::split_overlap(scratch.data,required,input,input_bytes)||detail::split_overlap(scratch.data,required,output,output_bytes)||
           detail::split_overlap(scratch.data,required,w.data,weight_bytes))
            throw std::invalid_argument("raw Q6 two-part projection requires distinct caller scratch");
        direct_project<Quant::Q6K,true,true,false,true,true,false,false,true,2><<<dim3((w.rows+Rows-1)/Rows,1,2),Threads,detail::direct_shared_bytes<Quant::Q6K,false,true>(2),stream>>>(w,input,static_cast<float *>(scratch.data),32,1);
        check_cuda(cudaGetLastError(),"owned raw Q6 two-part projection");
        const size_t vectors=output_bytes/sizeof(float4);
        direct_split2_reduce<<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(scratch.data),output,vectors);
        check_cuda(cudaGetLastError(),"owned raw Q6 two-part reduction");return;
    }
    const int tiles=(w.rows+127)/128,waves=(tiles+processors-1)/processors;
    const int units=100*tiles/(processors*waves)>=90?tiles:processors;
    const dim3 grid((w.rows+Rows-1)/Rows);
    if(w.quant==Quant::Q4K){
        if(units==tiles)direct_project<Quant::Q4K,true,true,false,false,true,false,false,true><<<grid,Threads,detail::direct_shared_bytes<Quant::Q4K,false,true>(2),stream>>>(w,input,output,32,units);
        else direct_project<Quant::Q4K,true,false,false,false,true,false,false,true><<<grid,Threads,detail::direct_shared_bytes<Quant::Q4K,false,true>(2),stream>>>(w,input,output,32,units);
    }else{
        if(units==tiles)direct_project<Quant::Q6K,true,true,false,false,true,false,false,true><<<grid,Threads,detail::direct_shared_bytes<Quant::Q6K,false,true>(2),stream>>>(w,input,output,32,units);
        else direct_project<Quant::Q6K,true,false,false,false,true,false,false,true><<<grid,Threads,detail::direct_shared_bytes<Quant::Q6K,false,true>(2),stream>>>(w,input,output,32,units);
    }
    check_cuda(cudaGetLastError(),"owned raw weight half-input projection");
}

void small_prefill_q6_planes_linear(Q6PlaneMatrix planes,const void *input,float *output,cudaStream_t stream){
    Matrix weight{planes.data,Quant::Q6K,planes.rows,planes.columns};
    const size_t raw_bytes=matrix_bytes(weight),wb=planes.bytes,ib=small_prefill_half_input_bytes(weight.columns,32),ob=size_t(weight.rows)*32*4;
    const bool byte_planes=wb!=raw_bytes;
    if(!split_sm70||code_words!=2||(byte_planes&&wb!=q6_byte_planes_bytes(weight))||weight.rows%64||!input||!output||
       reinterpret_cast<uintptr_t>(planes.data)%16||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(output,ob,input,ib)||detail::split_overlap(output,ob,planes.data,wb)||detail::split_overlap(input,ib,planes.data,wb))
        throw std::invalid_argument("Q6 plane projection requires complete distinct aligned PP32 extents");
    if(byte_planes)detail::prefetch_q6_project<true,1,true,true><<<weight.rows/64,256,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),output);
    else detail::prefetch_q6_project<true,1,true><<<weight.rows/64,256,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),output);
    check_cuda(cudaGetLastError(),"owned Q6 plane projection");
}

void small_prefill_q6_register_linear(Matrix weight,const void *input,float *output,cudaStream_t stream,bool prefetch) {
    const size_t wb=matrix_bytes(weight,true),ib=small_prefill_half_input_bytes(weight.columns,32),ob=size_t(weight.rows)*32*4;
    if(!split_sm70||code_words!=2||weight.quant!=Quant::Q6K||weight.layout!=BlockLayout::Q6Codes||weight.rows%64||
       !input||!output||reinterpret_cast<uintptr_t>(weight.data)%16||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(output,ob,input,ib)||detail::split_overlap(output,ob,weight.data,wb)||detail::split_overlap(input,ib,weight.data,wb))
        throw std::invalid_argument("register Q6 requires distinct aligned byte-code PP32 inventory");
    if(prefetch)detail::prefetch_q6_project<false><<<weight.rows/64,256,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),output);
    else detail::register_q6_project<<<weight.rows/64,128,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),output);
    check_cuda(cudaGetLastError(),"owned register Q6 projection");
}

void small_prefill_q6_raw_prefetch_linear(Matrix weight,const void *input,float *output,cudaStream_t stream,RawHalfScratch scratch){
    const size_t wb=matrix_bytes(weight),ib=small_prefill_half_input_bytes(weight.columns,32),ob=size_t(weight.rows)*32*4;
    if(!split_sm70||code_words!=2||weight.quant!=Quant::Q6K||weight.layout!=BlockLayout::Gguf||weight.rows%64||
       !input||!output||reinterpret_cast<uintptr_t>(weight.data)%16||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(output,ob,input,ib)||detail::split_overlap(output,ob,weight.data,wb)||detail::split_overlap(input,ib,weight.data,wb))
        throw std::invalid_argument("raw prefetch Q6 requires distinct aligned PP32 inventory");
    if(scratch.data){
        const size_t required=2*ob;
        if(weight.columns%512||scratch.bytes<required||reinterpret_cast<uintptr_t>(scratch.data)%16||
           detail::split_overlap(scratch.data,required,input,ib)||detail::split_overlap(scratch.data,required,output,ob)||detail::split_overlap(scratch.data,required,weight.data,wb))
            throw std::invalid_argument("raw prefetch Q6 split requires distinct aligned full partial arena");
        detail::prefetch_q6_project<true,2><<<dim3(weight.rows/64,1,2),256,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),static_cast<float *>(scratch.data));
        check_cuda(cudaGetLastError(),"owned raw prefetch Q6 split");
        const size_t vectors=ob/sizeof(float4);
        direct_split2_reduce<<<unsigned((vectors+255)/256),256,0,stream>>>(static_cast<const float *>(scratch.data),output,vectors);
        check_cuda(cudaGetLastError(),"owned raw prefetch Q6 split reduction");return;
    }
    detail::prefetch_q6_project<true><<<weight.rows/64,256,0,stream>>>(weight,static_cast<const detail::HalfInputBlock *>(input),output);
    check_cuda(cudaGetLastError(),"owned raw prefetch Q6 projection");
}

}
