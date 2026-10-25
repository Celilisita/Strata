#include "page_row.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
namespace strata::qwen35::native { namespace {
__global__ void page_row_kernel(const float *query,RamKvView v,const int *positions,int shift,int heads){
    const int valid=positions[0]-shift+1,page=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
    const int live=(valid+RamPage-1)/RamPage,groups=heads/v.heads;
    const size_t i=size_t(h)*v.pages+page;
    if(page>=live){if(!d)v.scores[i]=-CUDART_INF_F;return;}
    if(page<4||page>=live-64){if(!d)v.scores[i]=CUDART_INF_F;return;}
    const float lo=__half2float(v.bounds[i*512+d]),hi=__half2float(v.bounds[i*512+256+d]);
    __shared__ float partial[6][8];
    for(int q=0;q<groups;++q){
        const float x=query[size_t(h*groups+q)*256+d];float score=fmaxf(x*lo,x*hi);
        for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);
        if(!(d%32))partial[q][d/32]=score;
    }
    __syncthreads();
    if(d<32){
        float result=-CUDART_INF_F;
        for(int q=0;q<groups;++q){
            float score=d<8?partial[q][d]:0.f;
            for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);
            if(!d)result=fmaxf(result,score);
        }
        if(!d)v.scores[i]=result;
    }
}
bool same_view(RamKvView a,RamKvView b){
    #define SAME(x) if(a.x!=b.x)return false
    SAME(host_key);SAME(host_value);SAME(key);SAME(value);SAME(host_capacity);SAME(resident);SAME(heads);SAME(pages);
    SAME(slots);SAME(selected_tokens);SAME(selected_slots);SAME(keep_queries);SAME(bounds);SAME(scores);SAME(sorted_scores);
    SAME(page_ids);SAME(sorted_ids);SAME(offsets);SAME(page_slot);SAME(slot_page);SAME(selected_slot);SAME(miss);SAME(marked);
    SAME(miss_rank);SAME(free_slot);SAME(counters);SAME(last_use);SAME(epoch);SAME(sort_work);SAME(sort_bytes);
    #undef SAME
    return true;
}
void geometry(RamKvView v){
    if(v.resident!=40960||v.host_capacity!=40960||v.pages!=2560||v.slots!=2560||v.heads!=4||v.selected_tokens!=8192||v.selected_slots!=512||v.keep_queries!=3||
       !v.key||!v.value||v.key!=v.host_key||v.value!=v.host_value||!v.bounds||!v.scores||!v.epoch)
        throw std::invalid_argument("page-row exact current195 authority/geometry");
}
}
PageRowResources page_row_resources(){
    cudaFuncAttributes a{};int blocks=0,device=0;cudaDeviceProp p{};
    check_cuda(cudaGetDevice(&device),"page-row device");check_cuda(cudaGetDeviceProperties(&p,device),"page-row properties");
    check_cuda(cudaFuncGetAttributes(&a,page_row_kernel),"page-row attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,page_row_kernel,256,0),"page-row occupancy");
    return {a.numRegs,int(a.sharedSizeBytes),int(a.localSizeBytes),a.binaryVersion,blocks,p.maxThreadsPerMultiProcessor/256};
}
void page_row_scores(const float*q,RamKvView v,const int*p,int shift,int heads,cudaStream_t stream){
    geometry(v);if(!q||!p||heads!=24||shift!=0||reinterpret_cast<uintptr_t>(q)%4||reinterpret_cast<uintptr_t>(p)%4)
        throw std::invalid_argument("page-row query contract");
    page_row_kernel<<<dim3(v.pages,v.heads),256,0,stream>>>(q,v,p,shift,heads);
    check_cuda(cudaGetLastError(),"page-row scores");
}
bool page_row_patch(cudaGraphNode_t node,RamKvView expected,const float*baseq,const int*basep,int tokens,bool install,PageRowControl&control){
    geometry(expected);if(!baseq||!basep||tokens<1||tokens>3)throw std::invalid_argument("page-row patch range");
    cudaGraphNodeType type{};check_cuda(cudaGraphNodeGetType(node,&type),"page-row node type");if(type!=cudaGraphNodeTypeKernel)return false;
    cudaKernelNodeParams p{};check_cuda(cudaGraphKernelNodeGetParams(node,&p),"page-row retained params");
    if(p.gridDim.x!=2560||p.gridDim.y!=4||p.gridDim.z!=1||p.blockDim.x!=256||p.blockDim.y!=1||p.blockDim.z!=1)return false;
    if(!p.kernelParams||p.extra||p.sharedMemBytes)throw std::invalid_argument("page-row retained ABI");
    const float*q=*static_cast<const float**>(p.kernelParams[0]);RamKvView v=*static_cast<RamKvView*>(p.kernelParams[1]);
    const int*pos=*static_cast<const int**>(p.kernelParams[2]);int shift=*static_cast<int*>(p.kernelParams[3]),heads=*static_cast<int*>(p.kernelParams[4]);
    const uintptr_t qb=reinterpret_cast<uintptr_t>(baseq),qa=reinterpret_cast<uintptr_t>(q),pb=reinterpret_cast<uintptr_t>(basep),pa=reinterpret_cast<uintptr_t>(pos);
    const size_t stride=size_t(24)*256*sizeof(float);
    if(!same_view(v,expected)||qa<qb||qa-qb>=stride*tokens||(qa-qb)%stride||pa!=pb+(qa-qb)/stride*sizeof(int)||shift||heads!=24)
        throw std::invalid_argument("page-row exact retained argument identity");
    cudaFuncAttributes a{};int blocks=0;check_cuda(cudaFuncGetAttributes(&a,p.func),"page-row original attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,p.func,256,0),"page-row original occupancy");
    const auto candidate=page_row_resources();
    if(a.numRegs!=31||a.sharedSizeBytes!=32||a.localSizeBytes||candidate.blocks_per_sm<blocks)
        throw std::runtime_error("page-row occupancy/current195 resource identity");
    control={a.numRegs,int(a.sharedSizeBytes),int(a.localSizeBytes),blocks};
    if(install){void*args[]={&q,&v,&pos,&shift,&heads};p.func=reinterpret_cast<void*>(page_row_kernel);p.kernelParams=args;p.extra=nullptr;
        check_cuda(cudaGraphKernelNodeSetParams(node,&p),"page-row replace score node only");}
    return true;
}
}
