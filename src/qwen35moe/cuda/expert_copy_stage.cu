#include "strata/qwen35moe/expert_copy_stage_cuda.hpp"
#include <limits>
#include <stdexcept>
namespace strata::qwen35moe {
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
// CUDA12.8 const grid parameters avoid per-thread copies of the512-byte table.
__global__ void scatter_kernel(const __grid_constant__ ExpertStageTargets targets,
    const uint4* source,unsigned gate_words,unsigned down_words,unsigned experts){
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned gu_area=gate_words*experts,total=2*gu_area+down_words*experts;
    if(index>=total)return;
    unsigned local,width,offset;
    if(index<gu_area){local=index;width=gate_words;offset=0;}
    else if(index<2*gu_area){local=index-gu_area;width=gate_words;offset=gate_words;}
    else{local=index-2*gu_area;width=down_words;offset=2*gate_words;}
    const unsigned expert=local/width,within=local%width;
    auto* output=reinterpret_cast<uint4*>(targets.addresses[expert]);
    output[offset+within]=source[index];
}
}
void expert_stage_scatter(const void* stage,size_t stage_bytes,const ExpertStageTargets& targets,
                          size_t gu,size_t down,unsigned count,cudaStream_t stream){
    const auto limit=(std::numeric_limits<uintptr_t>::max)();const auto start=uintptr_t(stage);
    if(!start||start%16||!stream||!count||count>64||!gu||!down||gu%16||down%16||
       gu>1u<<20||down>1u<<20||2*gu+down>stage_bytes/count||
       stage_bytes>limit-start)throw std::invalid_argument("invalid expert staging scatter extent");
    const size_t bytes=2*gu+down,used=bytes*count;
    for(unsigned i=0;i<count;++i){const auto dst=targets.addresses[i];
        if(!dst||dst%16||bytes>limit-dst||(dst<start+used&&start<dst+bytes))
            throw std::invalid_argument("expert staging/cache alias or alignment");
        for(unsigned j=0;j<i;++j)if(dst<targets.addresses[j]+bytes&&targets.addresses[j]<dst+bytes)
            throw std::invalid_argument("expert staging duplicate destination");
    }
    const unsigned words=unsigned(used/16);
    scatter_kernel<<<(words+255)/256,256,0,stream>>>(targets,static_cast<const uint4*>(stage),unsigned(gu/16),unsigned(down/16),count);
    ck(cudaGetLastError());
}
ExpertStageResources expert_stage_resources(){
    cudaFuncAttributes a{};ck(cudaFuncGetAttributes(&a,scatter_kernel));int active=0;
    ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,scatter_kernel,256,0));
    return{a.numRegs,a.maxThreadsPerBlock,active,a.localSizeBytes,a.sharedSizeBytes};
}
}
