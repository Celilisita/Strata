#include "prefix_copy.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {namespace {
constexpr int Threads=128;
thread_local std::uint64_t host_submissions=0;
constexpr size_t StateVectors=PrefixCopyStateBytes/sizeof(uint4),HistoryVectors=PrefixCopyHistoryBytes/sizeof(uint4);
__global__ void restore(const __grid_constant__ PrefixCopyBatch batch){
    const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;if(i>=StateVectors+HistoryVectors)return;
    const auto r=batch.rows[blockIdx.y];
    if(i<StateVectors)static_cast<uint4*>(r.state)[i]=static_cast<const uint4*>(r.prefix_state)[i];
    else static_cast<uint4*>(r.history)[i-StateVectors]=static_cast<const uint4*>(r.prefix_history)[i-StateVectors];
}
struct Range {uintptr_t first,last;};
Range range(const void *p,size_t bytes){const auto a=reinterpret_cast<uintptr_t>(p);
    if(!p||a%16||bytes>std::numeric_limits<uintptr_t>::max()-a)throw std::invalid_argument("prefix copy alignment/extent");return {a,a+bytes};}
}
PrefixCopyResources prefix_copy_resources(){cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,restore),"prefix copy attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,restore,Threads,0),"prefix copy occupancy");
    return {a.numRegs,blocks,a.binaryVersion,a.sharedSizeBytes,a.localSizeBytes,sizeof(PrefixCopyBatch)};
}
void configure_prefix_copy(){int device=0;check_cuda(cudaGetDevice(&device),"prefix copy device");const auto c=capabilities(device);const auto r=prefix_copy_resources();
    if(c.major!=12||c.minor!=0||r.binary_version!=120||r.blocks_per_sm<2||r.local_bytes||r.parameter_bytes>4096)
        throw std::runtime_error("prefix copy requires qualified SM120 zero-local2CTA parameter ABI");
}
bool prefix_copy_enabled(){static const bool enabled=[](){const char *raw=std::getenv("STRATA_DENSE_FUSED_PREFIX_COPY");const std::string mode=raw?raw:"0";
    if(mode!="0"&&mode!="1")throw std::invalid_argument("prefix copy mode must be0or1");if(mode=="0")return false;
    int device=0;check_cuda(cudaGetDevice(&device),"prefix copy admission device");const auto c=capabilities(device);
    if(c.major!=12||c.minor!=0)return false;configure_prefix_copy();return true;}();return enabled;}
void validate_prefix_copy(const PrefixCopyBatch &batch){std::array<Range,PrefixCopyLayers*4> ranges{};size_t n=0;
    for(const auto &r:batch.rows){ranges[n++]=range(r.state,PrefixCopyStateBytes);ranges[n++]=range(r.prefix_state,PrefixCopyStateBytes);
        ranges[n++]=range(r.history,PrefixCopyHistoryBytes);ranges[n++]=range(r.prefix_history,PrefixCopyHistoryBytes);}
    for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j)if(ranges[i].first<ranges[j].last&&ranges[j].first<ranges[i].last)
        throw std::invalid_argument("prefix copy independent state/history leases overlap");
}
std::uint64_t prefix_copy_host_submissions(){return host_submissions;}
void prefix_copy_restore(PrefixCopyBatch batch,cudaStream_t stream){validate_prefix_copy(batch);
    restore<<<dim3(unsigned((StateVectors+HistoryVectors+Threads-1)/Threads),PrefixCopyLayers),Threads,0,stream>>>(batch);
    check_cuda(cudaGetLastError(),"one raw-bit retained prefix commit");++host_submissions;
}
}
