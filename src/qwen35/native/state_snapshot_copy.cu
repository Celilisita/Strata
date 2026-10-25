#include "state_snapshot_copy.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {namespace {
constexpr int Threads=128;
constexpr size_t Vectors=StateSnapshotCopyBytes/sizeof(uint4);
thread_local std::uint64_t submissions=0;
__global__ void snapshot(const __grid_constant__ StateSnapshotCopyBatch batch){const size_t i=size_t(blockIdx.x)*Threads+threadIdx.x;
    if(i<Vectors){const auto r=batch.rows[blockIdx.y];static_cast<uint4*>(r.saved_state)[i]=static_cast<const uint4*>(r.state)[i];}}
struct Range {uintptr_t first,last;};
Range range(const void *p){const auto a=reinterpret_cast<uintptr_t>(p);if(!p||a%16||StateSnapshotCopyBytes>std::numeric_limits<uintptr_t>::max()-a)throw std::invalid_argument("state snapshot range/alignment");return {a,a+StateSnapshotCopyBytes};}
}
StateSnapshotCopyResources state_snapshot_copy_resources(){cudaFuncAttributes a{};int blocks=0;check_cuda(cudaFuncGetAttributes(&a,snapshot),"state snapshot attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,snapshot,Threads,0),"state snapshot occupancy");return {a.numRegs,blocks,a.binaryVersion,a.sharedSizeBytes,a.localSizeBytes,sizeof(StateSnapshotCopyBatch)};}
void configure_state_snapshot_copy(){int device=0;check_cuda(cudaGetDevice(&device),"state snapshot device");const auto c=capabilities(device);const auto r=state_snapshot_copy_resources();
    if(c.major!=12||c.minor!=0||r.binary_version!=120||r.blocks_per_sm<2||r.local_bytes||r.parameter_bytes>4096)throw std::runtime_error("state snapshot requires SM120 zero-local2CTA ABI");}
bool state_snapshot_copy_enabled(){static const bool enabled=[](){const char *raw=std::getenv("STRATA_DENSE_FUSED_STATE_SNAPSHOT_COPY");const std::string mode=raw?raw:"0";
    if(mode!="0"&&mode!="1")throw std::invalid_argument("state snapshot mode must be0or1");if(mode=="0")return false;
    int device=0;check_cuda(cudaGetDevice(&device),"state snapshot admission device");const auto c=capabilities(device);if(c.major!=12||c.minor!=0)return false;configure_state_snapshot_copy();return true;}();return enabled;}
void validate_state_snapshot_copy(const StateSnapshotCopyBatch &batch){std::array<Range,StateSnapshotCopyLayers*2> ranges{};size_t n=0;
    for(const auto &r:batch.rows){ranges[n++]=range(r.saved_state);ranges[n++]=range(r.state);}
    for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j)if(ranges[i].first<ranges[j].last&&ranges[j].first<ranges[i].last)throw std::invalid_argument("state snapshot independent leases overlap");}
std::uint64_t state_snapshot_copy_host_submissions(){return submissions;}
void state_snapshot_copy(StateSnapshotCopyBatch batch,cudaStream_t stream){validate_state_snapshot_copy(batch);
    snapshot<<<dim3(unsigned((Vectors+Threads-1)/Threads),StateSnapshotCopyLayers),Threads,0,stream>>>(batch);check_cuda(cudaGetLastError(),"one raw-bit retained state snapshot");++submissions;}
}
