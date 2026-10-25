#include "strata/prefill/native_grouped_nvfp4.hpp"
#include "native_grouped_mmq_plan.hpp"
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
namespace strata::prefill::mmq {
namespace {
constexpr size_t Capacity=256*sizeof(NativeNvfp4Matrix)+65536*8+(65536/32+256)*sizeof(NativeMmqTile);
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void view(const void* p,size_t bytes,size_t alignment){const auto a=reinterpret_cast<uintptr_t>(p);
    if(!p||a%alignment||bytes>(std::numeric_limits<uintptr_t>::max)()-a)throw std::invalid_argument("NVFP4 provider view");}
void disjoint(const void* a,size_t an,const void* b,size_t bn){const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(x<y+bn&&y<x+an)throw std::invalid_argument("NVFP4 provider workspace/weight alias");}
struct Lease {
    uint8_t* data=nullptr;cudaEvent_t done{};bool used=false;
    Lease(){ck(cudaMallocHost(reinterpret_cast<void**>(&data),Capacity));
        try{ck(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));}catch(...){cudaFreeHost(data);throw;}}
    ~Lease(){if(used)cudaEventSynchronize(done);if(done)cudaEventDestroy(done);if(data)cudaFreeHost(data);}
};
}
struct NativeGroupedNvfp4Provider::Impl {
    cudaStream_t stream;std::array<Lease,4> leases;size_t next=0;
    explicit Impl(void* s):stream(static_cast<cudaStream_t>(s)){
        if(!stream)throw std::invalid_argument("NVFP4 provider requires ordered stream");
        int device=0,major=0,minor=0;ck(cudaGetDevice(&device));
        ck(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
        ck(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device));
        if(major!=12||minor!=0)throw std::invalid_argument("NVFP4 provider requires SM120");
    }
};
NativeGroupedNvfp4Provider::NativeGroupedNvfp4Provider(void* s):impl_(std::make_unique<Impl>(s)){}
NativeGroupedNvfp4Provider::~NativeGroupedNvfp4Provider()=default;
NativeGroupedMmqWork NativeGroupedNvfp4Provider::run_input(const unsigned long long* gate,const unsigned long long* up,
    const float* gate_scales,const float* up_scales,const int32_t* expert_ids,const int32_t* source_ids,
    int entries,int source_rows,const float* input,void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
    float* output,int columns,int outputs,int tile_rows,int wave,
    const std::function<void(const NativeNvfp4Matrix*,int,bool)>& ready){
    if(wave&&(!up||!ready||wave!=32||columns!=2048||outputs!=512||
              (source_rows!=1280&&source_rows!=4096)||entries!=source_rows*8))throw std::invalid_argument("bounded NVFP4 GU pipeline domain");
    if(!gate_scales||!expert_ids||bool(up)!=bool(up_scales)||(tile_rows!=32&&tile_rows!=64)||
       outputs<16||outputs>8192||outputs%16)throw std::invalid_argument("NVFP4 role/scales/tile geometry");
    const auto need=native_nvfp4_input_bytes(source_rows,columns);
    // Existing group directory and full original-entry maps; only tile count is parameterized here.
    auto plan=native_mmq_plan(gate,up,source_ids,entries,source_rows,false);
    std::vector<NativeNvfp4Matrix> matrices(plan.matrices.size());
    std::unordered_map<uint64_t,size_t> directory;directory.reserve(matrices.size());
    for(size_t i=0;i<matrices.size();++i){const auto& m=plan.matrices[i];directory.emplace(m.gate,i);
        matrices[i]={m.gate,m.up,1,1,m.begin,m.end,-1,0};
        for(int t=0;t<(m.end-m.begin+tile_rows-1)/tile_rows;++t)plan.tiles.push_back({int(i),t});}
    for(int e=0;e<entries;++e){const int id=expert_ids[e];
        if(id<0||id>=256||!(gate_scales[e]>0)||!std::isfinite(gate_scales[e])||
           (up&&(!(up_scales[e]>0)||!std::isfinite(up_scales[e]))))throw std::invalid_argument("NVFP4 expert scale/key");
        auto& m=matrices.at(directory.at(gate[e]));
        if(m.expert_id<0){m.expert_id=id;m.gate_scale=gate_scales[e];m.up_scale=up?up_scales[e]:1;}
        else if(m.expert_id!=id||m.gate_scale!=gate_scales[e]||(up&&m.up_scale!=up_scales[e]))
            throw std::invalid_argument("NVFP4 grouped expert/global identity mismatch");
    }
    const size_t mb=matrices.size()*sizeof(NativeNvfp4Matrix),maps=size_t(entries)*4,tb=plan.tiles.size()*sizeof(NativeMmqTile);
    const size_t cb=mb+2*maps+tb,ib=size_t(source_rows)*columns*4,ob=size_t(up?2:1)*entries*outputs*4;
    if(cb>controls_bytes||cb>Capacity||need>packed_bytes)throw std::invalid_argument("unreserved NVFP4 provider workspace");
    view(input,ib,4);view(controls,cb,16);view(packed,need,16);view(output,ob,4);
    const std::array<std::pair<const void*,size_t>,4> spans{{{input,ib},{controls,cb},{packed,need},{output,ob}}};
    for(size_t i=0;i<spans.size();++i)for(size_t j=0;j<i;++j){
        if(i==3&&j==0)continue; // Encode consumes all raw input before same-stream product writes output.
        disjoint(spans[i].first,spans[i].second,spans[j].first,spans[j].second);
    }
    const size_t weight_bytes=size_t(columns/64)*outputs*36;
    for(const auto& m:matrices)for(int p=0;p<(up?2:1);++p){const auto* w=reinterpret_cast<const void*>(p?m.up:m.gate);
        view(w,weight_bytes,4);for(const auto& s:spans)disjoint(w,weight_bytes,s.first,s.second);}
    auto& owner=*impl_;auto& lease=owner.leases[owner.next++%owner.leases.size()];
    if(lease.used){const auto ready=cudaEventQuery(lease.done);
        if(ready==cudaErrorNotReady)throw std::runtime_error("NVFP4 control lease requires caller drain");ck(ready);}
    std::memcpy(lease.data,matrices.data(),mb);std::memcpy(lease.data+mb,plan.input_ids.data(),maps);
    std::memcpy(lease.data+mb+maps,plan.output_ids.data(),maps);std::memcpy(lease.data+mb+2*maps,plan.tiles.data(),tb);
    auto* device=static_cast<uint8_t*>(controls);
    try{
        ck(cudaMemcpyAsync(device,lease.data,cb,cudaMemcpyHostToDevice,owner.stream));
        const auto encoded=encode_native_nvfp4(input,packed,packed_bytes,source_rows,columns,
            reinterpret_cast<int32_t*>(device+mb),owner.stream);
        size_t tile_begin=0;
        for(size_t first=0;first<matrices.size();first+=wave?size_t(wave):matrices.size()){
            const size_t end=wave?std::min(first+size_t(wave),matrices.size()):matrices.size();
            size_t tile_end=tile_begin;
            while(tile_end<plan.tiles.size()&&size_t(plan.tiles[tile_end].matrix)<end)++tile_end;
            if(tile_end==tile_begin)throw std::runtime_error("empty NVFP4 GU wave");
            if(wave)ready(matrices.data()+first,int(end-first),end==matrices.size());
            native_grouped_nvfp4(reinterpret_cast<NativeNvfp4Matrix*>(device),int(matrices.size()),up?2:1,
                reinterpret_cast<int32_t*>(device+mb+maps),output,entries,plan.max_rows,outputs,
                reinterpret_cast<NativeMmqTile*>(device+mb+2*maps)+tile_begin,int(tile_end-tile_begin),tile_rows,encoded,owner.stream);
            tile_begin=tile_end;
        }
        if(tile_begin!=plan.tiles.size())throw std::runtime_error("NVFP4 GU tile coverage");
        ck(cudaEventRecord(lease.done,owner.stream));lease.used=true;
    }catch(...){const auto e=cudaEventRecord(lease.done,owner.stream);lease.used=e==cudaSuccess;
        if(e!=cudaSuccess)cudaStreamSynchronize(owner.stream);throw;}
    return{cb,need,int(matrices.size()),1+int(wave?(matrices.size()+size_t(wave)-1)/size_t(wave):1),int(plan.tiles.size()),int(plan.tiles.size())};
}
NativeGroupedMmqWork NativeGroupedNvfp4Provider::run(const unsigned long long* gate,const unsigned long long* up,
    const float* gs,const float* us,const int32_t* ids,const int32_t* sources,int entries,int rows,const float* input,
    void* controls,size_t control_bytes,void* packed,size_t packed_bytes,float* output,int k,int n,int tile){
    return run_input(gate,up,gs,us,ids,sources,entries,rows,input,controls,control_bytes,packed,packed_bytes,output,k,n,tile,0,{});
}
NativeGroupedMmqWork NativeGroupedNvfp4Provider::run_gu_pipeline(const unsigned long long* gate,const unsigned long long* up,
    const float* gs,const float* us,const int32_t* ids,const int32_t* sources,int entries,int rows,const float* input,
    void* controls,size_t control_bytes,void* packed,size_t packed_bytes,float* output,int wave,
    const std::function<void(const NativeNvfp4Matrix*,int,bool)>& ready){
    return run_input(gate,up,gs,us,ids,sources,entries,rows,input,controls,control_bytes,packed,packed_bytes,output,2048,512,64,wave,ready);
}

}
