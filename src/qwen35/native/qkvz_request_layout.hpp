#pragma once
#include "strata/qwen35/native/request_layout.hpp"
#include "fp4_compensated_service.hpp"
#include "fp4_unpack_vector.hpp"
#include <chrono>
namespace strata::qwen35::native {
// The published Matrix tag is immutable. Only the Runtime inventory owns physical layout.
class QkvzRequestLayout {
    bool failed_=false;std::uint64_t epoch_=0;
    static bool desired(int prompt){if(prompt<1)throw std::invalid_argument("QKVZ request length");return prompt<=8192;}
public:
    void validate()const{if(failed_)throw std::runtime_error("QKVZ layout owner is failed; reload required");}
    template<class Inventory> bool change_required(const Inventory &inventory,int prompt)const{
        validate();const bool target=desired(prompt);
        for(const auto &item:inventory)if(item.second.role&&item.second.packed!=target)return true;
        return false;
    }
    template<class Inventory> void prepare(Inventory &inventory,void *scratch,size_t capacity,
            cudaStream_t owner,cudaStream_t stream,int prompt,bool cleared){
        validate();if(stream!=owner)throw std::invalid_argument("QKVZ layout stream owner");
        auto &stats=qkvz_request_layout_stats();stats={};stats.owner=reinterpret_cast<std::uintptr_t>(this);
        stats.scratch_bytes=capacity;stats.epoch=epoch_;const bool target=desired(prompt);stats.physical_mode=target?1:0;
        cudaStreamCaptureStatus capture{};check_cuda(cudaStreamIsCapturing(stream,&capture),"QKVZ boundary capture status");
        if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("QKVZ conversion cannot be captured");
        size_t q=0,z=0;bool change=false;
        for(const auto &item:inventory){const auto &e=item.second;if(!e.role)continue;
            const auto &w=e.view;const size_t bytes=nvfp4_value_bytes(w.rows,w.columns);
            if(!qkvz_packed_shape(w)||(e.role==1?w.rows!=10240:e.role!=2||w.rows!=6144)||!e.declared_packed||
               !scratch||bytes>capacity||!w.values||!w.scales||w.values==scratch)
                throw std::logic_error("QKVZ registered role/arena boundary");
            e.role==1?++q:++z;change|=e.packed!=target;
        }
        if(!q&&!z){stats.physical_mode=0;return;}
        if(!q||q!=z)throw std::logic_error("QKVZ incomplete exact role inventory");
        if(!change)return;
        if(!cleared)throw std::logic_error("QKVZ layout change requires cleared graphs");
        stats.oldgraphs_cleared=true;const auto begin=std::chrono::steady_clock::now();
        try{
            check_cuda(cudaStreamSynchronize(stream),"QKVZ old execution drain");
            for(auto &item:inventory){auto &e=item.second;if(!e.role||e.packed==target)continue;
                const size_t bytes=nvfp4_value_bytes(e.view.rows,e.view.columns);
                if(target)fp4_pack_values(e.view,scratch,stream);else fp4_unpack_values({e.view},scratch,stream);
                check_cuda(cudaMemcpyAsync(const_cast<void*>(e.view.values),scratch,bytes,cudaMemcpyDeviceToDevice,stream),"QKVZ same-owner values copyback");
                ++stats.matrix_count;stats.values_bytes+=bytes;stats.traffic_bytes+=4*bytes;
            }
            check_cuda(cudaStreamSynchronize(stream),"QKVZ layout publication drain");
            for(auto &item:inventory)if(item.second.role)item.second.packed=target;
            ++epoch_;stats.epoch=epoch_;stats.conversions=1;
        }catch(...){failed_=true;stats.failed=true;cudaStreamSynchronize(stream);throw;}
        stats.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    }
};
inline Nvfp4View qkvz_canonical_view(Nvfp4View w,bool materialize,void *scratch,size_t capacity,bool head,
        bool vector,std::uint64_t &vector_calls,CompensatedFfnCounters &counters,cudaStream_t stream){
    if(!materialize)return w;
    if(head||!scratch||nvfp4_value_bytes(w.rows,w.columns)>capacity)throw std::logic_error("packed canonical staging capacity/head lease");
    if(vector){fp4_unpack_vector({w},scratch,stream);++vector_calls;}else fp4_unpack_values({w},scratch,stream);
    w.values=scratch;qkvz_record_unpack(w);++counters.materialize_submissions;return w;
}
}
