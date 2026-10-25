#pragma once
#include <string>
#include "strata/qwen35/completed_v100/native/model.hpp"
#include "pp_gqa_borrowed.hpp"
namespace strata::completed_v100::qwen35::native {
class PpLongLibraryOption {
public:
    void select(std::string& mode){selected_=mode=="Q8A_P2LIB";if(selected_)mode="Q8A_P2VEC";}
    void reset(){selected_=false;}
    bool selected()const{return selected_;}
    unsigned long long recorded()const{return adapter_?adapter_->recorded():0;}
    void apply(DenseModel& model,DenseGemm& service,DeviceStorage& storage,const strata::completed_v100::qwen35::Spec& spec){
        auto& attention=model.attention_work();
        if(adapter_)adapter_->reset();
        if(!selected_){attention.use_pp_blocked(nullptr,nullptr,0);return;}
        if(spec.n_head!=24||spec.n_head_kv!=4||spec.head_dim!=256)throw std::invalid_argument("PP GQA confirmed shape");
        if(!adapter_){
            const auto r=pp_gqa::resources();
            if(r.local_bytes||r.registers>64||r.blocks<1)throw std::runtime_error("PP GQA resource gate");
            PrefillAttentionShape bound{256,24,4,8192,24576,32768,32768,1.f/16.f,{}};bound.q8=true;
            bytes_=pp_gqa::workspace(bound,128);arena_=storage.allocate(bytes_);
            adapter_=std::make_unique<PpGqaBorrowed>(service,arena_,bytes_);
        }
        // Native pp_kernel remains the fallback for every unsupported prefix.
        attention.use_pp_blocked(adapter_.get(),arena_,bytes_,128,24576);
    }
private:
    bool selected_=false;void* arena_=nullptr;size_t bytes_=0;
    std::unique_ptr<PpGqaBorrowed> adapter_;
};
}
