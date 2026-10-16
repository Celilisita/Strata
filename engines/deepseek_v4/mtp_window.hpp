#pragma once
#include "mtp_block_graph.hpp"
#include "mtp_embedding.hpp"
#include "ggml-alloc.h"
#include <map>
#include <memory>
#include <stdexcept>
namespace strata::deepseek_v4 {
// The uncompressed MTP window is independent of the target's compressed KV.
// Only accepted target inputs are appended; draft outputs never enter this KV.
class MtpWindow {
    struct Graph {
        ggml_context* meta=nullptr;ggml_gallocr_t alloc=nullptr;ggml_cgraph* graph=nullptr;
        ggml_tensor *embedded=nullptr,*hidden=nullptr,*ids=nullptr,*positions=nullptr;
        ~Graph(){if(alloc)ggml_gallocr_free(alloc);if(meta)ggml_free(meta);}
    };
    MtpSidecar& weights_;ggml_backend_t backend_;ggml_tensor* embedding_;
    ggml_context* meta_=nullptr;ggml_backend_buffer_t buffer_=nullptr;ggml_tensor* kv_=nullptr;
    std::map<int,std::unique_ptr<Graph>> catchup_;int rows_=0,last_position_=-1;
public:
    MtpWindow(MtpSidecar& weights,ggml_backend_t backend,ggml_tensor* embedding):weights_(weights),backend_(backend),embedding_(embedding){
        meta_=ggml_init({ggml_tensor_overhead(),nullptr,true});if(!meta_)throw std::runtime_error("MTP KV metadata allocation failed");
        kv_=ggml_new_tensor_2d(meta_,GGML_TYPE_F32,512,127);ggml_set_input(kv_);
        buffer_=ggml_backend_alloc_ctx_tensors(meta_,backend_);if(!buffer_)throw std::runtime_error("MTP KV allocation failed");reset();
    }
    ~MtpWindow(){catchup_.clear();if(buffer_)ggml_backend_buffer_free(buffer_);if(meta_)ggml_free(meta_);}
    size_t bytes()const{size_t n=ggml_backend_buffer_get_size(buffer_);for(const auto& entry:catchup_)if(entry.second->alloc)n+=ggml_gallocr_get_buffer_size(entry.second->alloc,0);return n;}
    ggml_tensor* tensor()const{return kv_;}int rows()const{return rows_;}
    void reset(){ggml_backend_buffer_clear(buffer_,0);rows_=0;last_position_=-1;}
    ggml_tensor* append(ggml_context* c,ggml_tensor* all,int count){
        if(all->ne[0]!=512||all->ne[1]!=127+count)throw std::runtime_error("MTP KV append geometry differs");
        return ggml_cpy(c,ggml_view_2d(c,all,512,127,all->nb[1],count*all->nb[1]),kv_);
    }
    void check(const int32_t* positions,int count)const{
        if(!positions||count<1||count>127||positions[0]<0||(rows_&&positions[0]!=last_position_+1))throw std::runtime_error("MTP KV input position discontinuity");
        for(int i=1;i<count;++i)if(positions[i]!=positions[0]+i)throw std::runtime_error("MTP KV positions not consecutive");
    }
    void committed(const int32_t* positions,int count){rows_=std::min(127,rows_+count);last_position_=positions[count-1];}
    void catch_up(const float* hidden,const int32_t* ids,const int32_t* positions,int count){
        check(positions,count);if(!hidden||!ids)throw std::runtime_error("MTP KV catch-up input missing");
        auto& owner=catchup_[count];if(!owner){owner=std::make_unique<Graph>();auto& p=*owner;
            p.meta=ggml_init({1024*ggml_tensor_overhead()+ggml_graph_overhead_custom(1024,false),nullptr,true});if(!p.meta)throw std::runtime_error("MTP catch-up metadata failed");
            auto* c=p.meta;p.hidden=ggml_new_tensor_3d(c,GGML_TYPE_F32,4096,4,count);p.ids=ggml_new_tensor_1d(c,GGML_TYPE_I32,count);p.positions=ggml_new_tensor_1d(c,GGML_TYPE_I32,count);
            for(auto* t:{p.hidden,p.ids,p.positions})ggml_set_input(t);
            p.embedded=mtp_embedding_input(c,embedding_,count);
            auto* next=ggml_concat(c,kv_,mtp_input_kv(c,weights_,embedding_,p.ids,p.hidden,p.positions,p.embedded),1);
            p.graph=ggml_new_graph_custom(c,1024,false);ggml_build_forward_expand(p.graph,append(c,next,count));
            p.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));if(!ggml_gallocr_alloc_graph(p.alloc,p.graph))throw std::runtime_error("MTP catch-up allocation failed");
        }
        auto& p=*owner;ggml_backend_tensor_set(p.hidden,hidden,0,size_t(count)*16384*sizeof(float));if(p.embedded)mtp_set_embedding_input(p.embedded,embedding_,ids,count);else ggml_backend_tensor_set(p.ids,ids,0,count*sizeof(int32_t));ggml_backend_tensor_set(p.positions,positions,0,count*sizeof(int32_t));
        if(ggml_backend_graph_compute(backend_,p.graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MTP KV catch-up failed");committed(positions,count);
    }
};
}
