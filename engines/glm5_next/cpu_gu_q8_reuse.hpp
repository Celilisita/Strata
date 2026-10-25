#pragma once
// Diagnostic-only typed input preparation on the existing native16 GGML pool.
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::glm5_next {
class CpuGuQ8Reuse {
public:
    struct Counts { uint64_t calls=0,quantized_blocks=0,copied_blocks=0; };
    static constexpr size_t block_bytes=292,row_bytes=16*block_bytes;
    CpuGuQ8Reuse(ggml_context* ctx,ggml_cgraph* graph,ggml_tensor* old_activation,
        ggml_tensor* ids,ggml_tensor* down,const std::array<ggml_tensor*,2>& banks,
        ggml_backend_sched_t scheduler,ggml_backend_t cpu,int threads) {
        if(!ctx||!graph||!old_activation||!ids||!down||threads!=16||
            old_activation->type!=GGML_TYPE_F32||old_activation->ne[0]!=4096||
            old_activation->ne[1]<1||old_activation->ne[1]>16||old_activation->ne[2]!=1||old_activation->ne[3]!=1||
            ids->type!=GGML_TYPE_I32||ids->ne[0]!=1||ids->ne[1]!=old_activation->ne[1]||
            ggml_type_size(GGML_TYPE_Q8_K)!=block_bytes||ggml_blck_size(GGML_TYPE_Q8_K)!=256||
            !ggml_is_contiguous(old_activation)||!ggml_is_contiguous(ids)||
            !ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float)
            throw std::runtime_error("GU Q8 reuse requires original B2 routed input and native16");
        routes_=size_t(ids->ne[1]);std::array<ggml_tensor*,2> gu{};
        for(int role=0;role<2;++role){const auto* bank=banks[role];
            if(!bank||bank->type!=GGML_TYPE_IQ3_S||bank->ne[0]!=4096||bank->ne[1]!=2048||
                bank->ne[2]!=288||bank->ne[3]!=1||!ggml_is_contiguous(bank)||
                !bank->buffer||!ggml_backend_buffer_is_host(bank->buffer)||
                ggml_get_type_traits_cpu(bank->type)->vec_dot_type!=GGML_TYPE_Q8_K)
                throw std::runtime_error("GU Q8 reuse requires original IQ3_S banks");
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* node=ggml_graph_node(graph,i);
                if(node->op==GGML_OP_MUL_MAT_ID&&node->src[0]==bank){
                    if(gu[role])throw std::runtime_error("GU Q8 reuse found duplicate bank node");gu[role]=node;}}
            if(!gu[role]||gu[role]->src[2]!=ids)
                throw std::runtime_error("GU Q8 reuse original bank/route node differs");
        }
        const auto* shared=gu[0]->src[1];
        if(!shared||shared!=gu[1]->src[1]||shared->type!=GGML_TYPE_F32||
            shared->op!=GGML_OP_RESHAPE||shared->src[0]!=old_activation||
            shared->ne[0]!=4096||shared->ne[1]!=1||shared->ne[2]!=int64_t(routes_)||shared->ne[3]!=1||
            !ggml_is_contiguous(shared))throw std::runtime_error("GU Q8 reuse shared reshape differs");
        activation_=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4096,2);
        positions_=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,int64_t(routes_));
        ggml_set_name(activation_,"strata.gu_q8.actual_positions");ggml_set_input(activation_);
        ggml_set_name(positions_,"strata.gu_q8.route_positions");ggml_set_input(positions_);
        ggml_tensor* args[]{activation_,positions_};
        packed_=ggml_custom_4d(ctx,GGML_TYPE_Q8_K,4096,1,int64_t(routes_),1,args,2,prepare,16,this);
        ggml_set_name(packed_,"strata.gu_q8.shared_routes");
        // Retained for the two real consumers and untimed defined-field proof.
        ggml_set_output(packed_);
        for(auto* node:gu)node->src[1]=packed_;
        ggml_graph_clear(graph);ggml_build_forward_expand(graph,down);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i){const auto* node=ggml_graph_node(graph,i);
            if(node->view_src==old_activation)throw std::runtime_error("GU Q8 old expanded input remains live");
            for(const auto* source:node->src)if(source==old_activation)
                throw std::runtime_error("GU Q8 old expanded input still has a consumer");}
        for(auto* input:{activation_,positions_,packed_})ggml_backend_sched_set_tensor_backend(scheduler,input,cpu);
    }
    void set_inputs(const float* two_actual,const uint8_t* positions) {
        std::array<int32_t,16> map{};
        for(size_t row=0;row<routes_;++row){if(positions[row]>1)throw std::runtime_error("GU Q8 reuse position outside B2");map[row]=positions[row];}
        counts_={};
        ggml_backend_tensor_set(activation_,two_actual,0,ggml_nbytes(activation_));
        ggml_backend_tensor_set(positions_,map.data(),0,routes_*sizeof(int32_t));
    }
    Counts counts()const {Counts out;for(const auto& row:counts_){out.calls+=row.value.calls;
        out.quantized_blocks+=row.value.quantized_blocks;out.copied_blocks+=row.value.copied_blocks;}return out;}
    std::vector<uint8_t> snapshot()const {std::vector<uint8_t> bytes(routes_*row_bytes);
        ggml_backend_tensor_get(packed_,bytes.data(),0,bytes.size());return bytes;}
    size_t routes()const{return routes_;}
    size_t input_bytes()const{return 2*4096*sizeof(float)+routes_*sizeof(int32_t);}
    size_t packed_bytes()const{return routes_*row_bytes;}
private:
    struct alignas(64) Worker {Counts value;};
    static void prepare(ggml_tensor* dst,int ith,int nth,void* userdata) {
        auto& self=*static_cast<CpuGuQ8Reuse*>(userdata);
        GGML_ASSERT(nth==16&&ith>=0&&ith<16&&dst==self.packed_&&dst->data);
        const auto* actual=dst->src[0];const auto* positions=dst->src[1];
        GGML_ASSERT(actual->data&&positions->data&&ggml_is_contiguous(dst));
        const auto* map=static_cast<const int32_t*>(positions->data);uint32_t active=0;
        for(size_t row=0;row<self.routes_;++row){GGML_ASSERT(map[row]>=0&&map[row]<2);active|=1u<<map[row];}
        // Each worker keeps one exact256-value block per active physical position.
        // No worker reads another worker's scratch or partially written block.
        // Canonical all-zero blocks leave bsums unwritten. Zero-initialize
        // those pre-existing dead bytes; defined d/qs/bsums remain canonical.
        alignas(64) std::array<uint8_t,2*block_bytes> local{};uint64_t quantized=0;
        const auto quantize=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
        for(unsigned position=0;position<2;++position)if(active&(1u<<position)){
            const auto* input=static_cast<const float*>(actual->data)+size_t(position)*4096+size_t(ith)*256;
            quantize(input,local.data()+position*block_bytes,256);++quantized;}
        for(size_t row=0;row<self.routes_;++row)
            std::memcpy(static_cast<uint8_t*>(dst->data)+row*row_bytes+size_t(ith)*block_bytes,
                local.data()+size_t(map[row])*block_bytes,block_bytes);
        self.counts_[size_t(ith)].value={1,quantized,self.routes_};
        // The ordinary graph-node completion barrier publishes all16 block lanes.
    }
    ggml_tensor *activation_=nullptr,*positions_=nullptr,*packed_=nullptr;
    size_t routes_=0;std::array<Worker,16> counts_{};
};
} // namespace strata::glm5_next
