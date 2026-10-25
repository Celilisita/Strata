#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/attention_global_aos_merge.hpp"
#include "strata/gemma4/native/attention_swa_warp_merge.hpp"
#include "strata/gemma4/native/attention_global_gqa_shared.hpp"
namespace strata::gemma4::native {
void Model::select_t1_paths(unsigned mask){
    if((mask&~2045u)||((mask&1u)&&(mask&16u))||((mask&256u)&&!(mask&32u))||((mask&128u)&&(mask&512u)))
        throw std::invalid_argument("Gemma296 T1 mask or conflicting245/264 QKV layouts");
    if(host_position_!=0)throw std::invalid_argument("Gemma296 T1 paths require reset P0");
    if((mask&(1u|16u))&&(single_projection(0,1)||single_projection(1,1)))
        throw std::invalid_argument("Gemma296 QKV research path conflicts with legacy Single40 policy");
    if((mask&4u)&&(single_projection(4,1)||head_sidecar_.enabled()||head_.data!=embedding_.data))
        throw std::invalid_argument("Gemma two-plane head requires raw tied inventory and independent residual policy");
    if(mask&(8u|16u|32u|64u|128u|256u|512u)){
        if(spec_.hidden!=5376||spec_.intermediate!=21504||spec_.heads!=32||layers_.size()!=60)
            throw std::invalid_argument("Gemma296 fixed native Model geometry");
    }
    if((mask&8u)&&(q6_down_storage_!=Q6DownStorage::Aligned224||q6_down_matrices_!=30))
        throw std::invalid_argument("Gemma262 requires unique30 native224 Down weights");
    if(mask&16u){
        size_t projections=0,q4=0,q6=0;
        for(size_t i=0;i<layers_.size();++i){
            const auto& layer=spec_.layers[i];const auto& w=layers_[i];
            const auto valid=[&](primitive::Matrix m,int rows,bool allow_q6=false){
                if(m.layout!=primitive::BlockLayout::Gguf||m.columns!=5376||m.rows!=rows||
                   (m.quant!=primitive::Quant::Q4K&&(!allow_q6||m.quant!=primitive::Quant::Q6K)))
                    throw std::invalid_argument("Gemma264 raw native QKV inventory");
                ++projections;q4+=m.quant==primitive::Quant::Q4K;q6+=m.quant==primitive::Quant::Q6K;
            };
            valid(w.q,layer.sliding?8192:16384);valid(w.k,layer.sliding?4096:2048);
            if(!layer.shared_raw_kv)valid(w.v,4096,true);
            else if(w.v.data!=w.k.data)throw std::invalid_argument("Gemma264 shared global V=K ownership");
        }
        if(projections!=170||q4!=145||q6!=25)throw std::invalid_argument("Gemma264 exact170 QKV inventory");
    }
    if(mask&32u){
        if(!split_decode_)throw std::invalid_argument("Gemma266 requires native split attention");
        size_t globals=0;
        for(const auto& layer:spec_.layers)if(!layer.sliding){
            if(layer.head_dim!=512||layer.kv_heads!=4||!layer.shared_raw_kv)
                throw std::invalid_argument("Gemma266 full-history global geometry");
            ++globals;
        }
        if(globals!=10)throw std::invalid_argument("Gemma266 ten full-history global layers");
    }
    if(mask&64u){
        if(!split_decode_||spec_.window!=1024||capacity_<1024||capacity_>262144)
            throw std::invalid_argument("Gemma270 nativeSWA requestcontext");
        size_t locals=0;
        for(const auto& layer:spec_.layers)if(layer.sliding){
            if(layer.head_dim!=256||layer.kv_heads!=16)
                throw std::invalid_argument("Gemma270 fixed localgeometry");
            ++locals;
        }
        if(locals!=50)throw std::invalid_argument("Gemma270 fiftySWA layers");
    }
    if(mask&512u){
        const auto arena=ffn_->attention_workspace();
        if(!ffn_->single_ffn_enabled()||ffn_->t1_q4_ffn_policy()==T1Q4FfnPolicy::Off||
            !arena.data||reinterpret_cast<std::uintptr_t>(arena.data)%8)
            throw std::invalid_argument("Gemma291 requires Single40/Q4GU and input8 arena");
        size_t matrices=0;
        for(const auto& layer:layers_)for(auto w:{layer.ffn.gate,layer.ffn.up}){
            if(!w.data||w.quant!=primitive::Quant::Q4K||w.layout!=primitive::BlockLayout::Gguf||
                w.rows!=21504||w.columns!=5376||reinterpret_cast<std::uintptr_t>(w.data)%16)
                throw std::invalid_argument("Gemma291 fixed raw Q4 weight16 inventory");
            ++matrices;
        }
        if(matrices!=120)throw std::invalid_argument("Gemma291 exact sixty GU pairs");
    }

    if(mask&1024u){
        if(!single_projection(2,1)||!single_projection(3,1))
            throw std::invalid_argument("Gemma302 requires existing local/global Single40 Out");
        size_t locals=0,globals=0;
        for(size_t i=0;i<layers_.size();++i){
            const auto w=layers_[i].out;const bool local=spec_.layers[i].sliding;
            if(!w.data||w.quant!=primitive::Quant::Q4K||w.layout!=primitive::BlockLayout::Gguf||
               w.rows!=5376||w.columns!=(local?8192:16384)||reinterpret_cast<std::uintptr_t>(w.data)%4)
                throw std::invalid_argument("Gemma302 all60 native Q4 Out inventory");
            local?++locals:++globals;
        }
        if(locals!=50||globals!=10)throw std::invalid_argument("Gemma302 exact50/10 Out geometry");
    }
    if(t1_paths_==mask)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma296 cold paths drain");
    release_graph();ffn_->select_t1_q6_down_striped(bool(mask&8u));
    ffn_->select_t1_gate_up_direct(false);
    ffn_->select_t1_gate_up_word(bool(mask&128u));
    ffn_->select_t1_gate_up_direct(bool(mask&512u));
    t1_paths_=mask;++reset_epoch_;
}
void Model::select_short_pp_backend(ShortPpBackend mode){
    if(mode!=ShortPpBackend::ExistingPanel3&&mode!=ShortPpBackend::FastPanel3&&mode!=ShortPpBackend::FastFullK)
        throw std::invalid_argument("Gemma305 short PP backend enum");
    if(host_position_!=0)throw std::invalid_argument("Gemma305 short PP requires reset P0");
    if(mode!=ShortPpBackend::ExistingPanel3&&(maximum_batch_!=8192||
        spec_.hidden!=5376||spec_.intermediate!=21504))
        throw std::invalid_argument("Gemma305 fast short PP exact native PP storage geometry");
    if(mode!=ShortPpBackend::ExistingPanel3){
        size_t matrices=0;
        for(const auto& layer:layers_)for(auto w:{layer.ffn.gate,layer.ffn.up}){
            if(!w.data||w.quant!=primitive::Quant::Q4K||w.layout!=primitive::BlockLayout::Gguf||
                w.rows!=21504||w.columns!=5376||reinterpret_cast<std::uintptr_t>(w.data)%16)
                throw std::invalid_argument("Gemma305 all120 raw Q4 GU weight16 inventory");
            ++matrices;
        }
        if(matrices!=120)throw std::invalid_argument("Gemma305 exact sixty GU pairs");
    }
    if(ffn_->short_pp_backend()==mode)return;
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma305 cold short PP drain");
    release_graph();ffn_->select_short_pp_backend(mode);++reset_epoch_;
}
bool Model::enqueue_parallel_global_t1(CacheView c,int tokens){
    if(!(t1_paths_&32u)||tokens!=1||c.window!=0)return false;
    const AosMergeT1Binding live{this,reset_epoch_,stream_};
    // This request identity is frozen at initial prefill and cleared on reset.
    // 284 admitted280 only at30720..32767, not for short or arbitrary requests.
    if((t1_paths_&256u)&&global_shared_long_request_){
        global_gqa_shared_t1(q_,attended_,c,position_,spec_.heads,tokens,
            {ffn_->attention_workspace(),live,capacity_},live);
        ++t1_global_shared280_enqueues_;return true;
    }
    global_parallel_aos_merge_t1(q_,attended_,c,position_,spec_.heads,tokens,
        {ffn_->attention_workspace(),live,capacity_},live);
    ++t1_global_merge_enqueues_;return true;
}
bool Model::enqueue_swa_warp_t1(CacheView c,int tokens){
    if(!(t1_paths_&64u)||tokens!=1||c.window==0)return false;
    const SwaWarpMergeT1Binding live{this,reset_epoch_,stream_};
    const auto arena=ffn_->attention_workspace();
    swa_warp_merge_t1(q_,attended_,c,position_,spec_.heads,tokens,
        {arena.data,arena.bytes,live,capacity_},live);
    ++t1_swa270_enqueues_;return true;
}
}
