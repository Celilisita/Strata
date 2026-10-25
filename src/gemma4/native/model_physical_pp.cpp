#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/pp_hh_only349_policy.hpp"
#include "attention_physical_pp.hpp"
#include "attention_compensated_pp.hpp"
#include "attention_compensated_pp_k1024.hpp"
#include "attention_three_product.hpp"
#include "attention_hh_only.hpp"

namespace strata::gemma4::native {
void Model::select_physical_global_pp(bool enabled){
    if(host_position_!=0||executable_||(enabled&&maximum_batch_!=8192))
        throw std::invalid_argument("Gemma physical PP selection requires reset/P0/batch8192");
    if(physical_global_pp_==enabled)return;
    // Cold lifecycle transition only. A stream capture cannot pass this join.
    // External/controller Graphs must already be released by their owner.
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma physical PP policy join");
    release_graph();++reset_epoch_;
    physical_global_pp_=enabled;
}

void Model::select_compensated_physical_global_pp(bool enabled){
    if(host_position_!=0||executable_||
       (enabled&&(!physical_global_pp_||maximum_batch_!=8192)))
        throw std::invalid_argument("Gemma290 compensatedPP requires coldP0/physicalPP/batch8192");
    if(compensated_physical_global_pp_==enabled)return;
    // External/controller graphs must already be released by their owner.
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma290 compensatedPP policy join");
    release_graph();++reset_epoch_;
    compensated_physical_global_pp_=enabled;
    compensated_physical_global_pp_request_=false;
    compensated_physical_global_pp_enqueues_=0;
}

void Model::select_pp_three_product333(bool enabled){
    if(host_position_!=0||executable_)
        throw std::invalid_argument("Gemma339 three-product PP selection requires reset P0");
    if(enabled)validate_global_attention_geometry();
    if(pp_three_product333_==enabled)return;
    // Independent cold selection: only eligible compensated K1024 requests use it.
    // Caller has already released controller/external Graphs.
    primitive::check_cuda(cudaStreamSynchronize(stream_),"Gemma339 PP333 cold policy join");
    release_graph();++reset_epoch_;pp_three_product333_=enabled;
    pp_three_product333_enqueues_=0;pp_k1024_enqueues_=0;
}

bool Model::enqueue_physical_global_pp(CacheView cache,int tokens){
    const bool chunk=(tokens==8192&&(host_position_==0||host_position_==8192||host_position_==16384))||
        (tokens==6144&&host_position_==24576);
    const bool long_shape=capacity_==32768&&chunk;
    const bool new8k_shape=pp349::is_k1024(pp_key_tile_)&&compensated_physical_global_pp_&&capacity_==10240&&tokens==8192&&host_position_==0;
    if(!physical_global_pp_request_||!blas_prefill_||(!long_shape&&!new8k_shape)||maximum_batch_!=8192||
       spec_.heads!=32||cache.capacity!=capacity_||
       cache.window!=0||cache.heads!=4||cache.dim!=512)return false;
    if(!physical_global_pp_){++parent_long_global_pp_enqueues_;return false;}
    const auto arena=ffn_->attention_workspace();
    // These exact extents are allocated in Model::Model. Raw K and V are dead
    // after store_kv; Out/next-layer projections cannot reuse them until this
    // attention finishes on the same ordered stream. No pointers are retained.
    const size_t raw_bytes=size_t(maximum_batch_)*16*256*sizeof(float);
    const PhysicalPpLease lease{this,stream_,k_,raw_bytes,v_,raw_bytes};
    // Physical K/scores/online state borrows the Ffn decoded-weight arena.
    // QKV has consumed it; Out and FFN reinitialize it after this queued work.
    // Draft shares the arena on this same stream, without concurrent lifetime.
    if(compensated_physical_global_pp_){
        if(pp349::is_k1024(pp_key_tile_)){
            const CompensatedPpK1024Binding live{this,reset_epoch_,stream_};
            const CompensatedPpK1024Lease lease{k_,raw_bytes,v_,raw_bytes,live};
            if(pp349::is_hh(pp_key_tile_)){
                if(!pp349::supported_chunk(capacity_,tokens,host_position_))
                    throw std::invalid_argument("Gemma349 HH only supports exact8K/30K GPU PP geometry");
                hh_only_physical_pp_349_attention(q_,attended_,cache,position_,host_position_,
                    spec_.heads,tokens,arena,lease,live);
            }else if(pp_three_product333_){
                three_product_physical_pp_333_attention(q_,attended_,cache,position_,host_position_,
                    spec_.heads,tokens,arena,lease,live);++pp_three_product333_enqueues_;
            }else compensated_physical_pp_k1024_attention(q_,attended_,cache,position_,host_position_,
                spec_.heads,tokens,arena,lease,live);
            ++pp_k1024_enqueues_;
        }else{
            const CompensatedPpBinding live{this,reset_epoch_,stream_};
            const CompensatedPpLease compensated{k_,raw_bytes,v_,raw_bytes,live};
            compensated_physical_pp_attention(q_,attended_,cache,position_,host_position_,
                spec_.heads,tokens,arena,compensated,live);
        }
        compensated_physical_global_pp_request_=true;
        // Logical compensated route total includes all inherited/HH leaf backends.
        ++compensated_physical_global_pp_enqueues_;
    }else{
        physical_global_pp_attention(q_,attended_,cache,position_,host_position_,
            spec_.heads,tokens,arena,lease,stream_);
    }
    // Retain the logical physicalroute total for the inherited40-call contract.
    ++physical_global_pp_enqueues_;
    return true;
}
}
