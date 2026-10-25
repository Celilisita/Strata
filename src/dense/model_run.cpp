#include "model_graph.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::dense {
bool Model::compact_async_available() const noexcept {
    return compact_async_ && compact_readback_ && compact_readback_->available();
}
const char *Model::compact_async_status() const noexcept {
    return compact_readback_ ? compact_readback_->status() : "disabled";
}
void Model::set_cost_modes(bool compact_async, bool norm_fusion) {
    if (compact_async && !compact_readback_)
        throw std::runtime_error("compact async requires a host packet at model load");
    if (norm_fusion && !norm_fusion_)
        throw std::runtime_error("norm fusion requires a constant at model load");
    if (norm_enabled_ != norm_fusion) decode_graph_.reset();
    compact_async_ = compact_async;
    norm_enabled_ = norm_fusion;
}
void Model::set_projection_fusion(bool enabled) {
    if (enabled && (!projection_pack_ || projection_pack_->group_count() == 0))
        throw std::runtime_error("projection fusion requires packed weights at model load");
    if (projection_fusion_ != enabled) decode_graph_.reset();
    projection_fusion_ = enabled;
}
size_t Model::projection_pack_group_count() const noexcept {
    return projection_pack_ ? projection_pack_->group_count() : 0;
}
size_t Model::projection_pack_member_count() const noexcept {
    return projection_pack_ ? projection_pack_->member_count() : 0;
}
void Model::set_device_greedy(bool enabled) {
    if (device_greedy_ != enabled) decode_graph_.reset();
    device_greedy_ = enabled;
    device_token_ = -1;
}
int Model::device_greedy_token() const {
    if (!device_greedy_ || device_token_ < 0 || position_ == 0)
        throw std::logic_error("dense compact greedy output is unavailable");
    return device_token_;
}
std::vector<float> Model::append(const std::vector<int> &ids, bool need_logits) {
    InputBatch batch;batch.ids=ids;
    return append_input(batch,need_logits);
}
std::vector<float> Model::append_input(const InputBatch &batch, bool need_logits) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    auto elapsed = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    cost_ = {};
    device_token_ = -1;
    const auto &ids=batch.ids;
    const int count = int(ids.size());
    if (count < 1 || count > 8192 || position_ > capacity_-count)
        throw std::invalid_argument("dense append exceeds batch or context capacity");
    for (int id : ids) if (id < 0 || uint64_t(id) >= spec_.n_vocab)
        throw std::invalid_argument("dense token ID is outside vocabulary");
    const bool positions=!batch.positions.empty();
    const auto images=batch.image_rows.size();
    if(images>size_t(count)||batch.embeddings.size()!=images*spec_.n_embd||
       (images&&!positions)||(!positions&&batch.next_logical>=0))
        throw std::invalid_argument("dense visual batch shape differs from input span");
    if(positions){
        if(spec_.architecture!="qwen35"||batch.positions.size()!=size_t(4*count)||
           batch.next_logical<logical_position_||uint64_t(batch.next_logical)>spec_.context_length)
            throw std::invalid_argument("dense visual position axes or cursor invalid");
        int maximum=0;
        for(size_t i=0;i<batch.positions.size();++i){
            const int p=batch.positions[i];
            if(p<0||uint64_t(p)>=spec_.context_length||(i>=size_t(3*count)&&p!=0))
                throw std::invalid_argument("dense visual position exceeds model context");
            if(i<size_t(3*count))maximum=std::max(maximum,p);
        }
        if(batch.next_logical<=maximum)throw std::invalid_argument("dense visual cursor precedes input positions");
    }else if(uint64_t(logical_position_)+count>spec_.context_length)
        throw std::invalid_argument("dense logical context exceeded");
    int64_t previous=-1;
    for(int64_t row:batch.image_rows){
        if(row<=previous||row>=count)throw std::invalid_argument("dense image row is duplicated or outside batch");
        previous=row;
    }
    for(float v:batch.embeddings)if(!std::isfinite(v))throw std::invalid_argument("nonfinite dense visual input");
    const int kv = (position_+count+255)/256*256;
    const bool decode = count == 1 && need_logits && position_ > 0 && !positions;
    std::unique_ptr<RequestGraph> transient;
    RequestGraph *request = nullptr;
    if (decode) {
        if (!decode_graph_ || decode_graph_->kv != kv ||
            decode_graph_->device_greedy != device_greedy_) {
            decode_graph_.reset();
            auto candidate = build_graph(count,kv,true,true);
            allocate_graph(*candidate);
            decode_graph_ = std::move(candidate);
        }
        request = decode_graph_.get();
    } else {
        // No retained graph may borrow a buffer that PP can reallocate.
        decode_graph_.reset();
        transient = build_graph(count,kv,need_logits,false,int(images));
        allocate_graph(*transient);
        request = transient.get();
    }
    const auto graph_ready = Clock::now();
    cost_.graph_ms = elapsed(begin,graph_ready);
    cost_.nodes = ggml_graph_n_nodes(request->graph);
    cost_.projection_fused_matmuls = request->projection_fused_matmuls;
    cost_.norm_fused_nodes = request->norm_fused_nodes;
    if(positions)request->host_positions=batch.positions;
    else for (int axis = 0; axis < request->axes; ++axis)
        for (int i = 0; i < count; ++i)
            request->host_positions[size_t(axis)*count+i] = axis == 3 ? 0 : logical_position_+i;
    const auto masked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int row = 0; row < count; ++row) {
        auto first = request->host_mask.begin()+size_t(row)*kv;
        std::fill(first,first+position_+row+1,ggml_fp16_t(0));
        std::fill(first+position_+row+1,first+kv,masked);
    }
    size_t row_bytes = 0;
    if (request->cache_rows) {
        request->host_cache_row = position_;
        row_bytes = sizeof(request->host_cache_row);
    }
    if(request->image_rows)
        row_bytes+=images*sizeof(int64_t)+batch.embeddings.size()*sizeof(float);
    // This guard is destroyed before transient/request/batch host inputs. It
    // drains partial submission on errors as well as the normal async path.
    BackendPending pending(backend_);
    // Enqueue after host preparation on compute's ordered backend stream.
    ggml_backend_tensor_set_async(backend_,request->tokens,ids.data(),0,ids.size()*sizeof(int));
    ggml_backend_tensor_set_async(backend_,request->positions,request->host_positions.data(),0,
                                  request->host_positions.size()*sizeof(int));
    ggml_backend_tensor_set_async(backend_,request->mask,request->host_mask.data(),0,
                                  request->host_mask.size()*sizeof(ggml_fp16_t));
    if(request->cache_rows)
        ggml_backend_tensor_set_async(backend_,request->cache_rows,&request->host_cache_row,0,
                                      sizeof(request->host_cache_row));
    if(request->image_rows){
        ggml_backend_tensor_set_async(backend_,request->image_rows,batch.image_rows.data(),0,images*sizeof(int64_t));
        ggml_backend_tensor_set_async(backend_,request->image_embeddings,batch.embeddings.data(),0,batch.embeddings.size()*sizeof(float));
    }
    const auto input_ready = Clock::now();
    cost_.input_ms = elapsed(graph_ready,input_ready);
    cost_.input_bytes = ids.size()*sizeof(int)+request->host_positions.size()*sizeof(int)+
                         request->host_mask.size()*sizeof(ggml_fp16_t)+row_bytes;
    const bool compact_async = request->device_greedy && compact_async_available();
    const auto status = compact_async ? ggml_backend_graph_compute_async(backend_,request->graph) :
                                       ggml_backend_graph_compute(backend_,request->graph);
    if (status != GGML_STATUS_SUCCESS)
        throw std::runtime_error("dense CUDA graph computation failed; request must reset");
    if (!compact_async) pending.dismiss();
    const auto computed = Clock::now();
    cost_.compute_ms = elapsed(input_ready,computed);
    std::vector<float> result(need_logits && !request->device_greedy ? size_t(spec_.n_vocab) : 0);
    if (request->device_greedy) {
        int32_t token = -1;
        float finite_sum = 0;
        if (compact_async) {
            compact_readback_->enqueue(request->greedy_id,request->finite_sum);
            pending.synchronize();
            const auto packet = compact_readback_->read();
            token = packet.token;finite_sum = packet.finite_sum;
            cost_.compact_async_steps = 1;
        } else {
            ggml_backend_tensor_get(request->greedy_id,&token,0,sizeof(token));
            ggml_backend_tensor_get(request->finite_sum,&finite_sum,0,sizeof(finite_sum));
        }
        // abs + sum propagates every NaN/Inf. Overflow from extreme finite
        // logits is conservatively rejected too; never emit an unchecked ID.
        if (!std::isfinite(finite_sum) || finite_sum < 0 ||
            token < 0 || uint64_t(token) >= spec_.n_vocab)
            throw std::runtime_error("nonfinite dense logits or invalid device greedy token");
        device_token_ = token;
        cost_.output_bytes = sizeof(token)+sizeof(finite_sum);
    } else if (request->logits) {
        ggml_backend_tensor_get(request->logits,result.data(),0,result.size()*sizeof(float));
        cost_.output_bytes = result.size()*sizeof(float);
    }
    cost_.readback_ms = elapsed(computed,Clock::now());
    cost_.execute_readback_ms = cost_.compute_ms+cost_.readback_ms;
    position_ += count;
    logical_position_=positions?batch.next_logical:logical_position_+count;
    return result;
}
} // namespace strata::dense
