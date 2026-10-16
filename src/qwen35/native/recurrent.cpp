#include "strata/qwen35/native/tiny_batch.hpp"
#include "strata/qwen35/native/small_prefill.hpp"
#include "strata/qwen35/native/recurrent.hpp"
#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/gdn.hpp"
#include "strata/qwen35/native/ops.hpp"
#include "small_qkv32_cache.hpp"
#include "small_q4_projection32.hpp"
#include <algorithm>
#include <cmath>

namespace strata::qwen35::native {
RecurrentLayer::RecurrentLayer(RecurrentWeights w, RecurrentShape s, int splits, RecurrentSchedule schedule)
    : weights_(w), shape_(s), schedule_(schedule) {
    if (!w.norm || !w.post_norm || !w.conv || !w.dt || !w.a || !w.gdn_norm ||
        s.hidden < 256 || s.hidden % 256 || (s.dim != 64 && s.dim != 128) || s.query_heads < 1 ||
        s.value_heads < s.query_heads || s.value_heads % s.query_heads || s.value_heads > 256 ||
        s.max_tokens < 1 || s.max_tokens > 8192 || !(s.epsilon > 0) || !std::isfinite(s.epsilon))
        throw std::invalid_argument("invalid own recurrent layer shape or parameters");
    const int inner = s.dim * s.value_heads, channels = s.dim * (2 * s.query_heads + s.value_heads);
    const bool code_ffn=w.ffn_down.layout==BlockLayout::Q6Codes;
    if(w.qkv.layout==BlockLayout::Q6Codes&&(!code_ffn||s.max_tokens!=32))throw std::invalid_argument("byte-code QKV layer admission is prepared PP32 only");
    if(code_ffn && s.max_tokens!=32)throw std::invalid_argument("prepared FFN layer admission is PP32 only");
    const Matrix matrices[] = {w.qkv, w.z, w.alpha, w.beta, w.out, w.ffn_gate, w.ffn_up, w.ffn_down};
    for(int i=0;i<8;++i)matrix_bytes(matrices[i],i==7||(i==0&&w.qkv.layout==BlockLayout::Q6Codes),false,i==5||i==6);
    if (w.qkv.rows != channels || w.z.rows != inner || w.alpha.rows != s.value_heads || w.beta.rows != s.value_heads ||
        w.qkv.columns != s.hidden || w.z.columns != s.hidden || w.alpha.columns != s.hidden || w.beta.columns != s.hidden ||
        w.out.rows != s.hidden || w.out.columns != inner || w.ffn_gate.columns != s.hidden || w.ffn_down.rows != s.hidden)
        throw std::invalid_argument("own recurrent layer matrix shape mismatch");
    const size_t decoded_capacity = (std::max)(2 * size_t(w.ffn_gate.rows) * w.ffn_gate.columns,
                                             size_t(w.ffn_down.rows) * w.ffn_down.columns);
    const int input_capacity = (std::max)(w.ffn_gate.columns, w.ffn_down.columns);
    for (auto m : matrices) if (size_t(m.rows) * m.columns > decoded_capacity || m.columns > input_capacity)
        throw std::invalid_argument("own recurrent projection exceeds shared staging capacity");
    decoded_capacity_=decoded_capacity;input_capacity_=input_capacity;
    size_t partial_rows = s.partial_rows_capacity;
    for (int i = 0; i < 5; ++i) {
        const auto p = schedule.projections[i];
        if (p.partitions < 1 || p.partitions > 32 || (p.granularity != 32 && p.granularity != 64 && p.granularity != 128))
            throw std::invalid_argument("invalid own recurrent projection schedule");
        if (p.partitions > 1) partial_rows = (std::max)(partial_rows, size_t(p.partitions) * matrices[i].rows);
    }
    partial_rows_capacity_=partial_rows;
    try {
        pp_ = std::make_unique<Ffn>(w.ffn_gate, w.ffn_up, w.ffn_down, s.max_tokens, Provider::OwnedStaged, splits);
        if(!code_ffn)tg_ = std::make_unique<Ffn>(w.ffn_gate, w.ffn_up, w.ffn_down, 1, Provider::OwnedVector);
        workspace_bytes_ = pp_->workspace_bytes() + (tg_?tg_->workspace_bytes():0);
        auto allocate = [&](float **p, size_t features) {
            check_cuda(cudaMalloc(reinterpret_cast<void **>(p), features * s.max_tokens * 4), "own recurrent activation allocation");
            allocations_.push_back(*p); workspace_bytes_ += features * s.max_tokens * 4;
        };
        allocate(&buffers_.norm, s.hidden);
        if(s.compact_workspace){
            if(w.ffn_gate.rows<(std::max)(channels,s.mixed_capacity))
                throw std::invalid_argument("FFN activation arena cannot hold shared attention channels");
            // These three buffers become FFN outputs only after attention finishes.
            buffers_.qkv=pp_->buffers().gate;buffers_.conv_raw=pp_->buffers().up;
            buffers_.conv_silu=pp_->buffers().activation;
        }else{
            allocate(&buffers_.qkv,(std::max)(channels,s.mixed_capacity));
            allocate(&buffers_.conv_raw,channels);allocate(&buffers_.conv_silu,channels);
        }
        allocate(&buffers_.z, inner); allocate(&buffers_.alpha, s.value_heads); allocate(&buffers_.beta_raw, s.value_heads);
        allocate(&buffers_.g, s.value_heads); allocate(&buffers_.beta, s.value_heads);
        allocate(&buffers_.q, s.dim * s.query_heads); allocate(&buffers_.k, s.dim * s.query_heads);
        allocate(&buffers_.v, inner); allocate(&buffers_.attention, inner); allocate(&buffers_.gated, inner);
        allocate(&buffers_.projected, s.hidden); allocate(&buffers_.residual, s.hidden);
        allocate(&buffers_.post_norm, s.hidden); allocate(&buffers_.output, s.hidden);
        const size_t down_rows=size_t(splits)*w.ffn_down.rows;
        if(pp_->buffers().down_partials && partial_rows<=down_rows){
            // Projection reductions finish before FFN Down begins on this stream.
            projection_partials_=pp_->buffers().down_partials;partial_rows_capacity_=down_rows;
        }else if(partial_rows)allocate(&projection_partials_,partial_rows);
    } catch (...) {
        for (void *p : allocations_) cudaFree(p);
        throw;
    }
}
RecurrentLayer::~RecurrentLayer() { for (void *p : allocations_) cudaFree(p); }
void RecurrentLayer::enable_qkv32(cudaStream_t stream,bool complete,bool register_reuse,bool prefetch,bool planes,bool byte_planes){
    if(qkv32_||shape_.max_tokens<32||(planes&&(!native32_||stream!=native32_stream_)))
        throw std::invalid_argument("QKV32 cache already enabled or admission changed");
    auto prepared=std::make_unique<SmallQkv32>(weights_.qkv,stream,complete,register_reuse,prefetch,planes,byte_planes);
    qkv32_extra_submissions_=prepared->extra_submissions();
    workspace_bytes_+=prepared->bytes();qkv32_=std::move(prepared);
}
size_t RecurrentLayer::qkv32_cache_bytes()const{return qkv32_?qkv32_->bytes():0;}
double RecurrentLayer::qkv32_cold_ms()const{return qkv32_?qkv32_->cold_ms():0;}
void RecurrentLayer::borrow_native32_qkv_planes(const void *data,size_t bytes,cudaStream_t stream){
    if(!native32_||qkv32_||stream!=native32_stream_||weights_.qkv.quant!=Quant::Q6K||
       weights_.qkv.layout!=BlockLayout::Gguf||weights_.qkv.rows%64||!data||
       reinterpret_cast<uintptr_t>(data)%16||(bytes!=matrix_bytes(weights_.qkv)&&bytes!=q6_byte_planes_bytes(weights_.qkv)))
        throw std::invalid_argument("borrowed QKV plane lease requires native32 raw geometry");
    q4projection32_->validate_lease(stream);
    const auto raw=reinterpret_cast<uintptr_t>(weights_.qkv.data),view=reinterpret_cast<uintptr_t>(data);
    if(raw<=view?view-raw<matrix_bytes(weights_.qkv):raw-view<bytes)throw std::invalid_argument("QKV plane view overlaps raw inventory");
    qkv_plane_data_=data;qkv_plane_bytes_=bytes;
}
void RecurrentLayer::rebind(RecurrentWeights w) {
    if(qkv32_&&!qkv32_->matches(w.qkv))throw std::invalid_argument("rebind would invalidate pinned QKV32 cache");
    if(q4projection32_&&!native32_&&!q4projection32_->matches({w.z,w.alpha,w.beta,w.out}))
        throw std::invalid_argument("rebind would invalidate borrowed Q4 projection stage");
    const Matrix next[]={w.qkv,w.z,w.alpha,w.beta,w.out,w.ffn_gate,w.ffn_up,w.ffn_down};
    const Matrix old[]={weights_.qkv,weights_.z,weights_.alpha,weights_.beta,weights_.out,
                        weights_.ffn_gate,weights_.ffn_up,weights_.ffn_down};
    if(!w.norm || !w.post_norm || !w.conv || !w.dt || !w.a || !w.gdn_norm)
        throw std::invalid_argument("invalid shared recurrent parameters");
    for(int i=0;i<8;++i){
        matrix_bytes(next[i],i==7||(i==0&&old[0].layout==BlockLayout::Q6Codes),false,i==5||i==6);
        if(i==7 && (next[i].layout==BlockLayout::Q6Codes||old[i].layout==BlockLayout::Q6Codes) && next[i].layout!=old[i].layout)throw std::invalid_argument("rebind changed FFN layer admission layout");
        if(next[i].rows!=old[i].rows || next[i].columns!=old[i].columns)
            throw std::invalid_argument("shared recurrent weight geometry changed");
    }
    if(native32_)q4projection32_->rebind({w.z,w.alpha,w.beta,w.out},native32_stream_);
    pp_->rebind(w.ffn_gate,w.ffn_up,w.ffn_down);if(tg_)tg_->rebind(w.ffn_gate,w.ffn_up,w.ffn_down);weights_=w;
    qkv_plane_data_=nullptr;qkv_plane_bytes_=0;
}
void RecurrentLayer::enable_native32(cudaStream_t stream){
    if(native32_||q4projection32_||qkv32_||shape_.max_tokens<32||weights_.qkv.rows%64||
       small_prefill_half_input_bytes(weights_.qkv.columns,32)>pp_->half_weight_capacity())
        throw std::invalid_argument("native32 requires unused shared raw inventory and sufficient input arena");
    auto stage=std::make_unique<SmallQ4Projection32>(std::array<Matrix,4>{weights_.z,weights_.alpha,weights_.beta,weights_.out},stream);
    pp_->enable_small_direct32(true);workspace_bytes_+=stage->bytes();q4projection32_=std::move(stage);
    native32_=true;native32_stream_=stream;fused_ops_=true;q4_norm_pair_=true;
}
void RecurrentLayer::enable_joint_projections(){
    if(joint_projections_||!schedule_.wide_grid||shape_.max_tokens<128)
        throw std::invalid_argument("joint projections require unused long wide-grid admission");
    configure_dense_joint();joint_projections_=true;
}
void RecurrentLayer::project_native32(const float *input,std::initializer_list<NativeProjection32> projections,cudaStream_t stream){
    if(!native32_||stream!=native32_stream_||!input||projections.size()<1||projections.size()>3)
        throw std::invalid_argument("native32 shared projection admission changed");
    q4projection32_->validate_lease(stream);
    const int columns=projections.begin()->weight.columns;
    const size_t encoded=small_prefill_half_input_bytes(columns,32);
    if(columns>input_capacity_||2*encoded>pp_->half_weight_capacity())throw std::invalid_argument("native32 shared input arena exceeded");
    for(const auto &p:projections){matrix_bytes(p.weight);
        if(!p.output||p.weight.columns!=columns||size_t(p.weight.rows)*columns>decoded_capacity_)
            throw std::invalid_argument("native32 projection group geometry changed");}
    bool prepared[2]{};auto *arena=static_cast<unsigned char *>(pp_->buffers().half_weights);
    for(const auto &p:projections){
        const int slot=p.weight.quant==Quant::Q6K?1:0;void *codes=arena+slot*encoded;
        if(!prepared[slot]){small_prefill_half_raw_quantize(input,codes,p.weight.quant,columns,32,stream);prepared[slot]=true;}
        if(p.weight.quant==Quant::Q4K&&q4_single_partitions(p.weight)==2)
            small_prefill_q4_single_linear(p.weight,codes,p.output,32,stream,projection_partials_,partial_rows_capacity_*32*4);
        else small_prefill_half_raw_linear(p.weight,codes,p.output,32,stream,pp_->multiprocessors(),
                                          {projection_partials_,partial_rows_capacity_*32*4},false);
    }
}
void RecurrentLayer::enable_q4_projection32(cudaStream_t stream,bool norm_pair,bool fused_ops){
    if((fused_ops&&!norm_pair)||q4projection32_||shape_.max_tokens!=32||tg_||!qkv32_||qkv32_extra_submissions_!=1)
        throw std::invalid_argument("Q4 projection stage requires explicit complete QKV and prepared FFN PP32 admission");
    auto stage=std::make_unique<SmallQ4Projection32>(std::array<Matrix,4>{weights_.z,weights_.alpha,weights_.beta,weights_.out},stream);
    workspace_bytes_+=stage->bytes();q4projection32_=std::move(stage);q4_norm_pair_=norm_pair;fused_ops_=fused_ops;
}
int RecurrentLayer::projection_partitions(Matrix w,int tokens,ReductionSchedule p)const{
    const size_t tiles=size_t((w.rows+63)/64)*size_t((tokens+63)/64);
    return schedule_.wide_grid&&tokens>=64&&w.rows>=w.columns&&tiles>=size_t(8*pp_->multiprocessors())?1:p.partitions;
}
void RecurrentLayer::project(Matrix w, const void *input, float *output, int tokens, ReductionSchedule p, cudaStream_t stream) {
    matrix_bytes(w);
    const int active_parts=tokens>=64?projection_partitions(w,tokens,p):p.partitions;
    // The shared arena is sized for max_tokens; partials consume the active span.
    if(!input || !output || tokens<1 || tokens>shape_.max_tokens || w.columns>input_capacity_ ||
       size_t(w.rows)*w.columns>decoded_capacity_ || p.partitions<1 || p.partitions>32 ||
       (tokens>=64 && active_parts>1 && size_t(w.rows)*active_parts*size_t(tokens)>
        partial_rows_capacity_*size_t(shape_.max_tokens)))
        throw std::invalid_argument("shared projection staging exceeded");
    if (tokens == 1) vector_linear(w, input, output, stream);
    else if(tiny_batches(tokens))tiny_batch_linear(w,input,output,tokens,stream);
    else if(tokens<64)small_prefill_linear(w,input,output,tokens,stream,pp_->multiprocessors());
    else {
        p.partitions=active_parts;
        const auto &scratch = pp_->buffers();
        decode_linear(w, scratch.half_weights, stream);
        if(pp_gemm_&&tokens>=128&&w.rows>=128){
            pp_gemm_->project(scratch.half_weights,input,output,w.rows,w.columns,tokens,stream);return;
        }
        const bool joint=joint_projections_&&tokens>=128&&w.rows>=128&&w.columns%(p.partitions*64)==0;
        if(joint&&p.partitions>1)
            dense_joint_split(scratch.half_weights,input,output,projection_partials_,w.rows,w.columns,tokens,p.partitions,stream);
        else if(joint) dense_joint_linear(scratch.half_weights,input,output,w.rows,w.columns,tokens,stream);
        else if (p.partitions > 1)
            dense_partitioned(scratch.half_weights, input, output, projection_partials_,
                              w.rows, w.columns, tokens, p.partitions, p.granularity, stream);
        else dense_linear(scratch.half_weights, input, output, w.rows, w.columns, tokens, stream);
    }
}
void RecurrentLayer::execute(RecurrentInputs input, RecurrentStateOutputs next, int tokens, cudaStream_t stream,const PhaseMarks *marks) {
    const auto &s = shape_; const auto &w = weights_; auto &b = buffers_;
    if(!tg_ && tokens!=32)throw std::invalid_argument("prepared FFN layer supports admitted PP32 service only");
    if (!input.hidden || !input.conv_history || !input.state || !next.conv_history || !next.state ||
        input.conv_history == next.conv_history || tokens < 1 || tokens > s.max_tokens)
        throw std::invalid_argument("invalid own recurrent layer invocation/lifetimes");
    if(marks)for(auto event:*marks)if(!event)throw std::invalid_argument("incomplete recurrent diagnostic event lease");
    auto mark=[&](int i){if(marks)check_cuda(cudaEventRecord((*marks)[i],stream),"bounded recurrent diagnostic mark");};
    const bool vector = tokens == 1;
    const bool active_q4=q4projection32_&&tokens==32,active_fused=fused_ops_&&active_q4;
    const bool packed=fused_pack32_&&tokens==32&&w.qkv.quant==Quant::Q6K;
    if(native32_&&tokens==32&&stream!=native32_stream_)throw std::invalid_argument("native32 setup stream changed");
    const int inner = s.dim * s.value_heads, channels = s.dim * (2 * s.query_heads + s.value_heads);
    const auto &scratch = vector ? tg_->buffers() : pp_->buffers();
    mark(0);
    void *joint_q6=packed?(qkv32_?qkv32_->encoding_input(stream):scratch.half_weights):nullptr;
    if(packed)q4projection32_->prepare_rms(input.hidden,w.norm,b.norm,nullptr,s.hidden,1,s.epsilon,joint_q6,stream);
    else rms_norm(input.hidden, w.norm, b.norm, s.hidden, tokens, s.epsilon, stream);
    void *prepared = vector ? scratch.quant_input : scratch.half_input;
    if(packed){}
    else if (vector) vector_quantize(b.norm, prepared, s.hidden, stream);
    else if(active_q4&&q4_norm_pair_)q4projection32_->prepare(b.norm,s.hidden,stream);
    else if(tiny_batches(tokens))tiny_batch_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.norm,prepared,s.hidden,tokens,stream);
    else to_half(b.norm, prepared, size_t(s.hidden) * tokens, stream);
    if(active_q4&&!q4_norm_pair_)q4projection32_->prepare(b.norm,s.hidden,stream);
    mark(1);
    if(native32_&&tokens==32){
        if(qkv32_)qkv32_->execute(w.qkv,b.norm,b.qkv,stream,packed);
        else{
        const void *qkv_input=q4projection32_->prepared_input(s.hidden,stream);
        if(w.qkv.quant==Quant::Q6K){qkv_input=scratch.half_weights;
            if(!packed)small_prefill_half_raw_quantize(b.norm,scratch.half_weights,w.qkv.quant,s.hidden,32,stream);}
        if(qkv_plane_data_)small_prefill_q6_planes_linear({qkv_plane_data_,w.qkv.rows,w.qkv.columns,qkv_plane_bytes_},qkv_input,b.qkv,stream);
        else if(w.qkv.quant==Quant::Q6K)small_prefill_q6_raw_prefetch_linear(w.qkv,qkv_input,b.qkv,stream);
        else if(w.qkv.quant==Quant::Q4K)small_prefill_q4_single_linear(w.qkv,qkv_input,b.qkv,32,stream);
        else small_prefill_half_raw_linear(w.qkv,qkv_input,b.qkv,32,stream,pp_->multiprocessors(),
                                     {projection_partials_,partial_rows_capacity_*32*4});
        }
    }
    else if(qkv32_&&tokens==32)qkv32_->execute(w.qkv,b.norm,b.qkv,stream);
    else project(w.qkv, prepared, b.qkv, tokens, schedule_.projections[0], stream);
    mark(2);
    if(active_q4)q4projection32_->linear(w.z,b.z,stream);
    else project(w.z, prepared, b.z, tokens, schedule_.projections[1], stream);
    mark(3);
    if(active_fused)q4projection32_->small_pair(b.alpha,b.beta_raw,stream);
    else if(active_q4&&q4_norm_pair_){q4projection32_->linear(w.alpha,b.alpha,stream);q4projection32_->linear(w.beta,b.beta_raw,stream);}
    else{project(w.alpha, prepared, b.alpha, tokens, schedule_.projections[2], stream);
         project(w.beta, prepared, b.beta_raw, tokens, schedule_.projections[3], stream);}
    mark(4);
    causal_conv4(b.qkv, input.conv_history, w.conv, b.conv_raw, b.conv_silu, next.conv_history, channels, tokens, stream);
    if(next.prefix.state[0])conv_prefix_history(b.qkv,input.conv_history,next.prefix,channels,tokens,stream);
    if(active_fused)gdn_prepare(b.conv_silu,b.alpha,b.beta_raw,w.dt,w.a,b.q,b.k,b.v,b.g,b.beta,
                              s.dim,s.query_heads,s.value_heads,tokens,s.epsilon,stream);
    else {
    gdn_split(b.conv_silu, b.q, b.k, b.v, s.dim, s.query_heads, s.value_heads, tokens, s.epsilon, stream);
    gdn_parameters(b.alpha, b.beta_raw, w.dt, w.a, b.g, b.beta, s.value_heads, tokens, stream);
    }
    if(next.prefix.state[0])
        gdn_recurrence_prefix({b.q,b.k,b.v,b.g,b.beta,input.state},
            {s.dim,s.dim,s.query_heads,s.value_heads,tokens},b.attention,next.state,next.prefix,stream);
    else if(tokens>=64&&pp_gdn_kernel_){
        const GdnShape shape{s.dim,s.dim,s.query_heads,s.value_heads,tokens};
        const size_t bytes=pp_->half_weight_capacity();
        if(pp_gdn_workspace_(shape)>bytes)throw std::runtime_error("PP GDN exceeds idle decoded-weight lease");
        pp_gdn_kernel_({b.q,b.k,b.v,b.g,b.beta,input.state},shape,b.attention,next.state,
                       scratch.half_weights,bytes,stream);
    }
    else if(tokens>=64&&pp_gdn_rows_==4)
        gdn_recurrence_rows({b.q,b.k,b.v,b.g,b.beta,input.state},
                           {s.dim,s.dim,s.query_heads,s.value_heads,tokens},b.attention,next.state,4,stream);
    else gdn_recurrence({b.q, b.k, b.v, b.g, b.beta, input.state},
                        {s.dim, s.dim, s.query_heads, s.value_heads, tokens}, b.attention, next.state, stream);
    if(packed)q4projection32_->prepare_rms(b.attention,w.gdn_norm,b.gated,b.z,s.dim,s.value_heads,s.epsilon,nullptr,stream);
    else rms_norm(b.attention, w.gdn_norm, b.gated, s.dim, s.value_heads * tokens, s.epsilon, stream, b.z);
    mark(5);
    if(packed){}
    else if (vector) vector_quantize(b.gated, prepared, inner, stream);
    else if(active_q4)q4projection32_->prepare(b.gated,inner,stream);
    else if(tiny_batches(tokens))tiny_batch_quantize(b.gated,prepared,inner,tokens,stream);
    else if(tokens<64)small_prefill_quantize(b.gated,prepared,inner,tokens,stream);
    else to_half(b.gated, prepared, size_t(inner) * tokens, stream);
    mark(6);
    if(active_q4)q4projection32_->linear(w.out,b.projected,stream);
    else project(w.out, prepared, b.projected, tokens, schedule_.projections[4], stream);
    mark(7);
    if(active_fused)add_rms_norm(b.projected,input.hidden,w.post_norm,b.residual,b.post_norm,s.hidden,tokens,s.epsilon,stream);
    else {
    add(b.projected, input.hidden, b.residual, size_t(s.hidden) * tokens, stream);
    rms_norm(b.residual, w.post_norm, b.post_norm, s.hidden, tokens, s.epsilon, stream);
    }
    mark(8);
    Ffn &ffn = *(vector ? tg_ : pp_);
    ffn.execute(b.post_norm, tokens, stream);
    mark(9);
    add(ffn.buffers().output, b.residual, b.output, size_t(s.hidden) * tokens, stream);
    mark(10);
}
}
