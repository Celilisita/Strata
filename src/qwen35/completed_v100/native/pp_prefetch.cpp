#include "pp_prefetch_core.hpp"
#include "strata/qwen35/completed_v100/native/gdn_subwarp16.hpp"
#include <algorithm>
namespace strata::completed_v100::qwen35::native {
namespace {
bool same(Matrix a,Matrix b){return a.data==b.data&&a.quant==b.quant&&a.layout==b.layout&&a.rows==b.rows&&a.columns==b.columns;}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
void separate(const void* a,size_t na,const void* b,size_t nb){if(!a||!b||overlap(a,na,b,nb))throw std::invalid_argument("PP prefetch overlapping live leases");}
}
PrefetchNvfp4::PrefetchNvfp4(size_t weights,int width,int tokens):Nvfp4Gemm(weights,width,tokens),core_(std::make_unique<PrefetchCore>(blas_lease())){}
PrefetchNvfp4::~PrefetchNvfp4()=default;
void PrefetchNvfp4::select(std::string& mode){core_->enabled=mode=="Q8A_P2PREF";if(core_->enabled)mode="Q8A_P2VEC";}
void PrefetchNvfp4::reset(cudaStream_t stream){core_->reset(stream);}
unsigned long long PrefetchNvfp4::recorded()const{return core_->calls;}
unsigned long long PrefetchNvfp4::builds()const{return core_->built;}
unsigned long long PrefetchNvfp4::hits()const{return core_->reused;}
bool PrefetchNvfp4::prepare(Matrix g,Matrix u,Matrix d,Buffers scratch,int maximum,GdnInputs x,GdnShape s,
    float* out,float* final,void* work,size_t bytes,PrefetchGdnKernel kernel,cudaStream_t stream){
    if(!core_->enabled||kernel!=gdn_recurrence_subwarp16_dual||(s.tokens!=1280&&s.tokens!=8192)||s.key_dim!=128||s.value_dim!=128||
       !q8_pp_fused()||!q8_pp_vector()||!q8_pp_pair()||!q8_pp_p2()||!q8_pp_decode8()||g.quant!=Quant::NVFP4||u.quant!=g.quant||d.quant!=g.quant)
        return false;
    if(core_->pending)throw std::logic_error("PP prefetch unconsumed preceding layer");
    original_matrix_bytes(g);original_matrix_bytes(u);original_matrix_bytes(d);
    if(g.rows!=u.rows||g.columns!=u.columns||d.rows!=g.columns||d.columns!=g.rows||maximum<s.tokens||maximum>8192||
       g.rows%128||g.columns%64||!scratch.activation||!scratch.half_input||!scratch.gate)
        throw std::invalid_argument("PP prefetch original matrix/arena geometry");
    const size_t decoded=size_t(g.rows)*g.columns*4,capacity=size_t(g.rows)*maximum*4;
    if(decoded>capacity)return false; // Existing GPU path remains valid for smaller arenas.
    const size_t encoded=size_t((std::max)(g.rows,g.columns))*s.tokens*2,packed=size_t(g.rows)*s.tokens*8;
    separate(scratch.activation,decoded,scratch.half_input,encoded);separate(scratch.activation,decoded,scratch.gate,packed);
    separate(scratch.half_input,encoded,scratch.gate,packed);
    const void* sources[]{x.q,x.k,x.v,x.log_decay,x.beta,x.state,out,final,work};
    const size_t qk=size_t(s.tokens)*s.query_heads*128*4,v=size_t(s.tokens)*s.value_heads*128*4;
    const size_t ab=size_t(s.tokens)*s.value_heads*4,state=size_t(s.value_heads)*128*128*4;
    const size_t spans[]{qk,qk,v,ab,ab,state,v,state,bytes};
    for(int i=0;i<9;++i){separate(scratch.activation,decoded,sources[i],spans[i]);separate(scratch.half_input,encoded,sources[i],spans[i]);}
    for(Matrix matrix:{g,u,d}){const size_t raw=original_matrix_bytes(matrix);
        separate(scratch.activation,decoded,matrix.data,raw);separate(scratch.half_input,encoded,matrix.data,raw);}
    core_->bind(stream);core_->lease={g,u,d,scratch,maximum,x,s,out,final,work,bytes};
    core_->gdn(stream);core_->pending=true;++core_->calls;return true;
}
bool PrefetchNvfp4::consume(Matrix g,Matrix u,Matrix d,const float* x,float* packed,size_t capacity,float* out,int tokens,cudaStream_t stream){
    if(!core_->pending)return false;
    const auto& l=core_->lease;
    if(!core_->enabled||!same(g,l.gate)||!same(u,l.up)||!same(d,l.down)||tokens!=l.shape.tokens||
       stream!=core_->main||packed!=l.scratch.gate||!x||!out||capacity<size_t(2)*g.rows*tokens)
        throw std::logic_error("PP prefetch consumed by foreign layer/request");
    separate(x,size_t(tokens)*g.columns*4,l.scratch.activation,size_t(g.rows)*g.columns*4);
    separate(out,size_t(tokens)*g.columns*4,l.scratch.activation,size_t(g.rows)*g.columns*4);
    const size_t hidden=size_t(tokens)*g.columns*4,encoded=size_t(tokens)*(std::max)(g.rows,g.columns)*2,packed_bytes=size_t(tokens)*g.rows*8;
    separate(x,hidden,l.scratch.half_input,encoded);separate(out,hidden,l.scratch.half_input,encoded);
    separate(x,hidden,packed,packed_bytes);separate(out,hidden,packed,packed_bytes);
    core_->ffn(x,packed,out,tokens,stream);core_->pending=false;record_prefetched_ffn();return true;
}
void pp_prefetch_select(DenseGemm* p,std::string& mode){if(auto* x=dynamic_cast<PrefetchNvfp4*>(p))x->select(mode);}
void pp_prefetch_reset(DenseGemm* p,cudaStream_t stream){if(auto* x=dynamic_cast<PrefetchNvfp4*>(p))x->reset(stream);}
bool try_pp_prefetch_gdn(DenseGemm* p,Matrix g,Matrix u,Matrix d,Buffers b,int maximum,GdnInputs x,GdnShape s,
    float* out,float* final,void* work,size_t bytes,PrefetchGdnKernel kernel,cudaStream_t stream){
    auto* provider=dynamic_cast<PrefetchNvfp4*>(p);return provider&&provider->prepare(g,u,d,b,maximum,x,s,out,final,work,bytes,kernel,stream);
}
bool try_pp_prefetch_ffn(DenseGemm* p,Matrix g,Matrix u,Matrix d,const float* x,float* packed,size_t capacity,float* out,int tokens,cudaStream_t stream){
    auto* provider=dynamic_cast<PrefetchNvfp4*>(p);return provider&&provider->consume(g,u,d,x,packed,capacity,out,tokens,stream);
}
unsigned long long pp_prefetch_recorded(DenseGemm* p){auto* x=dynamic_cast<PrefetchNvfp4*>(p);return x?x->recorded():0;}
unsigned long long pp_prefetch_builds(DenseGemm* p){auto* x=dynamic_cast<PrefetchNvfp4*>(p);return x?x->builds():0;}
unsigned long long pp_prefetch_hits(DenseGemm* p){auto* x=dynamic_cast<PrefetchNvfp4*>(p);return x?x->hits():0;}
}
