#include "strata/gemma4/native/qkv_single_t1.hpp"
#include "strata/gemma4/native/vector_striped_q4.hpp"
#include <cstdint>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Columns=5376;
constexpr size_t SourceBytes=size_t(Columns)*sizeof(float);
constexpr size_t PayloadBytes=size_t(Columns/32)*sizeof(SingleQ8Block);
struct Span {const void* data;size_t bytes;};
void validate(Span s,size_t alignment){
    const auto first=reinterpret_cast<std::uintptr_t>(s.data);
    if(!s.data||!s.bytes||first%alignment||s.bytes>std::numeric_limits<std::uintptr_t>::max()-first)
        throw std::invalid_argument("Gemma264 aligned span extent");
}
bool overlaps(Span a,Span b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.data),y=reinterpret_cast<std::uintptr_t>(b.data);
    return x<y+b.bytes&&y<x+a.bytes;
}
void binding(QkvSingleBinding b){
    if(!b.owner||!b.epoch||!b.stream)throw std::invalid_argument("Gemma264 missing live binding");
}
}
QkvSingleLease prepare_qkv_single_t1(const float* normalized,void* arena,size_t bytes,QkvSingleBinding b){
    binding(b);validate({normalized,SourceBytes},alignof(float));
    validate({arena,bytes},alignof(SingleQ8Block));
    if(bytes<PayloadBytes||overlaps({normalized,SourceBytes},{arena,bytes}))
        throw std::invalid_argument("Gemma264 Single40 producer arena lifetime");
    return {normalized,prepare_single_q8(normalized,arena,bytes,Columns,1,b.stream),b};
}
void project_qkv_single_t1(primitive::Matrix w,QkvSingleLease lease,float* output,QkvSingleBinding live){
    binding(live);const auto b=lease.binding;const auto& x=lease.input;
    if(b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream)
        throw std::invalid_argument("Gemma264 stale owner/epoch/stream");
    const bool q4=w.quant==primitive::Quant::Q4K&&
        (w.rows==2048||w.rows==4096||w.rows==8192||w.rows==16384);
    const bool q6=w.quant==primitive::Quant::Q6K&&w.rows==4096;
    if((!q4&&!q6)||w.layout!=primitive::BlockLayout::Gguf||w.columns!=Columns||
       x.columns!=Columns||x.tokens!=1||x.bytes!=PayloadBytes)
        throw std::invalid_argument("Gemma264 exact nativeQKV Single40 T1; head excluded");
    const Span spans[]{{w.data,primitive::matrix_bytes(w)},{lease.normalized,SourceBytes},
        {x.data,x.bytes},{output,size_t(w.rows)*sizeof(float)}};
    for(int i=0;i<4;++i)validate(spans[i],i==0?(q6?2:4):alignof(float));
    for(int i=0;i<4;++i)for(int j=0;j<i;++j)
        if(overlaps(spans[i],spans[j]))throw std::invalid_argument("Gemma264 operand/output alias");
    if(q4)linear_single_q4_striped_t1(w,x,output,live.stream);
    else linear_single_q8_fast(w,x,output,live.stream);
}
}
