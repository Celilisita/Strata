#include "q35_shared_prepare.hpp"
#include "nvfp4_q8_fastsoa.cuh"
#include <limits>
namespace strata::completed_v100::qwen35::native {
namespace {
constexpr int Width=5120,Tokens=3;
// Three int4 code planes and one float4 scale plane for each 16-value group.
constexpr size_t EncodedBytes=size_t(Width/16)*64;
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
bool aligned(const void* p,size_t n){return p&&reinterpret_cast<uintptr_t>(p)%n==0;}
}
void Q35SharedNvfp4::shared_prepare(bool enabled,cudaStream_t stream){
    if(shared_==enabled&&(!enabled||stream_==stream))return;
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(stream,&capture),"Q35 shared prepare policy capture state");
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::logic_error("Q35 shared prepare policy requires released graphs");
    if(epoch_==(std::numeric_limits<uint64_t>::max)())throw std::overflow_error("Q35 shared prepare epoch exhausted");
    shared_=enabled;stream_=enabled?stream:nullptr;++epoch_;
}
Q35SharedPrepareLease Q35SharedNvfp4::lease(const void* producer,const float* input,
    void* arena,size_t bytes,cudaStream_t stream)const{
    if(!shared_||stream!=stream_||!producer||!input||!arena)return {};
    return {this,producer,input,arena,bytes,stream,epoch_};
}
bool Q35SharedNvfp4::group(const Q35SharedPrepareLease& l,const void* producer,
    const GdnPrefillProjection* p,const float* input,int tokens,cudaStream_t stream){
    if(!shared_||!geometry_||tokens!=Tokens||!q8_activation()||!q8_fastsoa()||!p||!producer||
       stream!=stream_||l.owner!=this||l.producer!=producer||l.input!=input||
       l.stream!=stream||l.epoch!=epoch_||l.bytes<EncodedBytes||
       !aligned(l.arena,16)||!aligned(input,4))return false;
    const size_t input_bytes=size_t(Width)*Tokens*4;
    if(overlap(l.arena,EncodedBytes,input,input_bytes))return false;
    // No enqueue occurs until every member and all borrowed spans pass.
    for(int i=0;i<4;++i){
        const Matrix m=p[i].matrix;
        if(m.quant!=Quant::NVFP4||m.layout!=BlockLayout::Nvfp4Scale||
           m.columns!=Width||m.rows<1||!aligned(m.data,16)||!aligned(p[i].output,4))return false;
        const size_t raw=OriginalNvfp4HeaderBytes+size_t(m.rows)*(Width/64)*36;
        const size_t out=size_t(m.rows)*Tokens*4;
        if(overlap(l.arena,EncodedBytes,m.data,raw)||overlap(l.arena,EncodedBytes,p[i].output,out)||
           overlap(input,input_bytes,p[i].output,out))return false;
        for(int j=0;j<4;++j){
            const Matrix other=p[j].matrix;
            if(other.rows<1||!p[j].output)return false;
            if(j<i&&overlap(p[i].output,out,p[j].output,size_t(other.rows)*Tokens*4))return false;
            if(other.quant==Quant::NVFP4&&other.layout==BlockLayout::Nvfp4Scale&&other.columns==Width&&
               overlap(p[j].output,size_t(other.rows)*Tokens*4,m.data,raw))return false;
        }
    }
    // Original group16 max/rounding and warp-wide nonfinite propagation remain.
    // Every invocation, including every captured replay, writes fresh codes.
    nvfp4_q8fs_prepare(input,l.arena,Width,Tokens,stream);
    for(int i=0;i<4;++i){
        const auto* raw=static_cast<const unsigned char*>(p[i].matrix.data)+OriginalNvfp4HeaderBytes;
        nvfp4_q8fs_project(raw,static_cast<const float*>(p[i].matrix.data),nullptr,nullptr,
            l.arena,p[i].output,nullptr,nullptr,p[i].matrix.rows,Width,Tokens,stream);
    }
    ++groups_;return true;
}
bool try_q35_shared_prepare(DenseGemm* service,const void* producer,const GdnPrefillProjection* p,
    const float* input,void* arena,size_t bytes,int tokens,cudaStream_t stream){
    if(tokens!=Tokens)return false;
    auto* provider=dynamic_cast<Q35SharedNvfp4*>(service);
    if(!provider)return false;
    return provider->group(provider->lease(producer,input,arena,bytes,stream),producer,p,input,tokens,stream);
}
}
