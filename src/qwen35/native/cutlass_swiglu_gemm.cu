#include "strata/qwen35/native/dense_gemm.hpp"
#include "cutlass_swiglu_tiles.cuh"
#include <climits>

namespace strata::qwen35::native {
namespace {
class FusedSwiGlu final:public DenseGemm {
    DenseGemm &fallback_;cudaStream_t stream_;
    int architecture_=0,tile_=0;unsigned long long fused_calls_=0;
    bool enabled_=true;
    CutlassTileInfo info_;
public:
    FusedSwiGlu(DenseGemm &fallback,cudaStream_t stream,int tile):fallback_(fallback),stream_(stream),tile_(tile){
        const auto workspace=fallback.borrowed_workspace();
        if(!stream||workspace.stream!=stream||tile<0||tile>6)
            throw std::invalid_argument("fused CUTLASS borrowed service lease invalid");
        int device=0;cudaDeviceProp caps{};check_cuda(cudaGetDevice(&device),"fused CUTLASS device");
        check_cuda(cudaGetDeviceProperties(&caps,device),"fused CUTLASS capabilities");architecture_=caps.major*10+caps.minor;
        if(architecture_<70)throw std::invalid_argument("fused Tensor Core provider requires SM70+");
        using namespace cutlass_detail;
        info_.tile=tile_;ConfigureTile configure{info_};
        if(architecture_<80)dispatch_tile<cutlass::arch::Sm70>(tile_,configure);
        else dispatch_tile<cutlass::arch::Sm80>(tile_,configure);
        if(tile_==5){
            CutlassTileInfo unused;ConfigureTile diagnostic{unused};
            if(architecture_<80)dispatch_tile<cutlass::arch::Sm70>(0,diagnostic);
            else dispatch_tile<cutlass::arch::Sm80>(0,diagnostic);
        }
    }
    ~FusedSwiGlu()override{cudaStreamSynchronize(stream_);}
    void select(bool enabled){enabled_=enabled;}
    CutlassTileInfo info()const{return info_;}
    bool can_fuse_swiglu(int r,int k,int t)const override{return enabled_&&r>0&&r<=INT_MAX/2&&r%64==0&&k>0&&k%32==0&&t>=128&&t%128==0;}
    void project_swiglu(const void *w,const void *x,void *y,float *logical,float *gate,float *up,
                        int r,int k,int t,cudaStream_t stream)override{
        if(!w||!x||!y||!logical||stream!=stream_||!can_fuse_swiglu(r,k,t)||bool(gate)!=bool(up)||
           (reinterpret_cast<uintptr_t>(y)%4)||y==x||y==w)
            throw std::invalid_argument("fused SwiGLU operand/stream geometry invalid");
        using namespace cutlass_detail;
        LaunchTile launch{w,x,y,logical,gate,up,r,k,t,stream};
        const int tile=tile_==5&&gate?0:tile_;
        if(architecture_<80)dispatch_tile<cutlass::arch::Sm70>(tile,launch);
        else dispatch_tile<cutlass::arch::Sm80>(tile,launch);
        ++fused_calls_;
    }
    void project(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{fallback_.project(w,x,y,r,k,t,s);}
    bool project_pair(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{return fallback_.project_pair(w,x,y,r,k,t,s);}
    void batched_fp16(const void *a,const void *b,float *c,GemmBatch p,cudaStream_t s)override{fallback_.batched_fp16(a,b,c,p,s);}
    size_t workspace_bytes()const override{return fallback_.workspace_bytes();}
    unsigned long long calls()const override{return fused_calls_+fallback_.calls();}
    GemmWorkspace borrowed_workspace()const override{return fallback_.borrowed_workspace();}
};
}
std::unique_ptr<DenseGemm> make_cutlass_swiglu_gemm(DenseGemm &fallback,cudaStream_t stream,int tile){
    return std::make_unique<FusedSwiGlu>(fallback,stream,tile);
}
void select_cutlass_swiglu(DenseGemm &service,bool enabled){dynamic_cast<FusedSwiGlu &>(service).select(enabled);}
CutlassTileInfo cutlass_swiglu_info(DenseGemm &service){return dynamic_cast<FusedSwiGlu &>(service).info();}
}
