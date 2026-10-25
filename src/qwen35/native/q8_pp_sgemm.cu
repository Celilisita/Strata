#include "q8_pp_sgemm.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <utility>

namespace strata::qwen35::native {
namespace {
constexpr int Rows=48,Columns=5120,Pairs=48,Maximum=8192,Threshold=256;
constexpr size_t Elements=size_t(Rows)*Columns,RawBytes=Elements/32*34;
constexpr size_t Workspace=4u*1024*1024;
struct Pair {Matrix alpha,beta;};
void require(bool value,const char *what){if(!value)throw std::invalid_argument(what);}
void blas(cublasStatus_t value,const char *what){
    if(value!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(what)+": cuBLAS "+std::to_string(int(value)));
}
bool overlaps(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    require(x<=std::numeric_limits<uintptr_t>::max()-an&&y<=std::numeric_limits<uintptr_t>::max()-bn,"Q8 PP pointer extent overflow");
    return x<y+bn&&y<x+an;
}
std::array<Pair,Pairs> inventory(const strata::qwen35::Spec &s,const DenseWeights &w){
    require(s.n_layers==64&&s.layers.size()==64&&w.recurrent.size()==64&&
        s.n_embd==Columns&&s.ssm_rank==Rows&&s.n_mtp_layers==1,"Q36 Q8 PP complete model geometry");
    std::array<Pair,Pairs> result{};size_t count=0;
    for(size_t layer=0;layer<64;++layer){
        const auto &l=s.layers[layer];
        require(l.recurrent==(layer%4!=3),"Q36 Q8 PP recurrent layer order");
        if(!l.recurrent)continue;
        const auto &actual=w.recurrent[layer];
        for(int role=0;role<2;++role){
            const auto *tensor=role?l.beta:l.alpha;const Matrix m=role?actual.beta:actual.alpha;
            const auto name="blk."+std::to_string(layer)+(role?".ssm_beta.weight":".ssm_alpha.weight");
            require(tensor&&tensor->name==name&&tensor->type==8&&
                tensor->shape==std::vector<uint64_t>{Columns,Rows},"Q36 original alpha/beta tensor identity");
            require(m.data&&m.quant==Quant::Q8&&m.layout==BlockLayout::Gguf&&m.rows==Rows&&m.columns==Columns&&
                reinterpret_cast<uintptr_t>(m.data)%2==0&&matrix_bytes(m)==RawBytes,"Q36 retained raw Q8 matrix binding");
        }
        require(count<result.size(),"Q36 Q8 PP pair count overflow");
        result[count++]={actual.alpha,actual.beta};
    }
    require(count==Pairs,"Q36 Q8 PP requires all48 pairs before use");
    for(size_t i=0;i<Pairs;++i)for(auto a:{result[i].alpha,result[i].beta}){
        for(size_t j=0;j<i;++j)for(auto b:{result[j].alpha,result[j].beta})
            require(!overlaps(a.data,RawBytes,b.data,RawBytes),"Q36 Q8 PP duplicate borrowed raw payload");
        require(!overlaps(result[i].alpha.data,RawBytes,result[i].beta.data,RawBytes),"Q36 Q8 PP alpha/beta alias");
    }
    return result;
}
__global__ void decode(const unsigned char *const *raw,float *cache,size_t count,unsigned *invalid){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const size_t matrix=i/Elements,element=i%Elements;
    const auto *block=raw[matrix]+element/32*34;
    const float value=__fmul_rn(__half2float(*reinterpret_cast<const half *>(block)),
        float(reinterpret_cast<const signed char *>(block+2)[element%32]));
    if(!isfinite(value))atomicOr(invalid,1u);
    cache[i]=value; // Exact original GGUF half-delta times signed8; no recoding.
}
__global__ void scatter(const float *both,float *a,float *b,int tokens){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*Rows)return;
    const size_t token=i/Rows,row=i%Rows;
    a[i]=both[token*2*Rows+row];b[i]=both[token*2*Rows+Rows+row];
}
}
struct Q8PpSgemm::Impl {
    const std::array<Pair,Pairs> pairs;
    const int maximum;
    const cudaStream_t stream;
    DeviceStorage storage;
    cublasHandle_t handle=nullptr;
    float *cache=nullptr,*combined=nullptr,*scalars=nullptr;
    void *workspace=nullptr;
    struct Ready {
        const float *x=nullptr;float *a=nullptr,*b=nullptr;
        std::bitset<Maximum+1> tokens;
    };
    std::array<Ready,Pairs> ready{};
    Q8PpSgemmStats measured;
    Impl(const strata::qwen35::Spec &s,const DenseWeights &w,int max_tokens,cudaStream_t ordered)
        :pairs(inventory(s,w)),maximum(max_tokens),stream(ordered){
        require(maximum>=Threshold&&maximum<=Maximum,"Q36 Q8 PP capacity");
        cudaStreamCaptureStatus capture{};
        check_cuda(cudaStreamIsCapturing(stream,&capture),"Q8 PP cold capture check");
        require(capture==cudaStreamCaptureStatusNone,"Q8 PP setup must precede Graph capture");
        const auto start=std::chrono::steady_clock::now();size_t total=0;
        check_cuda(cudaMemGetInfo(&measured.free_before,&total),"Q8 PP pre-allocation budget");
        measured.cache_bytes=Pairs*2*Elements*4;measured.workspace_bytes=Workspace;
        measured.scratch_bytes=size_t(maximum)*2*Rows*4;
        measured.cold_transient_bytes=Pairs*2*sizeof(void *)+4;
        DeviceStorage cold;
        std::array<const unsigned char *,Pairs*2> raw{};
        const float coefficients[]={1,0};unsigned bad=0;
        try{
            cache=static_cast<float *>(storage.allocate(measured.cache_bytes));
            combined=static_cast<float *>(storage.allocate(measured.scratch_bytes));
            scalars=static_cast<float *>(storage.allocate(8));workspace=storage.allocate(Workspace);
            auto **device_raw=static_cast<const unsigned char **>(cold.allocate(Pairs*2*sizeof(void *)));
            auto *invalid=static_cast<unsigned *>(cold.allocate(4));
            for(size_t i=0;i<Pairs;++i){raw[2*i]=static_cast<const unsigned char *>(pairs[i].alpha.data);
                raw[2*i+1]=static_cast<const unsigned char *>(pairs[i].beta.data);}
            check_cuda(cudaMemcpyAsync(scalars,coefficients,8,cudaMemcpyHostToDevice,stream),"Q8 PP coefficients upload");
            check_cuda(cudaMemcpyAsync(device_raw,raw.data(),sizeof(raw),cudaMemcpyHostToDevice,stream),"Q8 PP complete raw pointer table");
            check_cuda(cudaMemsetAsync(invalid,0,4,stream),"Q8 PP finite flag reset");
            decode<<<unsigned((Pairs*2*Elements+255)/256),256,0,stream>>>(device_raw,cache,Pairs*2*Elements,invalid);
            check_cuda(cudaGetLastError(),"Q8 PP exact F32 decode");
            check_cuda(cudaMemcpyAsync(&bad,invalid,4,cudaMemcpyDeviceToHost,stream),"Q8 PP finite flag readback");
            check_cuda(cudaStreamSynchronize(stream),"Q8 PP cold source/cache lifetime");
            if(bad)throw std::runtime_error("Q8 PP nonfinite original decoded weight");
            cold.release(device_raw);cold.release(invalid);
            blas(cublasCreate(&handle),"Q8 PP cuBLAS create");
            blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH),"Q8 PP full F32 policy");
            blas(cublasSetAtomicsMode(handle,CUBLAS_ATOMICS_NOT_ALLOWED),"Q8 PP atomic policy");
            blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_DEVICE),"Q8 PP stable F32 coefficients");
            blas(cublasSetStream(handle,stream),"Q8 PP fixed ordered stream");
            blas(cublasSetWorkspace(handle,workspace,Workspace),"Q8 PP bounded workspace");
            cublasMath_t mode{};blas(cublasGetMathMode(handle,&mode),"Q8 PP verify full F32 policy");
            if(mode!=CUBLAS_PEDANTIC_MATH)throw std::runtime_error("Q8 PP pedantic F32 policy not retained");
            check_cuda(cudaMemGetInfo(&measured.free_after,&total),"Q8 PP post-allocation budget");
            if(measured.free_after<DeviceStorage::reserve_bytes())throw std::runtime_error("Q8 PP lacks configured GPU reserve");
            measured.resident_bytes=storage.bytes();
            measured.cold_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }catch(...){cudaStreamSynchronize(stream);if(handle)cublasDestroy(handle);handle=nullptr;throw;}
    }
    ~Impl(){cudaStreamSynchronize(stream);if(handle)cublasDestroy(handle);}
    size_t find(Matrix alpha,Matrix beta)const{
        for(size_t i=0;i<Pairs;++i)if(alpha.data==pairs[i].alpha.data&&beta.data==pairs[i].beta.data){
            for(auto m:{alpha,beta})require(m.quant==Quant::Q8&&m.layout==BlockLayout::Gguf&&m.rows==Rows&&m.columns==Columns,
                "Q8 PP borrowed identity geometry changed");
            return i;
        }
        throw std::invalid_argument("Q8 PP pair absent from this Runtime's immutable inventory");
    }
    void admit_operands(const float *x,float *a,float *b)const{
        // Once per pair, before its first eager call. Use full fixed capacities
        // so future shapes cannot extend a live view into another allocation.
        const size_t in=size_t(maximum)*Columns*4,out=size_t(maximum)*Rows*4;
        require(!overlaps(x,in,a,out)&&!overlaps(x,in,b,out)&&!overlaps(a,out,b,out),"Q8 PP fixed operand capacities alias");
        for(auto region:{std::pair<const void *,size_t>{cache,measured.cache_bytes},
            {combined,measured.scratch_bytes},{workspace,Workspace},{scalars,8}}){
            require(!overlaps(x,in,region.first,region.second)&&!overlaps(a,out,region.first,region.second)&&
                !overlaps(b,out,region.first,region.second),"Q8 PP operands alias auxiliary storage");
        }
        for(const auto &pair:pairs)for(auto raw:{pair.alpha,pair.beta})
            require(!overlaps(x,in,raw.data,RawBytes)&&!overlaps(a,out,raw.data,RawBytes)&&
                !overlaps(b,out,raw.data,RawBytes),"Q8 PP operands alias retained raw weight storage");
    }
    void project(Matrix alpha,Matrix beta,const float *x,float *a,float *b,int tokens,cudaStream_t s){
        require(s==stream&&tokens>=Threshold&&tokens<=maximum&&x&&a&&b&&
            reinterpret_cast<uintptr_t>(x)%4==0&&reinterpret_cast<uintptr_t>(a)%4==0&&reinterpret_cast<uintptr_t>(b)%4==0,
            "Q8 PP ordered operands/span");
        const size_t in=size_t(tokens)*Columns*4,out=size_t(tokens)*Rows*4;
        require(!overlaps(x,in,a,out)&&!overlaps(x,in,b,out)&&!overlaps(a,out,b,out)&&
            !overlaps(x,in,cache,measured.cache_bytes)&&!overlaps(a,out,cache,measured.cache_bytes)&&
            !overlaps(b,out,cache,measured.cache_bytes),"Q8 PP aliased live operands");
        const size_t index=find(alpha,beta);auto &lease=ready[index];
        cudaStreamCaptureStatus capture{};
        check_cuda(cudaStreamIsCapturing(stream,&capture),"Q8 PP eager/capture readiness");
        if(lease.x)require(lease.x==x&&lease.a==a&&lease.b==b,"Q8 PP fixed recurrent operand lifetime changed");
        if(capture!=cudaStreamCaptureStatusNone)
            require(lease.tokens.test(tokens),"Q8 PP graph shape requires its previous normal eager execution");
        if(!lease.x)admit_operands(x,a,b);
        const float *weights=cache+index*2*Elements;
        // Row-major [96,K] and [T,K], ordinary F32 operands/output and pedantic
        // F32 accumulation. No TF32, half activation, quantization or input cache.
        blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,2*Rows,tokens,Columns,
            scalars,weights,Columns,x,Columns,scalars+1,combined,2*Rows),"Q8 PP stacked pedantic SGEMM");
        scatter<<<unsigned((size_t(tokens)*Rows+255)/256),256,0,stream>>>(combined,a,b,tokens);
        check_cuda(cudaGetLastError(),"Q8 PP stacked output scatter");++measured.host_submissions;
        if(capture==cudaStreamCaptureStatusNone){
            if(!lease.tokens.test(tokens))++measured.eager_shapes;
            lease.x=x;lease.a=a;lease.b=b;lease.tokens.set(tokens);
        } // No prepare projection, synchronization, allocation or host transfer.
    }
};
bool q8_pp_sgemm_requested(){
    const char *value=std::getenv("STRATA_DENSE_Q8_PP_SGEMM");
    if(!value||std::string(value)=="0")return false;
    if(std::string(value)!="1")throw std::invalid_argument("Q8 PP SGEMM flag must be0/1");
    return true;
}
Q8PpSgemm::Q8PpSgemm(const strata::qwen35::Spec &s,const DenseWeights &w,int t,cudaStream_t stream)
    :impl_(std::make_unique<Impl>(s,w,t,stream)){}
Q8PpSgemm::~Q8PpSgemm()=default;
void Q8PpSgemm::project(Matrix a,Matrix b,const float *x,float *y,float *z,int t,cudaStream_t stream){impl_->project(a,b,x,y,z,t,stream);}
Q8PpSgemmStats Q8PpSgemm::stats()const{return impl_->measured;}
}
