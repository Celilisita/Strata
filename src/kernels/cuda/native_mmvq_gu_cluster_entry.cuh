// Included in strata::kernels beside the existing grouped entrypoints.
namespace {
struct TgGuClusterResourceCache {int v[12]{};int error=0;};
const TgGuClusterResourceCache& tg_gu_cluster_resource_cache(){
    static thread_local const TgGuClusterResourceCache cached=[] {
        TgGuClusterResourceCache r;cudaDeviceProp p{};cudaFuncAttributes a{};
        auto checked=[&](cudaError_t e){if(e==cudaSuccess)return true;r.error=int(e);return false;};
        if(!checked(cudaGetDevice(&r.v[0]))||!checked(cudaGetDeviceProperties(&p,r.v[0])))return r;
        r.v[1]=p.major;r.v[2]=p.minor;r.v[3]=p.multiProcessorCount;
        if(p.major!=7||p.minor!=5||p.multiProcessorCount!=72){r.v[8]=1;return r;}
        if(!checked(cudaFuncGetAttributes(&a,native_tg_gu_cluster_kernel)))return r;
        r.v[4]=a.binaryVersion;r.v[5]=a.numRegs;r.v[6]=a.maxThreadsPerBlock;
        r.v[9]=int(a.sharedSizeBytes);r.v[10]=int(a.localSizeBytes);
        if(a.binaryVersion!=75)r.v[8]|=2;
        if(a.localSizeBytes)r.v[8]|=4;
        if(a.sharedSizeBytes!=66*sizeof(float)||a.sharedSizeBytes>p.sharedMemPerBlock)r.v[8]|=8;
        if(a.numRegs<1||a.numRegs>255||a.numRegs*128>p.regsPerBlock||a.maxThreadsPerBlock<128)r.v[8]|=16;
        if(!r.v[8]){
            if(!checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.v[7],native_tg_gu_cluster_kernel,128,0)))return r;
            if(r.v[7]<1)r.v[8]|=32;
        }
        return r;
    }();
    return cached;
}
static thread_local unsigned long long tg_gu_cluster_submitted=0;
bool tg_gu_cluster_span(const void* p,size_t n){
    const auto at=reinterpret_cast<uintptr_t>(p);return p&&!(at&3)&&n<=UINTPTR_MAX-at;
}
}
extern "C" int strata_tg_gu_cluster_resources(int* values){
    if(!values)return -1;const auto& r=tg_gu_cluster_resource_cache();
    for(int i=0;i<12;++i)values[i]=r.v[i];return r.error;
}
unsigned long long native_tg_gu_cluster_submissions(){return tg_gu_cluster_submitted;}
bool native_tg_gu_cluster_launch(const unsigned long long* addresses,const int32_t* rows,
    const unsigned long long* down,const void* input,const void* shared_gate,const void* shared_up,
    float* hidden,float* shared_hidden,void* stream){
    validate_stream(stream);
    const void* ptrs[]={addresses,rows,down,input,shared_gate,shared_up,hidden,shared_hidden};
    const size_t bytes[]={16*8,16*4,8*8,64*sizeof(Q81Block),8*144*512,8*144*512,8*512*4,512*4};
    for(int i=0;i<8;++i){
        if(!tg_gu_cluster_span(ptrs[i],bytes[i]))throw std::invalid_argument("invalid fixed TG GU cluster span");
        if(i>=6)for(int j=0;j<i;++j){
            const auto a=reinterpret_cast<uintptr_t>(ptrs[i]),b=reinterpret_cast<uintptr_t>(ptrs[j]);
            if(a<b+bytes[j]&&b<a+bytes[i])throw std::invalid_argument("TG GU cluster output alias");
        }
    }
    const auto& r=tg_gu_cluster_resource_cache();
    if(r.error)throw std::runtime_error("TG GU cluster resource query failed");
    if(r.v[8])return false;
    native_tg_gu_cluster_kernel<<<dim3(512,9),dim3(WARP,4),0,static_cast<cudaStream_t>(stream)>>>(
        addresses,rows,down,static_cast<const Q81Block*>(input),shared_gate,shared_up,hidden,shared_hidden);
    launch_check();++tg_gu_cluster_submitted;return true;
}
