// Included in strata::kernels next to the original grouped entrypoints.
namespace {
struct TgPairedResources {int v[12]{};int error=0;};
template<typename Kernel>TgPairedResources tg_paired_resource(Kernel kernel,int threads){
    TgPairedResources r;cudaDeviceProp p{};cudaFuncAttributes a{};
    auto checked=[&](cudaError_t e){if(e==cudaSuccess)return true;r.error=int(e);return false;};
    if(!checked(cudaGetDevice(&r.v[0]))||!checked(cudaGetDeviceProperties(&p,r.v[0])))return r;
    r.v[1]=p.major;r.v[2]=p.minor;r.v[3]=p.multiProcessorCount;
    if(p.major!=7||p.minor!=5||p.multiProcessorCount!=72){r.v[8]=1;return r;}
    if(!checked(cudaFuncGetAttributes(&a,kernel)))return r;
    r.v[4]=a.binaryVersion;r.v[5]=a.numRegs;r.v[6]=a.maxThreadsPerBlock;
    r.v[9]=int(a.sharedSizeBytes);r.v[10]=int(a.localSizeBytes);
    if(a.binaryVersion!=75)r.v[8]|=2;
    if(a.localSizeBytes)r.v[8]|=4;
    if(a.sharedSizeBytes)r.v[8]|=8;
    if(a.numRegs<1||a.numRegs>255||a.numRegs*threads>p.regsPerBlock||a.maxThreadsPerBlock<threads)r.v[8]|=16;
    if(!r.v[8]){if(!checked(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.v[7],kernel,threads,0)))return r;if(!r.v[7])r.v[8]|=32;}
    return r;
}
const TgPairedResources& tg_paired_resource_cache(int type,int stage){
    // One Runtime-owned device/context per thread; no new per-token queries.
    static thread_local const TgPairedResources quant=tg_paired_resource(native_tg_paired_quant_kernel,256);
    if(!stage)return quant;
    if(type==12){static thread_local const auto v=tg_paired_resource(native_tg_paired_down_kernel<Q4KTraits>,128);return v;}
    if(type==13){static thread_local const auto v=tg_paired_resource(native_tg_paired_down_kernel<Q5KTraits>,128);return v;}
    static thread_local const auto v=tg_paired_resource(native_tg_paired_down_kernel<Q6KTraits>,128);return v;
}
static thread_local unsigned long long tg_paired_submitted=0;
bool tg_paired_span(const void* p,size_t n,size_t alignment=4){const auto a=reinterpret_cast<uintptr_t>(p);return p&&!(a&(alignment-1))&&n<=UINTPTR_MAX-a;}
}
extern "C" int strata_tg_quant_down_pair_resources(int type,int stage,int* values){
    if(!values||(type!=12&&type!=13&&type!=14)||(stage!=0&&stage!=1))return -1;
    const auto& r=tg_paired_resource_cache(type,stage);for(int i=0;i<12;++i)values[i]=r.v[i];return r.error;
}
unsigned long long native_tg_quant_down_pair_submissions(){return tg_paired_submitted;}
bool native_tg_quant_down_pair_launch(int type,const float* expert,const float* shared,
    const unsigned long long* addresses,const int32_t* rows,void* expert_q8,void* shared_q8,
    const void* shared_down,float* parts,float* shared_output,void* stream){
    if(type!=12&&type!=13&&type!=14)throw std::invalid_argument("unsupported fixed paired Down type");
    validate_stream(stream);
    const void* ptr[]={expert,shared,addresses,rows,shared_down,expert_q8,shared_q8,parts,shared_output};
    const size_t weight=size_t(2048)*2*(type==12?144:type==13?176:210);
    const size_t bytes[]={8*512*4,512*4,8*8,8*4,weight,8*16*sizeof(Q81Block),16*sizeof(Q81Block),8*2048*4,2048*4};
    for(int i=0;i<9;++i){if(!tg_paired_span(ptr[i],bytes[i],i==2?8:4))throw std::invalid_argument("invalid paired quant/Down span");
        if(i>=5)for(int j=0;j<i;++j){const auto a=reinterpret_cast<uintptr_t>(ptr[i]),b=reinterpret_cast<uintptr_t>(ptr[j]);
            if(a<b+bytes[j]&&b<a+bytes[i])throw std::invalid_argument("paired quant/Down output alias");}}
    for(int stage=0;stage<2;++stage){const auto& r=tg_paired_resource_cache(type,stage);
        if(r.error)throw std::runtime_error("paired quant/Down resource query failed");if(r.v[8])return false;}
    const auto s=static_cast<cudaStream_t>(stream);
    native_tg_paired_quant_kernel<<<18,256,0,s>>>(expert,shared,addresses,static_cast<Q81Block*>(expert_q8),static_cast<Q81Block*>(shared_q8));launch_check();
    const dim3 grid(512,9),threads(32,4);
    if(type==12)native_tg_paired_down_kernel<Q4KTraits><<<grid,threads,0,s>>>(addresses,rows,static_cast<const Q81Block*>(expert_q8),static_cast<const Q81Block*>(shared_q8),shared_down,parts,shared_output);
    else if(type==13)native_tg_paired_down_kernel<Q5KTraits><<<grid,threads,0,s>>>(addresses,rows,static_cast<const Q81Block*>(expert_q8),static_cast<const Q81Block*>(shared_q8),shared_down,parts,shared_output);
    else native_tg_paired_down_kernel<Q6KTraits><<<grid,threads,0,s>>>(addresses,rows,static_cast<const Q81Block*>(expert_q8),static_cast<const Q81Block*>(shared_q8),shared_down,parts,shared_output);
    launch_check();++tg_paired_submitted;return true;
}
