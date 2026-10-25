// Included after Q4KTraits in native_mmvq.cu's anonymous namespace.
// Keep the existing GU2 two-warp Q4 dot order; only pair Gate/Up of one feature.
__launch_bounds__(4*WARP,1)
__global__ void native_tg_gu_cluster_kernel(const unsigned long long* addresses,const int32_t* input_rows,
    const unsigned long long* down_addresses,const Q81Block* input_q8,
    const void* shared_gate,const void* shared_up,float* expert_hidden,float* shared_hidden){
    using F=Q4KTraits;
    static_assert(F::DIV==256&&F::T==16&&F::KBY==8&&WARP==32);
    const int entry=int(blockIdx.y),row=int(blockIdx.x),part=int(threadIdx.y)/2;
    const int local_warp=int(threadIdx.y)%2,lane=int(threadIdx.x),tid=local_warp*WARP+lane;
    const bool shared=entry==8;
    // Uniform CTA early return. This is the old masked SwiGLU ownership rule.
    if(!shared&&!down_addresses[entry])return;
    const auto gate=shared?reinterpret_cast<uintptr_t>(shared_gate):uintptr_t(addresses[entry]);
    const auto up=shared?reinterpret_cast<uintptr_t>(shared_up):uintptr_t(addresses[8+entry]);
    // Pointer coherence is the existing route/lease contract, not a new directory.
    const auto* weights=reinterpret_cast<const F::Block*>(part?up:gate);
    const int source=shared?0:input_rows[part*8+entry];
    const auto* x=input_q8+size_t(source)*(2048/Q8K);
    float value=0;
    for(int kbx=tid/F::T;kbx<2048/F::DIV;kbx+=2*WARP/F::T){
        const int kqs=F::kqs(tid);
        const auto wv=F::load(weights+size_t(row)*(2048/F::DIV)+kbx,kqs);
        value+=F::apply(wv,x+kbx*F::KBY,kqs);
    }
    __shared__ float scratch[66];
    if(local_warp)scratch[part*WARP+lane]=value;
    __syncthreads();
    if(!local_warp){
        value+=scratch[part*WARP+lane];value=warp_sum(value);
        if(lane==0)scratch[64+part]=value;
    }
    __syncthreads();
    if(threadIdx.y==0&&lane==0){
        const float g=scratch[64],u=scratch[65];
        const float h=(g/(1.0f+expf(-g)))*u;
        if(shared)shared_hidden[row]=h;else expert_hidden[size_t(entry)*512+row]=h;
    }
}
