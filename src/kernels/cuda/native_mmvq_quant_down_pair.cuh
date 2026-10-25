// Included after existing traits; uses original native Q8 helpers and dot math.
__launch_bounds__(QUANT_THREADS,1)
__global__ void native_tg_paired_quant_kernel(const float* expert,const float* shared,
    const unsigned long long* addresses,Q81Block* expert_q8,Q81Block* shared_q8){
    pdl_trigger();pdl_wait();
    const int row=int(blockIdx.x)/2,i=(int(blockIdx.x)%2)*QUANT_THREADS+int(threadIdx.x);
    const bool is_shared=row==8;
    if(!is_shared&&!addresses[row])return; // Uniform CTA, before any inactive input read.
    const float* x=is_shared?shared:expert+size_t(row)*512;
    Q81Block* y=is_shared?shared_q8:expert_q8+size_t(row)*16;
    const float xi=x[i];
    const float amax=warp_max(fabsf(xi));
    const float sum=warp_sum(xi);
    const float d=q8_1_finite(amax/127.0f);
    const int8_t q=q8_1_quant(xi,d,amax);
    y[i/Q8K].qs[i%Q8K]=q;
    if(i%Q8K==0)y[i/Q8K].ds=q8_1_ds(d,sum);
}

template<typename F>
__launch_bounds__(4*WARP,1)
__global__ void native_tg_paired_down_kernel(const unsigned long long* addresses,
    const int32_t* rows,const Q81Block* expert_q8,const Q81Block* shared_q8,
    const void* shared_down,float* parts,float* shared_output){
    static_assert(F::DIV==256&&(F::T==16||F::T==32)&&WARP==32);
    const int entry=int(blockIdx.y),row=4*int(blockIdx.x)+int(threadIdx.y),lane=int(threadIdx.x);
    const bool is_shared=entry==8;
    const auto address=is_shared?reinterpret_cast<uintptr_t>(shared_down):uintptr_t(addresses[entry]);
    if(!address)return; // Original nullable owner; shared address is always nonnull.
    const auto* weights=reinterpret_cast<const typename F::Block*>(address);
    const auto* x=is_shared?shared_q8:expert_q8+size_t(rows[entry])*(512/Q8K);
    float value=0;
    for(int kbx=lane/F::T;kbx<512/F::DIV;kbx+=WARP/F::T){
        const int kqs=F::kqs(lane);
        const auto wv=F::load(weights+size_t(row)*(512/F::DIV)+kbx,kqs);
        value+=F::apply(wv,x+kbx*F::KBY,kqs);
    }
    value=warp_sum(value);
    if(lane==0){if(is_shared)shared_output[row]=value;else parts[size_t(entry)*2048+row]=value;}
}
