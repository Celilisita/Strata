template<bool F16,bool Fp4>
__device__ void group_mma(const unsigned* a,const unsigned* b,float* d,int ad,int bd) {
    if constexpr(F16) asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    else blackwell_mma<Fp4>(a,b,d,ad,bd);
}
template<class F,bool Fp4,bool F16=false>
__device__ void group_product(const unsigned* aq,const unsigned* bq,int weight_group,
    int input_group,int source_b,int source_rows,int lane,float* dot) {
    constexpr int E=F16 ? 2 : Fp4 ? 8 : 4, AD=F16 ? 1 : Fp4 ? F::AD4 : F::AD8;
    constexpr int BD=F16 ? 1 : Fp4 ? 3 : 2, Words=F::K/E, Step=F16 ? 16 : Fp4 ? 64 : 32;
    constexpr int Stride=AD*Words+(AD*Words>=8 ? 4 : 2);
#pragma unroll
    for(int k=0;k<F::K;k+=Step)
#pragma unroll
        for(int ad=0;ad<AD;++ad) {
            unsigned a[4]={};const int start=k+(lane%4)*E;
#pragma unroll
            for(int h=0;h<2;++h) {
                const int row=lane/4+h*8;
                if(start<F::K) a[h]=aq[(weight_group*16+row)*Stride+ad*Words+start/E];
                if(start+4*E<F::K) a[h+2]=aq[(weight_group*16+row)*Stride+ad*Words+start/E+4];
            }
#pragma unroll
            for(int bd=0;bd<BD;++bd) {
                unsigned b[2]={};const size_t base=(size_t(input_group)*source_rows+source_b)*BD+bd;
                if(start<F::K) b[0]=bq[base*Words+start/E];
                if(start+4*E<F::K) b[1]=bq[base*Words+start/E+4];
                group_mma<F16,Fp4>(a,b,dot,ad,bd);
            }
        }
}
