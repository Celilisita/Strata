// Two independent dense N64 subtiles share one512-K weight decode.
// No row map: token-local F32 group finishes retain their original order.
template<class F>
__launch_bounds__(256,2)
__global__ void dense_group_f16_128(const typename F::Block* weights,float* output,
    int width,int rows,int outputs,NativeMmqEncodedInput encoded) {
    constexpr int Panel=512,Groups=Panel/F::K,Words=F::K/2,Stride=Words+4;
    constexpr int RawStride=(Panel/256)*int(sizeof(typename F::Block))+4;
    constexpr int Raw=16*RawStride,Meta=16*Groups*int(sizeof(uint2));
    constexpr int Staging=Raw>Meta ? Raw : Meta;
    __shared__ __align__(16) unsigned char staging[Staging+16*Groups*Stride*4];
    auto* aq=reinterpret_cast<unsigned*>(staging+Staging);
    auto* wm=reinterpret_cast<uint2*>(staging);
    const auto* bq=static_cast<const unsigned*>(encoded.codes);
    const auto* bm=static_cast<const GroupInputMeta*>(encoded.metadata);
    const int lane=int(threadIdx.x)%32,warp=int(threadIdx.x)/32;
    const int row0=int(blockIdx.x)*16,first=int(blockIdx.y)*128+warp*8;
    const int row=lane/4,col=2*(lane%4);
    int source_b[2],source0[2],source1[2];
#pragma unroll
    for(int sub=0;sub<2;++sub) {
        const int token=first+sub*64;
        source_b[sub]=token+row<rows ? token+row : 0;
        source0[sub]=token+col<rows ? token+col : 0;
        source1[sub]=token+col+1<rows ? token+col+1 : 0;
    }
    float total[2][4]={};
    for(int panel=0;panel<width/Panel;++panel) {
        constexpr int RowWords=(Panel/256)*int(sizeof(typename F::Block))/4;
        for(int i=int(threadIdx.x);i<16*RowWords;i+=256) {
            const auto* original=reinterpret_cast<const unsigned*>(weights+size_t(row0+i/RowWords)*(width/256)+panel*2);
            reinterpret_cast<unsigned*>(staging)[(i/RowWords)*(RowWords+1)+i%RowWords]=original[i%RowWords];
        }
        __syncthreads();
        uint2 saved[2];
        for(int i=int(threadIdx.x),j=0;i<16*Groups;i+=256,++j) {
            const int g=i/16,r=i%16;
            const auto& w=reinterpret_cast<const typename F::Block*>(staging+r*RawStride)[g/(256/F::K)];
            saved[j]=F::meta(w,g%(256/F::K));
            auto code=[&](int k){return F::code(w,g%(256/F::K),k);};
#pragma unroll
            for(int word=0;word<Words;++word) aq[i*Stride+word]=group_code_word<false,true>(code,word,0,1);
        }
        __syncthreads();
        for(int i=int(threadIdx.x),j=0;i<16*Groups;i+=256,++j) wm[i]=saved[j];
        __syncthreads();
        for(int g=0;g<Groups;++g) {
            const int input_group=panel*Groups+g;
            const auto w0=wm[g*16+row],w1=wm[g*16+row+8];
#pragma unroll
            for(int sub=0;sub<2;++sub) if(first+sub*64<rows) {
                float dot[4]={};
                group_product<F,false,true>(aq,bq,g,input_group,source_b[sub],encoded.source_rows,lane,dot);
                const auto x0=bm[size_t(input_group)*encoded.source_rows+source0[sub]];
                const auto x1=bm[size_t(input_group)*encoded.source_rows+source1[sub]];
                total[sub][0]=__fadd_rn(total[sub][0],F::finish(w0,x0.d8,x0.sum,dot[0]));
                total[sub][1]=__fadd_rn(total[sub][1],F::finish(w0,x1.d8,x1.sum,dot[1]));
                total[sub][2]=__fadd_rn(total[sub][2],F::finish(w1,x0.d8,x0.sum,dot[2]));
                total[sub][3]=__fadd_rn(total[sub][3],F::finish(w1,x1.d8,x1.sum,dot[3]));
            }
        }
        __syncthreads(); // All warps, including both inactive tails, protect panel reuse.
    }
#pragma unroll
    for(int sub=0;sub<2;++sub)
#pragma unroll
        for(int h=0;h<2;++h)
#pragma unroll
            for(int j=0;j<2;++j) if(first+sub*64+col+j<rows)
                output[size_t(first+sub*64+col+j)*outputs+row0+row+h*8]=total[sub][h*2+j];
}
template<class F>
void launch_dense_group_f16_128(const void* weights,const void* input,void* codes,void* meta,
    float* output,int width,int rows,int outputs,void* stream) {
    const int cells=rows*(width/F::K);
    encode_group_input<F,false,true><<<(cells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(input),static_cast<unsigned*>(codes),static_cast<GroupInputMeta*>(meta),width,rows);
    NativeMmqEncodedInput encoded{codes,meta,nullptr,rows};
    dense_group_f16_128<F><<<dim3(outputs/16,(rows+127)/128),256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const typename F::Block*>(weights),output,width,rows,outputs,encoded);
}
