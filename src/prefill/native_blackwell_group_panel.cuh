// Bounded512-K panels trade barriers for CTA occupancy on sparse expert GU.
// Same integer digit dots and quant-group scaled F32 sum as full-K group MMQ.
template<class F,bool Fp4,bool Active,bool F16,int Tokens>
__launch_bounds__(Tokens*4,2)
__global__ void group_panel_mmq(const NativeMmqMatrix* matrices,const NativeMmqTile* tiles,
    const int32_t* output_ids,float* output,int width,int rows,int outputs,int parts,
    NativeMmqEncodedInput encoded) {
    constexpr int Threads=Tokens*4,Panel=512,Groups=Panel/F::K;
    constexpr int E=F16?2:Fp4?8:4,AD=F16?1:Fp4?F::AD4:F::AD8,Words=F::K/E;
    constexpr int Stride=AD*Words+(AD*Words>=8?4:2),RawStride=2*int(sizeof(typename F::Block))+4;
    constexpr int Raw=16*RawStride,Meta=16*Groups*int(sizeof(uint2)),Staging=Raw>Meta?Raw:Meta;
    __shared__ __align__(16) unsigned char staging[Staging+16*Groups*Stride*4];
    auto* aq=reinterpret_cast<unsigned*>(staging+Staging);auto* wm=reinterpret_cast<uint2*>(staging);
    const int lane=int(threadIdx.x)%32,warp=int(threadIdx.x)/32,subtiles=64/Tokens;
    const auto t=Active?tiles[blockIdx.y/subtiles]:NativeMmqTile{int(blockIdx.z)/parts,int(blockIdx.y)/subtiles};
    const auto m=matrices[t.matrix];const int part=int(blockIdx.z)%parts;
    const int first=m.begin+t.tile*64+int(blockIdx.y%subtiles)*Tokens,token=first+warp*8;
    if(first>=m.end) return;
    const auto* weights=reinterpret_cast<const typename F::Block*>(uintptr_t(part?m.up:m.gate));
    const int row0=int(blockIdx.x)*16,row=lane/4,col=2*(lane%4),blocks=width/256;
    const int source_b=token+row<m.end?__ldg(encoded.source_ids+token+row):0;
    const int source0=token+col<m.end?__ldg(encoded.source_ids+token+col):0;
    const int source1=token+col+1<m.end?__ldg(encoded.source_ids+token+col+1):0;
    const auto* bq=static_cast<const unsigned*>(encoded.codes);
    const auto* bm=static_cast<const GroupInputMeta*>(encoded.metadata);
    float total[4]={};
    for(int panel=0;panel<width/Panel;++panel) {
        constexpr int RowWords=2*int(sizeof(typename F::Block))/4;
        for(int i=int(threadIdx.x);i<16*RowWords;i+=Threads) {
            const auto* original=reinterpret_cast<const unsigned*>(weights+size_t(row0+i/RowWords)*blocks+panel*2);
            reinterpret_cast<unsigned*>(staging)[(i/RowWords)*(RowWords+1)+i%RowWords]=original[i%RowWords];
        }
        __syncthreads();
        uint2 saved[4];
        for(int i=int(threadIdx.x),j=0;i<16*Groups;i+=Threads,++j) {
            const int g=i/16,r=i%16;
            const auto& w=reinterpret_cast<const typename F::Block*>(staging+r*RawStride)[g/(256/F::K)];
            saved[j]=F::meta(w,g%(256/F::K));
            auto code=[&](int k){return F::code(w,g%(256/F::K),k);};
#pragma unroll
            for(int d=0;d<AD;++d)
#pragma unroll
                for(int word=0;word<Words;++word) aq[i*Stride+d*Words+word]=group_code_word<Fp4,F16>(code,word,d,AD);
        }
        __syncthreads();
        for(int i=int(threadIdx.x),j=0;i<16*Groups;i+=Threads,++j) wm[i]=saved[j];
        __syncthreads();
        if(token<m.end) for(int g=0;g<Groups;++g) {
            const int input_group=panel*Groups+g;float dot[4]={};
            group_product<F,Fp4,F16>(aq,bq,g,input_group,source_b,encoded.source_rows,lane,dot);
            const auto x0=bm[size_t(input_group)*encoded.source_rows+source0],x1=bm[size_t(input_group)*encoded.source_rows+source1];
            const auto w0=wm[g*16+row],w1=wm[g*16+row+8];
            total[0]=__fadd_rn(total[0],F::finish(w0,x0.d8,x0.sum,dot[0]));
            total[1]=__fadd_rn(total[1],F::finish(w0,x1.d8,x1.sum,dot[1]));
            total[2]=__fadd_rn(total[2],F::finish(w1,x0.d8,x0.sum,dot[2]));
            total[3]=__fadd_rn(total[3],F::finish(w1,x1.d8,x1.sum,dot[3]));
        }
        __syncthreads(); // Protect next panel's raw load, including inactive tail warps.
    }
#pragma unroll
    for(int h=0;h<2;++h)
#pragma unroll
        for(int j=0;j<2;++j) if(token+col+j<m.end) {
            const int dst=output_ids[token+col+j];
            output[size_t(part)*rows*outputs+size_t(dst)*outputs+row0+row+h*8]=total[h*2+j];
        }
}
template<class F,bool Fp4,bool Active,bool F16=false,int Tokens=32>
void launch_group_panel(const NativeMmqMatrix* matrices,int groups,int parts,const int32_t* ids,
    float* output,int rows,int max_rows,int width,int outputs,void* stream,
    const NativeMmqTile* tiles,int tile_count,NativeMmqEncodedInput encoded) {
    const dim3 grid(outputs/16,(Active?tile_count:(max_rows+63)/64)*(64/Tokens),Active?parts:groups*parts);
    group_panel_mmq<F,Fp4,Active,F16,Tokens><<<grid,Tokens*4,0,static_cast<cudaStream_t>(stream)>>>(
        matrices,tiles,ids,output,width,rows,outputs,parts,encoded);
}
