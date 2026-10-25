// Each warp owns M16 x N8. Eight warps share decoded weights for N64.
// Quant-group integer dots are exact; only their F32 scaled sums reorder.
template<class F,bool Fp4,bool Active,bool F16=false>
__launch_bounds__(256,1)
__global__ void blackwell_group_mmq(const NativeMmqMatrix* matrices,
    const NativeMmqTile* tiles,const int32_t* output_ids,float* output,
    int width,int rows,int outputs,int parts,NativeMmqEncodedInput encoded) {
    constexpr int E=F16 ? 2 : Fp4 ? 8 : 4, AD=F16 ? 1 : Fp4 ? F::AD4 : F::AD8, Words=F::K/E;
    constexpr int Stride=AD*Words+(AD*Words>=8 ? 4 : 2);
    const int lane=int(threadIdx.x)%32, warp=int(threadIdx.x)/32;
    const auto t=Active ? tiles[blockIdx.y] : NativeMmqTile{int(blockIdx.z)/parts,int(blockIdx.y)};
    const auto m=matrices[t.matrix]; const int part=int(blockIdx.z)%parts;
    const int first=m.begin+t.tile*64, token=first+warp*8;
    if(first>=m.end) return;
    const auto* weights=reinterpret_cast<const typename F::Block*>(uintptr_t(part ? m.up : m.gate));
    const int blocks=width/256, groups=width/F::K, row0=int(blockIdx.x)*16;
    const int row_words=blocks*int(sizeof(typename F::Block))/4, stride=(row_words+1)*4;
    const int raw_bytes=16*stride, meta_bytes=16*groups*int(sizeof(uint2));
    const int staging_bytes=raw_bytes>meta_bytes ? raw_bytes : meta_bytes;
    extern __shared__ unsigned char staging[];
    auto* aq=reinterpret_cast<unsigned*>(staging+staging_bytes);
    const auto* original=reinterpret_cast<const unsigned*>(weights+size_t(row0)*blocks);
    for(int i=int(threadIdx.x);i<16*row_words;i+=256)
        reinterpret_cast<unsigned*>(staging)[(i/row_words)*(row_words+1)+i%row_words]=original[i];
    __syncthreads();
    // Each thread retains its metadata before the raw-byte storage is reused.
    uint2 saved[4];
    for(int i=int(threadIdx.x),j=0;i<16*groups;i+=256,++j) {
        const int g=i/16,r=i%16;
        const auto& w=reinterpret_cast<const typename F::Block*>(staging+r*stride)[g/(256/F::K)];
        saved[j]=F::meta(w,g%(256/F::K));
        auto code=[&](int k) { return F::code(w,g%(256/F::K),k); };
#pragma unroll
        for(int d=0;d<AD;++d)
#pragma unroll
            for(int word=0;word<Words;++word) aq[i*Stride+d*Words+word]=group_code_word<Fp4,F16>(code,word,d,AD);
    }
    __syncthreads();
    auto* wm=reinterpret_cast<uint2*>(staging);
    for(int i=int(threadIdx.x),j=0;i<16*groups;i+=256,++j) wm[i]=saved[j];
    __syncthreads();
    if(token>=m.end) return;
    const int row=lane/4,col=2*(lane%4),start=(lane%4)*E;
    const int source_b=token+row<m.end ? __ldg(encoded.source_ids+token+row) : 0;
    const int source0=token+col<m.end ? __ldg(encoded.source_ids+token+col) : 0;
    const int source1=token+col+1<m.end ? __ldg(encoded.source_ids+token+col+1) : 0;
    const auto* bq=static_cast<const unsigned*>(encoded.codes);
    const auto* bm=static_cast<const GroupInputMeta*>(encoded.metadata);
    float total[4]={};
#pragma unroll 1
    for(int g=0;g<groups;++g) {
        float dot[4]={};
        group_product<F,Fp4,F16>(aq,bq,g,g,source_b,encoded.source_rows,lane,dot);
        const auto x0=bm[size_t(g)*encoded.source_rows+source0];
        const auto x1=bm[size_t(g)*encoded.source_rows+source1];
        const auto w0=wm[g*16+row],w1=wm[g*16+row+8];
        total[0]=__fadd_rn(total[0],F::finish(w0,x0.d8,x0.sum,dot[0]));
        total[1]=__fadd_rn(total[1],F::finish(w0,x1.d8,x1.sum,dot[1]));
        total[2]=__fadd_rn(total[2],F::finish(w1,x0.d8,x0.sum,dot[2]));
        total[3]=__fadd_rn(total[3],F::finish(w1,x1.d8,x1.sum,dot[3]));
    }
#pragma unroll
    for(int h=0;h<2;++h)
#pragma unroll
        for(int j=0;j<2;++j) if(token+col+j<m.end) {
            const int dst=output_ids[token+col+j];
            output[size_t(part)*rows*outputs+size_t(dst)*outputs+row0+row+h*8]=total[h*2+j];
        }
}
#include "native_blackwell_group128.cuh"
template<class F,bool Fp4,bool Active,bool F16=false>
void launch_group_quant(const NativeMmqMatrix* matrices,int groups,int parts,const int32_t* ids,
    float* output,int rows,int max_rows,int width,int outputs,void* stream,
    const NativeMmqTile* tiles,int tile_count,NativeMmqEncodedInput encoded) {
    const char* setting=std::getenv("STRATA_BENCH_GROUP_PANEL_TOKENS");
    const int tokens=setting ? std::atoi(setting) : 0;
    if(setting && std::strcmp(setting,"0") && std::strcmp(setting,"32") && std::strcmp(setting,"64"))
        throw std::invalid_argument("group panel tokens require0,32or64");
    if(encoded.tile_rows==128) {
        if(tokens && tokens!=64) throw std::invalid_argument("group128 requires panel tokens0 or64");
        if constexpr(!Fp4 && !F16 && F::K==32 && F::AD8==1) {
            if(width==2048 && parts==2 && tokens==64)
                launch_group128<F,Active,true>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
            else launch_group128<F,Active>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        }
        else throw std::invalid_argument("group128 requires Q4/Q5 FP8");
        return;
    }
    if(encoded.tile_rows!=64) throw std::invalid_argument("group rows require64 or128");
    if(width==2048 && parts==2 && tokens) {
        if(tokens==32) launch_group_panel<F,Fp4,Active,F16,32>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        else launch_group_panel<F,Fp4,Active,F16,64>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        return;
    }
    constexpr int AD=F16 ? 1 : Fp4 ? F::AD4 : F::AD8;
    constexpr int Words=F::K/(F16 ? 2 : Fp4 ? 8 : 4), Stride=AD*Words+(AD*Words>=8 ? 4 : 2);
    const int raw=16*((width/256)*int(sizeof(typename F::Block))+4);
    const int meta=16*(width/F::K)*int(sizeof(uint2));
    const int bytes=(raw>meta ? raw : meta)+16*(width/F::K)*Stride*4;
    if(bytes>48*1024) CUDA_CHECK(cudaFuncSetAttribute(blackwell_group_mmq<F,Fp4,Active,F16>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    const dim3 grid(outputs/16,Active ? tile_count : (max_rows+63)/64,Active ? parts : groups*parts);
    blackwell_group_mmq<F,Fp4,Active,F16><<<grid,256,bytes,static_cast<cudaStream_t>(stream)>>>(
        matrices,tiles,ids,output,width,rows,outputs,parts,encoded);
}
