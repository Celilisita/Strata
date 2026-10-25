// Own sparse Q4 Gate/Up product. Same compact source/plan and scaled K32 chain;
// one raw-s8 IMMA replaces the two exact FP8 activation-digit IMMA operations.
template<int Tile, bool Active>
__launch_bounds__(256, Tile == 64 ? 2 : 1)
__global__ void group_gu_int8(const NativeMmqMatrix* matrices, const NativeMmqTile* tiles,
    const int32_t* output_ids, float* output, int rows, NativeMmqEncodedInput encoded) {
    static_assert(Tile == 64 || Tile == 128);
    constexpr int Width=2048, Outputs=512, Panel=512, Groups=16, Stride=12, Subs=Tile/64;
    constexpr int RawStride=2*int(sizeof(Q4KBlock))+4, Raw=16*RawStride;
    __shared__ __align__(16) unsigned char staging[Raw+16*Groups*Stride*4];
    auto* aq=reinterpret_cast<unsigned*>(staging+Raw);
    auto* wm=reinterpret_cast<uint2*>(staging);
    const int lane=int(threadIdx.x)%32, warp=int(threadIdx.x)/32;
    const auto t=Active ? tiles[blockIdx.y] : NativeMmqTile{int(blockIdx.z)/2,int(blockIdx.y)};
    const auto m=matrices[t.matrix]; const int part=int(blockIdx.z)%2;
    const int first=m.begin+t.tile*Tile, row0=int(blockIdx.x)*16;
    if(first>=m.end) return; // Uniform CTA decision, before any shared barrier.
    const auto* weights=reinterpret_cast<const Q4KBlock*>(uintptr_t(part ? m.up : m.gate));
    const int row=lane/4, col=2*(lane%4), word=lane%4;
    int source_b[Subs],source0[Subs],source1[Subs];
#pragma unroll
    for(int sub=0;sub<Subs;++sub) {
        const int token=first+warp*8+sub*64;
        source_b[sub]=token+row<m.end ? __ldg(encoded.source_ids+token+row) : 0;
        source0[sub]=token+col<m.end ? __ldg(encoded.source_ids+token+col) : 0;
        source1[sub]=token+col+1<m.end ? __ldg(encoded.source_ids+token+col+1) : 0;
    }
    const auto* bq=static_cast<const unsigned*>(encoded.codes);
    const auto* bm=static_cast<const GroupInputMeta*>(encoded.metadata);
    float total[Subs][4]={};
    for(int panel=0;panel<Width/Panel;++panel) {
        constexpr int RowWords=2*int(sizeof(Q4KBlock))/4;
        for(int i=int(threadIdx.x);i<16*RowWords;i+=256) {
            const auto* original=reinterpret_cast<const unsigned*>(weights+
                size_t(row0+i/RowWords)*(Width/256)+panel*2);
            reinterpret_cast<unsigned*>(staging)[(i/RowWords)*(RowWords+1)+i%RowWords]=original[i%RowWords];
        }
        __syncthreads();
        const int cell=int(threadIdx.x), g=cell/16, r=cell%16;
        const auto& weight=reinterpret_cast<const Q4KBlock*>(staging+r*RawStride)[g/8];
        const uint2 saved=GroupQ4::meta(weight,g%8);
#pragma unroll
        for(int v=0;v<8;++v) {
            unsigned packed=0;
#pragma unroll
            for(int p=0;p<4;++p) packed|=unsigned(uint8_t(GroupQ4::code(weight,g%8,v*4+p)))<<(p*8);
            aq[cell*Stride+v]=packed;
        }
        __syncthreads();
        wm[cell]=saved;
        __syncthreads();
        for(int g=0;g<Groups;++g) {
            const int input_group=panel*Groups+g;
            const auto w0=wm[g*16+row],w1=wm[g*16+row+8];
            unsigned a[4]={aq[(g*16+row)*Stride+word],aq[(g*16+row+8)*Stride+word],
                aq[(g*16+row)*Stride+word+4],aq[(g*16+row+8)*Stride+word+4]};
#pragma unroll
            for(int sub=0;sub<Subs;++sub) if(first+warp*8+sub*64<m.end) {
                const size_t base=(size_t(input_group)*encoded.source_rows+source_b[sub])*8+word;
                unsigned b[2]={bq[base],bq[base+4]};float dot[4]={};
                blackwell_int8_mma(a,b,dot);
                const auto x0=bm[size_t(input_group)*encoded.source_rows+source0[sub]];
                const auto x1=bm[size_t(input_group)*encoded.source_rows+source1[sub]];
                total[sub][0]=__fadd_rn(total[sub][0],GroupQ4::finish(w0,x0.d8,x0.sum,dot[0]));
                total[sub][1]=__fadd_rn(total[sub][1],GroupQ4::finish(w0,x1.d8,x1.sum,dot[1]));
                total[sub][2]=__fadd_rn(total[sub][2],GroupQ4::finish(w1,x0.d8,x0.sum,dot[2]));
                total[sub][3]=__fadd_rn(total[sub][3],GroupQ4::finish(w1,x1.d8,x1.sum,dot[3]));
            }
        }
        __syncthreads(); // Every inactive tail warp protects the next raw panel.
    }
#pragma unroll
    for(int sub=0;sub<Subs;++sub)
#pragma unroll
        for(int h=0;h<2;++h)
#pragma unroll
            for(int j=0;j<2;++j) {
                const int token=first+warp*8+sub*64+col+j;
                if(token<m.end) output[size_t(part)*rows*Outputs+
                    size_t(output_ids[token])*Outputs+row0+row+h*8]=total[sub][h*2+j];
            }
}
template<int Tile, bool Active>
void launch_gu_int8(const NativeMmqMatrix* matrices,int groups,const int32_t* ids,float* output,
    int rows,int max_rows,void* stream,const NativeMmqTile* tiles,int tile_count,NativeMmqEncodedInput encoded) {
    const dim3 grid(32,Active ? tile_count : (max_rows+Tile-1)/Tile,Active ? 2 : groups*2);
    group_gu_int8<Tile,Active><<<grid,256,0,static_cast<cudaStream_t>(stream)>>>(matrices,tiles,ids,output,rows,encoded);
}
template<int Tile, bool Active> NativeBlackwellGroupResources gu_int8_resources() {
    cudaFuncAttributes a{};int active=0;
    CUDA_CHECK(cudaFuncGetAttributes(&a,group_gu_int8<Tile,Active>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,group_gu_int8<Tile,Active>,256,0));
    return {a.numRegs,256,active,a.localSizeBytes,a.sharedSizeBytes,0};
}
inline void gu_int8_spans(const std::array<std::pair<uintptr_t,size_t>,4>& spans) {
    for(size_t i=0;i<spans.size();++i) if(spans[i].second) {
        const auto [start,bytes]=spans[i];
        if(!start || start>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("GU INT8 span overflow/null");
        for(size_t j=0;j<i;++j) if(spans[j].second &&
            start<spans[j].first+spans[j].second && spans[j].first<start+bytes)
            throw std::invalid_argument("GU INT8 source/workspace/output alias");
    }
}
