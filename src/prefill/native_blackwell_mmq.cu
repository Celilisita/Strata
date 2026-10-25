#include "strata/prefill/native_blackwell_mmq.hpp"
#include "strata/prefill/native_mmq_tile_policy.hpp"
#include "strata/prefill/native_blackwell_dense_int8_policy.hpp"
#include "strata/prefill/native_blackwell_dense_wide_policy.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "common.cuh"
#include "mmq.cuh"
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <array>
#include <utility>
#include <limits>
namespace strata::prefill::mmq {
namespace {
constexpr int WARP = 32, WARPS = 4, QK = 256, Q8K = 32, QI = 32, VDR = 2;
using Q4KBlock = block_q4_K;
using Q5KBlock = block_q5_K;
static_assert(sizeof(Q5KBlock)==176 && offsetof(Q5KBlock,scales)==4 &&
    offsetof(Q5KBlock,qh)==16 && offsetof(Q5KBlock,qs)==48);
using Q6KBlock = block_q6_K;
using Q81Block = block_q8_1;
__device__ __forceinline__ int load_int_b2(const void* source, int word) {
    const auto* h = static_cast<const uint16_t*>(source)+2*word;
    return int(uint32_t(h[0]) | (uint32_t(h[1]) << 16));
}
#include "../kernels/cuda/native_k_dot.cuh"
#include "../kernels/cuda/native_q4_traits.cuh"
#include "../kernels/cuda/native_q6_traits.cuh"
#include "native_canonical_mmq.cuh"
#include "native_blackwell_digits.cuh"
#include "native_blackwell_convert.cuh"
#include "native_blackwell_int8.cuh"
#include "native_blackwell_panels.cuh"
#include "native_blackwell_input.cuh"
#include "native_blackwell_tile.cuh"
#include "native_blackwell_quant_group.cuh"
#include "native_blackwell_group_input.cuh"
#include "native_blackwell_swiglu_input.cuh"
#include "native_blackwell_group_product.cuh"
#include "native_blackwell_group_panel.cuh"
#include "native_blackwell_group_tile.cuh"
#include "native_blackwell_dense_group.cuh"
#include "native_blackwell_dense128.cuh"
#include "native_blackwell_dense_int8_group.cuh"
#include "native_blackwell_dense_wide_int8.cuh"
#include "native_blackwell_dense_i128_int8.cuh"
#include "native_blackwell_gu_int8.cuh"
#include "native_blackwell_gu_int8_i32.cuh"
#include "native_blackwell_gu_int8_i64.cuh"
#include "native_blackwell_down_int8.cuh"
void check_device() {
    int device = 0; cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&p,device));
    if (p.major != 12 || p.minor != 0) throw std::runtime_error("Blackwell digit MMQ requires SM120");
}
template<class F, bool Fp4, bool Int8 = false>
void launch_grouped(const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* ids, const void* input, float* output, int rows, int max_rows,
    int width, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count, const NativeMmqEncodedInput& encoded) {
    const int bytes = blackwell_panel_bytes<F,Fp4,Int8>(width);
    const dim3 grid(unsigned(outputs/8),unsigned(tiles ? tile_count : (max_rows+63)/64),
                    unsigned(tiles ? parts : groups*parts));
    if (bytes+8320 > 48*1024) {
        CUDA_CHECK(cudaFuncSetAttribute(blackwell_matrix_mmq<F,Fp4,true,true,Int8>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
        CUDA_CHECK(cudaFuncSetAttribute(blackwell_matrix_mmq<F,Fp4,true,false,Int8>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    }
    if (tiles) blackwell_matrix_mmq<F,Fp4,true,true,Int8><<<grid,256,bytes,static_cast<cudaStream_t>(stream)>>>(
        nullptr,matrices,tiles,ids,static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,outputs,parts,encoded);
    else blackwell_matrix_mmq<F,Fp4,true,false,Int8><<<grid,256,bytes,static_cast<cudaStream_t>(stream)>>>(
        nullptr,matrices,nullptr,ids,static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,outputs,parts,encoded);
}
}
template<class F> void launch_dense_int8(const void* weights, const NativeMmqEncodedInput& encoded,
    float* output,int width,int rows,int outputs,void* stream) {
    const int bytes=blackwell_panel_bytes<F,false,true>(width);
    if(bytes+8320>48*1024) CUDA_CHECK(cudaFuncSetAttribute(
        blackwell_matrix_mmq<F,false,false,false,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    blackwell_matrix_mmq<F,false,false,false,true><<<dim3(outputs/8,(rows+63)/64),256,bytes,
        static_cast<cudaStream_t>(stream)>>>(static_cast<const typename F::Block*>(weights),
        nullptr,nullptr,nullptr,nullptr,output,width,rows,outputs,outputs,1,encoded);
}
void run_blackwell_dense_int8(int type,const void* weights,const void* raw_q8,void* packed,
    size_t packed_bytes,float* output,int width,int rows,int outputs,void* stream) {
    check_device();
    if(!weights || !raw_q8 || !packed || !output || !stream || width<512 || width>4096 ||
        rows<1 || rows>8192 || outputs<8 || outputs>8192 || outputs%8 ||
        blackwell_input_bytes(type,width,rows)>packed_bytes)
        throw std::invalid_argument("unreserved Blackwell dense integer product");
    const auto encoded=repack_blackwell_input(raw_q8,packed,type,width,rows,nullptr,5,stream);
    if(type==GGML_TYPE_Q4_K) launch_dense_int8<BwQ4>(weights,encoded,output,width,rows,outputs,stream);
    else launch_dense_int8<BwQ6>(weights,encoded,output,width,rows,outputs,stream);
    CUDA_CHECK(cudaGetLastError());
}
bool blackwell_q5_tc_enabled() {
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_BENCH_Q5_TC");
        if (!value || !std::strcmp(value,"0")) return false;
        if (!std::strcmp(value,"1")) return true;
        throw std::invalid_argument("Q5 Tensor Core selector requires0 or1");
    }();
    return enabled;
}
int blackwell_dense_tile_rows(int type, int rows) {
    static const auto mode=native_mmq_tile_mode(std::getenv("STRATA_BENCH_DENSE_TILE_ROWS"), "Blackwell dense tile rows");
    return native_mmq_dense_tile_rows(mode, type, rows);
}
template<class F> void launch_dense_variant(const void* weights,const void* input,void* codes,void* meta,
    float* output,int width,int rows,int outputs,void* stream,int tile_rows) {
    if(tile_rows==128) launch_dense_group_f16_128<F>(weights,input,codes,meta,output,width,rows,outputs,stream);
    else launch_dense_group_f16<F>(weights,input,codes,meta,output,width,rows,outputs,stream);
}
void run_blackwell_dense_group_f16(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,void* stream) {
    run_blackwell_dense_group_f16_tile(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,blackwell_dense_tile_rows(type,rows));
}
static void validate_dense_group_views(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,void* stream,int tile_rows) {
    check_device();
    if(tile_rows!=64 && tile_rows!=128) throw std::invalid_argument("Blackwell dense tile rows require64 or128");
    if((type!=12 && type!=13 && type!=14) || !weights || !input || !codes || !meta || !output || !stream ||
        width<512 || width>4096 || width%512 || rows<1 || rows>4096 || outputs<16 || outputs>8192 || outputs%16 ||
        code_bytes<size_t(rows)*width*2 || meta_bytes<size_t(rows)*(width/(type==14 ? 16 : 32))*sizeof(GroupInputMeta))
        throw std::invalid_argument("unreserved dense quant-group views");
    static_assert(sizeof(Q81Block)==36 && sizeof(GroupInputMeta)==8);
    if(uintptr_t(weights)%4 || uintptr_t(input)%4 || uintptr_t(codes)%16 || uintptr_t(meta)%8 || uintptr_t(output)%4)
        throw std::invalid_argument("unaligned dense quant-group views");
    const size_t block_bytes=type==12 ? sizeof(Q4KBlock) : type==13 ? sizeof(Q5KBlock) : sizeof(Q6KBlock);
    const std::array<std::pair<uintptr_t,size_t>,5> spans{{
        {uintptr_t(weights),size_t(outputs)*(width/256)*block_bytes},
        {uintptr_t(input),size_t(rows)*(width/32)*sizeof(Q81Block)},
        {uintptr_t(codes),size_t(rows)*width*2},
        {uintptr_t(meta),size_t(rows)*(width/(type==14 ? 16 : 32))*sizeof(GroupInputMeta)},
        {uintptr_t(output),size_t(rows)*outputs*sizeof(float)}}};
    for(size_t i=0;i<spans.size();++i) {
        const auto [begin,bytes]=spans[i];
        if(begin>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("dense quant-group view overflow");
        for(size_t j=0;j<i;++j)
            if(begin<spans[j].first+spans[j].second && spans[j].first<begin+bytes)
                throw std::invalid_argument("aliased dense quant-group views");
    }
}
#include "native_blackwell_dense_wide_api.cuh"
#include "native_blackwell_dense_i128_api.cuh"
#include "native_blackwell_dense_swizzle_api.cuh"
void run_blackwell_dense_group_f16_tile(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,void* stream,int tile_rows) {
    validate_dense_group_views(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,tile_rows);
    if(type==12) launch_dense_variant<GroupQ4>(weights,input,codes,meta,output,width,rows,outputs,stream,tile_rows);
    else if(type==13) launch_dense_variant<GroupQ5>(weights,input,codes,meta,output,width,rows,outputs,stream,tile_rows);
    else launch_dense_variant<GroupQ6>(weights,input,codes,meta,output,width,rows,outputs,stream,tile_rows);
    CUDA_CHECK(cudaGetLastError());
}
bool blackwell_dense_group_int8_enabled(int type,int rows,int width,int outputs) {
    static const unsigned mask=blackwell_dense_int8_mask(std::getenv("STRATA_BENCH_DENSE_INT8"));
    return blackwell_dense_int8_selected(mask,type,rows,width,outputs);
}
void run_blackwell_dense_group_int8_tile(int type,const void* weights,const void* input,void* codes,
    size_t code_bytes,void* meta,size_t meta_bytes,float* output,int width,int rows,int outputs,void* stream,int tile_rows) {
    if((type!=12 && type!=13) || tile_rows!=128)
        throw std::invalid_argument("dense group INT8 requires Q4_K or Q5_K with tile128");
    static const auto policy=blackwell_dense_wide_policy(std::getenv("STRATA_BENCH_DENSE_INT8_OUTPUT_ROWS"));
    const int feature_rows=blackwell_dense_wide_rows(policy,type,rows,width,outputs);
    if(try_blackwell_dense_i128(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,feature_rows)) return;
    if(try_blackwell_dense_swizzle(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,feature_rows)) return;
    if(feature_rows!=16) {
        run_blackwell_dense_wide_int8_tile(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
            width,rows,outputs,stream,feature_rows);
        return;
    }
    validate_dense_group_views(type,weights,input,codes,code_bytes,meta,meta_bytes,output,
        width,rows,outputs,stream,tile_rows);
    if(type==12) launch_dense_group_int8<GroupQ4,128>(weights,input,codes,meta,output,width,rows,outputs,stream);
    else launch_dense_group_int8<GroupQ5,128>(weights,input,codes,meta,output,width,rows,outputs,stream);
    CUDA_CHECK(cudaGetLastError());
}
template<class F,int Tile> BlackwellDenseResources dense_int8_resources() {
    cudaFuncAttributes attr{}; int active=0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_group_int8<F,Tile>));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_group_int8<F,Tile>,256,0));
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,attr.maxThreadsPerBlock,active};
}
BlackwellDenseResources blackwell_dense_group_int8_resources(int type,int width,int tile_rows) {
    check_device();
    if((type!=12 && type!=13) || width<512 || width>4096 || width%512 ||
       tile_rows!=128) throw std::invalid_argument("invalid dense INT8 resource query");
    if(type==12) return dense_int8_resources<GroupQ4,128>();
    return dense_int8_resources<GroupQ5,128>();
}
template<class F,int Tile> BlackwellDenseResources dense_resources() {
    cudaFuncAttributes attr{}; int active=0;
    if constexpr(Tile==64) {
        CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_group_f16<F>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_group_f16<F>,256,0));
    } else {
        CUDA_CHECK(cudaFuncGetAttributes(&attr,dense_group_f16_128<F>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_group_f16_128<F>,256,0));
    }
    return {attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,attr.maxThreadsPerBlock,active};
}
BlackwellDenseResources blackwell_dense_resources(int type,int width,int tile_rows) {
    check_device();
    if((type!=12 && type!=13 && type!=14) || width<512 || width>4096 || width%512 ||
       (tile_rows!=64 && tile_rows!=128)) throw std::invalid_argument("invalid dense resource query");
    if(type==12) return tile_rows==64 ? dense_resources<GroupQ4,64>() : dense_resources<GroupQ4,128>();
    if(type==13) return tile_rows==64 ? dense_resources<GroupQ5,64>() : dense_resources<GroupQ5,128>();
    return tile_rows==64 ? dense_resources<GroupQ6,64>() : dense_resources<GroupQ6,128>();
}
int blackwell_mmq_backend(int type, int parts) {
    static const int selected = [] {
        const char* s = std::getenv("STRATA_NATIVE_MMQ_BACKEND");
        if (!s || !*s || !std::strcmp(s,"canonical")) return 0;
        if (!std::strcmp(s,"int8")) return 5;
        if (!std::strcmp(s,"group-fp8")) return 6;
        if (!std::strcmp(s,"group-fp4")) return 7;
        if (!std::strcmp(s,"group-f16")) return 9;
        if (!std::strcmp(s,"group-mixed")) return 10;
        if (!std::strcmp(s,"fp8")) return 1;
        if (!std::strcmp(s,"fp4")) return 2;
        if (!std::strcmp(s,"auto")) return 3;
        if (!std::strcmp(s,"fp4-down")) return 4;
        throw std::invalid_argument("STRATA_NATIVE_MMQ_BACKEND must be canonical, fp8, fp4, auto, fp4-down, int8, group-fp8 or group-fp4");
    }();
    // Q5 has no canonical/other-backend implementation in this provider.
    if (type==GGML_TYPE_Q5_K) return blackwell_q5_tc_enabled() && selected==6 && parts==1 ? 6 : 0;
    // Measured1024-row service curve: canonical GU/Q6 Down, FP8 Q4 Down.
    if (selected == 3) return parts == 1 && type == GGML_TYPE_Q4_K ? 1 : 0;
    if (selected == 4) return parts == 1 ? 2 : 0;
    if(selected==10) return parts==2 ? 0 : type==GGML_TYPE_Q4_K ? 6 : 9;
    return selected;
}
size_t blackwell_input_bytes(int type, int width, int rows) {
    if ((type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K) || width < 512 || width%512 ||
        rows < 1 || rows > 65536) throw std::invalid_argument("invalid bounded Blackwell input");
    const int groups = (width/256)*(type == GGML_TYPE_Q6_K ? 32 : 16);
    return size_t(rows)*width*2+size_t(rows)*groups*sizeof(BlackwellInputMeta);
}
NativeMmqEncodedInput repack_blackwell_input(const void* source, void* packed, int type,
    int width, int rows, const int32_t* ids, int backend, void* stream) {
    blackwell_input_bytes(type,width,rows);
    if (!source || !packed || uintptr_t(packed)%16 || !stream || (backend != 1 && backend != 2 && backend != 5 && backend != 6 && backend != 7 && backend != 9))
        throw std::invalid_argument("invalid Blackwell encoded input view");
    if (type == GGML_TYPE_Q4_K) {
        if (backend == 9) launch_group_input<GroupQ4,false,true>(source,packed,width,rows,stream);
        else if (backend == 6) launch_group_input<GroupQ4,false>(source,packed,width,rows,stream);
        else if (backend == 7) launch_group_input<GroupQ4,true>(source,packed,width,rows,stream);
        else if (backend == 5) launch_encode_input<BwQ4,false,true>(source,packed,width,rows,stream);
        else if (backend == 1) launch_encode_input<BwQ4,false>(source,packed,width,rows,stream);
        else launch_encode_input<BwQ4,true>(source,packed,width,rows,stream);
    } else if(type==GGML_TYPE_Q5_K) {
        if(backend!=6) throw std::invalid_argument("Q5 input requires the qualified FP8 group provider");
        launch_group_input<GroupQ5,false>(source,packed,width,rows,stream);
    } else {
        if (backend == 9) launch_group_input<GroupQ6,false,true>(source,packed,width,rows,stream);
        else if (backend == 6) launch_group_input<GroupQ6,false>(source,packed,width,rows,stream);
        else if (backend == 7) launch_group_input<GroupQ6,true>(source,packed,width,rows,stream);
        else if (backend == 5) launch_encode_input<BwQ6,false,true>(source,packed,width,rows,stream);
        else if (backend == 1) launch_encode_input<BwQ6,false>(source,packed,width,rows,stream);
        else launch_encode_input<BwQ6,true>(source,packed,width,rows,stream);
    }
    CUDA_CHECK(cudaGetLastError());
    return {packed,static_cast<const unsigned char*>(packed)+size_t(rows)*width*2,ids,rows};
}
BlackwellFusedInputResources blackwell_swiglu_input_resources(bool diagnostic) {
    check_device();
    const void* kernel=diagnostic ? reinterpret_cast<const void*>(encode_swiglu_group_input<true>) :
        reinterpret_cast<const void*>(encode_swiglu_group_input<false>);
    cudaFuncAttributes attr{}; int blocks=0;
    CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,256,0));
    return {attr.numRegs,256,blocks,attr.sharedSizeBytes,attr.localSizeBytes};
}
NativeMmqEncodedInput encode_blackwell_swiglu_input(const float* gate, const float* up,
    void* packed, size_t packed_bytes, int type, int width, int rows,
    const int32_t* ids, void* stream, void* diagnostic) {
    if ((type!=12 && type!=13) || width!=512 || rows<1 || rows>65536 ||
        !gate || !up || !packed || !stream || uintptr_t(gate)%4 || uintptr_t(up)%4 ||
        uintptr_t(packed)%16 || (diagnostic && uintptr_t(diagnostic)%4))
        throw std::invalid_argument("invalid bounded fused Down input");
    const size_t required=blackwell_input_bytes(type,width,rows);
    if (packed_bytes<required) throw std::invalid_argument("short fused Down input workspace");
    const size_t values=size_t(width)*rows;
    const std::array<std::pair<uintptr_t,size_t>,4> ranges{{
        {uintptr_t(gate),values*4},{uintptr_t(up),values*4},{uintptr_t(packed),required},
        {uintptr_t(diagnostic),diagnostic ? values/32*sizeof(Q81Block) : 0}}};
    for (size_t i=0;i<ranges.size();++i) if (ranges[i].second) {
        const auto [begin,bytes]=ranges[i];
        if (begin>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("fused Down input span overflow");
        for (size_t j=0;j<i;++j) if (ranges[j].second &&
            begin<ranges[j].first+ranges[j].second && ranges[j].first<begin+bytes)
            throw std::invalid_argument("fused Down input/output aliases");
    }
    auto* meta=reinterpret_cast<GroupInputMeta*>(static_cast<unsigned char*>(packed)+values*2);
    const unsigned blocks=unsigned((values+255)/256);
    if (diagnostic) encode_swiglu_group_input<true><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(
        gate,up,static_cast<unsigned*>(packed),meta,static_cast<Q81Block*>(diagnostic),width,rows);
    else encode_swiglu_group_input<false><<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(
        gate,up,static_cast<unsigned*>(packed),meta,nullptr,width,rows);
    CUDA_CHECK(cudaGetLastError());
    return {packed,meta,ids,rows};
}
void run_blackwell_grouped(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* ids, const void* input, float* output, int rows, int max_rows,
    int width, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count, int backend, const NativeMmqEncodedInput& encoded) {
    check_device();
    if(type==GGML_TYPE_Q5_K && (backend!=6 || parts!=1))
        throw std::invalid_argument("Q5 grouped product requires FP8 Down");
    if(encoded.tile_rows!=64 && (encoded.tile_rows!=128 || backend!=6 || (type!=12 && type!=13)))
        throw std::invalid_argument("group128 requires Q4/Q5 FP8");
    if (!encoded.codes || !encoded.metadata || !encoded.source_ids || encoded.source_rows < 1 ||
        !matrices || !ids || !output || !stream || groups < 1 || groups > 256 ||
        (parts != 1 && parts != 2) || rows < 1 || rows > 65536 || max_rows < 1 || max_rows > 8192 ||
        width < 512 || width > (type == GGML_TYPE_Q4_K ? 2048 : 1024) || width%512 || outputs < 8 || outputs%8)
        throw std::invalid_argument("invalid bounded Blackwell grouped product");
    if (backend != 1 && backend != 2 && backend != 5 && backend != 6 && backend != 7 && backend != 9) throw std::invalid_argument("invalid Blackwell backend");
    if (backend == 6 || backend == 7 || backend == 9) {
        if (outputs%16) throw std::invalid_argument("quant-group output rows require16 alignment");
#define GROUP_RUN(F,P,A) launch_group_quant<F,P,A>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded)
        if(type==GGML_TYPE_Q4_K) {
            if(backend==9) { if(tiles) launch_group_quant<GroupQ4,false,true,true>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded); else launch_group_quant<GroupQ4,false,false,true>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded); }
            else if(backend==6) { if(tiles) GROUP_RUN(GroupQ4,false,true); else GROUP_RUN(GroupQ4,false,false); }
            else { if(tiles) GROUP_RUN(GroupQ4,true,true); else GROUP_RUN(GroupQ4,true,false); }
        } else if(type==GGML_TYPE_Q5_K) {
            if(tiles) GROUP_RUN(GroupQ5,false,true); else GROUP_RUN(GroupQ5,false,false);
        } else {
            if(backend==9) { if(tiles) launch_group_quant<GroupQ6,false,true,true>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded); else launch_group_quant<GroupQ6,false,false,true>(matrices,groups,parts,ids,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded); }
            else if(backend==6) { if(tiles) GROUP_RUN(GroupQ6,false,true); else GROUP_RUN(GroupQ6,false,false); }
            else { if(tiles) GROUP_RUN(GroupQ6,true,true); else GROUP_RUN(GroupQ6,true,false); }
        }
#undef GROUP_RUN
        CUDA_CHECK(cudaGetLastError()); return;
    }
    if (type == GGML_TYPE_Q4_K) {
        if (backend == 5) launch_grouped<BwQ4,false,true>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        else if (backend == 1) launch_grouped<BwQ4,false>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        else launch_grouped<BwQ4,true>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
    } else if (type == GGML_TYPE_Q6_K) {
        if (backend == 5) launch_grouped<BwQ6,false,true>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        else if (backend == 1) launch_grouped<BwQ6,false>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
        else launch_grouped<BwQ6,true>(matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,encoded);
    } else throw std::invalid_argument("unsupported Blackwell quant format");
    CUDA_CHECK(cudaGetLastError());
}
template<class F, bool Active, bool Panel = false>
NativeBlackwellGroupResources group_resources(int width, int tile_rows) {
    NativeBlackwellGroupResources result;
    result.dynamic_shared_bytes = Panel && tile_rows==64 ? 0 : size_t(group_fp8_shared_bytes<F>(Panel ? 512 : width));
    const auto kernel = tile_rows == 128 ? blackwell_group128_mmq<F, Active, Panel>
        : Panel ? group_panel_mmq<F, false, Active, false, 64>
                : blackwell_group_mmq<F, false, Active, false>;
    if (result.dynamic_shared_bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(result.dynamic_shared_bytes)));
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    result.registers = attributes.numRegs;
    result.local_bytes = attributes.localSizeBytes;
    result.static_shared_bytes = attributes.sharedSizeBytes;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&result.active_blocks_per_sm,
        kernel, result.threads, result.dynamic_shared_bytes));
    return result;
}
NativeBlackwellGroupResources blackwell_group_resources(int type, int width, int tile_rows, bool active, int panel_tokens) {
    check_device();
    if ((type != 12 && type != 13) || width < 512 || width % 512 ||
        width > (type == 12 ? 2048 : 1024) || (tile_rows != 64 && tile_rows != 128) ||
        (panel_tokens!=0 && panel_tokens!=64))
        throw std::invalid_argument("invalid Q4/Q5 FP8 group resource query");
    if(type==12 && width==2048 && panel_tokens==64)
        return active ? group_resources<GroupQ4, true, true>(width, tile_rows)
                      : group_resources<GroupQ4, false, true>(width, tile_rows);
    if (type == 12) return active ? group_resources<GroupQ4, true>(width, tile_rows)
                                  : group_resources<GroupQ4, false>(width, tile_rows);
    return active ? group_resources<GroupQ5, true>(width, tile_rows)
                  : group_resources<GroupQ5, false>(width, tile_rows);
}
#include "native_blackwell_gu_int8_api.cuh"
#include "native_blackwell_gu_int8_i32_api.cuh"
#include "native_blackwell_gu_int8_i64_api.cuh"
#include "native_blackwell_down_int8_api.cuh"
} // namespace strata::prefill::mmq
