#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include "small_prefill_mma.cuh"
#include "small_prefill_half_codes.hpp"
#include "small_prefill_q4_half_codes.hpp"
#include "small_prefill_q4_scaled_codes.hpp"
#include "small_prefill_direct_limits.cuh"
#include "small_prefill_q6_planes.hpp"

namespace strata::completed_v100::qwen35::native {
namespace {
template<bool ByteCodes>
__global__ void prepack_q6_planes(const unsigned char *source,unsigned char *output,size_t count,int blocks){
    const size_t index=size_t(blockIdx.x)*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    if(index>=count)return;
    const size_t row=index/blocks,k=index%blocks,tile=(row/64)*blocks+k;
    const auto *input=reinterpret_cast<const unsigned short *>(source+index*210);
    constexpr int CodeBytes=ByteCodes?256:192,TileBytes=64*(CodeBytes+18);
    auto *codes=output+tile*TileBytes+(row%64)*CodeBytes;
    if constexpr(ByteCodes){
        for(int word=lane;word<64;word+=32)
            reinterpret_cast<unsigned *>(codes)[word]=unsigned(detail::weight_codes<Quant::Q6K>(source+index*210,word/8,(word%8)*4));
    }else for(int word=lane;word<96;word+=32)reinterpret_cast<unsigned short *>(codes)[word]=input[word];
    if(lane<8){
        auto *meta=reinterpret_cast<unsigned short *>(output+tile*TileBytes+64*CodeBytes);
        meta[lane*64+row%64]=input[96+lane];
    }
    if(!lane)*reinterpret_cast<unsigned short *>(output+tile*TileBytes+64*CodeBytes+1024+(row%64)*2)=input[104];
}
// A reversible permutation of the 96 metadata bits. Scale/min remain six
// bits each; d/dmin and every quantized nibble keep their original bits.
__device__ unsigned descriptor(const unsigned char *p, int group) {
    const int part = group & 3;
    const unsigned scale = group < 4 ? p[4 + part] & 63 :
        (p[12 + part] & 15) | ((p[4 + part] >> 6) << 4);
    const unsigned minimum = group < 4 ? p[8 + part] & 63 :
        (p[12 + part] >> 4) | ((p[8 + part] >> 6) << 4);
    return scale | (minimum << 6);
}
__global__ void prepack_q4_pairs(const unsigned char *source, unsigned char *output, size_t count) {
    const size_t block = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (block >= count) return;
    const auto *input = source + block * 144;
    auto *result = output + block * 144;
    if (lane < 9)
        reinterpret_cast<uint4 *>(result)[lane] = reinterpret_cast<const uint4 *>(input)[lane];
    __syncwarp();
    if (lane < 4) {
        const unsigned metadata = descriptor(input, lane * 2) | (descriptor(input, lane * 2 + 1) << 12);
        result[4 + lane * 3] = static_cast<unsigned char>(metadata);
        result[5 + lane * 3] = static_cast<unsigned char>(metadata >> 8);
        result[6 + lane * 3] = static_cast<unsigned char>(metadata >> 16);
    }
}
__global__ void prepack_q6_codes(const unsigned char *source, unsigned char *output, size_t count) {
    const size_t block = size_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (block >= count) return;
    const auto *input = source + block * 210;
    auto *codes = output + block * Q6CodesBytes;
    for (int word = lane; word < 64; word += 32)
        reinterpret_cast<unsigned *>(codes)[word] = unsigned(detail::weight_codes<Quant::Q6K>(
            input, word / 8, (word % 8) * 4));
    // Metadata is a separate plane: original sixteen scale bytes then delta bits.
    if (lane < Q6CodesMetadataBytes / 2) {
        auto *metadata = output + count * Q6CodesBytes + block * Q6CodesMetadataBytes;
        reinterpret_cast<unsigned short *>(metadata)[lane] =
            reinterpret_cast<const unsigned short *>(input + 192)[lane];
    }
}
// Codes are chunk-major within64-row K256 tiles. Metadata preserves original
// scale/delta bits and initializes its fourteen padding bytes explicitly.
__global__ void prepack_q6_half_codes(const unsigned char *source,unsigned char *output,
                                     size_t count,int blocks) {
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;
    const int lane=threadIdx.x%32;
    if(block>=count)return;
    const size_t row=block/blocks,k=block%blocks,tile=(row/64)*blocks+k;
    const auto *input=source+block*210;
    auto *codes=reinterpret_cast<unsigned *>(output)+tile*64*128;
    for(int word=lane;word<64;word+=32){
        const unsigned packed=unsigned(detail::weight_codes<Quant::Q6K>(input,word/8,(word%8)*4));
        const uint2 value=make_uint2(detail::half_codes(packed,true,0),detail::half_codes(packed,true,2));
        const size_t index=(word/32)*64*64+(row%64)*64+(word%32)*2;
        *reinterpret_cast<uint2 *>(codes+index)=value;
    }
    if(lane<16){
        auto *metadata=reinterpret_cast<unsigned short *>(output+count*512)+tile*64*16+(row%64)*16;
        metadata[lane]=lane<9?reinterpret_cast<const unsigned short *>(input+192)[lane]:0;
    }
}
__device__ unsigned scale_half_codes(unsigned bits,half2 scale){
    const half2 v=__hmul2(__halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),__ushort_as_half(static_cast<unsigned short>(bits>>16))),scale);
    return unsigned(__half_as_ushort(__low2half(v)))|(unsigned(__half_as_ushort(__high2half(v)))<<16);
}
template<bool Scaled>
__global__ void prepack_q4_half_codes(const unsigned char *source,unsigned char *output,size_t count,int blocks){
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;const int lane=threadIdx.x%32;
    if(block>=count)return;
    const size_t row=block/blocks,k=block%blocks,tile=(row/64)*blocks+k;
    const auto *input=source+block*144;auto *codes=reinterpret_cast<unsigned *>(output)+tile*64*128;
    const uint4 raw=*reinterpret_cast<const uint4 *>(input);
    for(int word=lane;word<64;word+=32){
        const int group=word/8,position=(word%8)*4;
        const unsigned packed=(*reinterpret_cast<const unsigned *>(input+16+(group/2)*32+position)>>((group%2)*4))&0x0f0f0f0fu;
        const size_t index=(word/32)*64*64+(row%64)*64+(word%32)*2;
        uint2 value=make_uint2(detail::half_codes(packed,false,0),detail::half_codes(packed,false,2));
        if constexpr(Scaled){
            const int w=(group*12)>>5,shift=(group*12)&31;
            const unsigned low=w==0?raw.y:w==1?raw.z:raw.w,high=w==0?raw.z:w==1?raw.w:0;
            const half scale=__float2half_rn(float(__funnelshift_r(low,high,shift)&63u));
            const half2 multiplier=__halves2half2(scale,scale);
            value=make_uint2(scale_half_codes(value.x,multiplier),scale_half_codes(value.y,multiplier));
        }
        *reinterpret_cast<uint2 *>(codes+index)=value;
    }
    if(lane<8){
        const int word=(lane*12)>>5,shift=(lane*12)&31;
        const unsigned low=word==0?raw.y:word==1?raw.z:raw.w,high=word==0?raw.z:word==1?raw.w:0;
        const unsigned meta=__funnelshift_r(low,high,shift)&4095u;
        const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x)));
        const float minimum=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x>>16)));
        auto *affine=reinterpret_cast<float2 *>(output+count*512)+tile*64*8;
        if constexpr(Scaled)affine[lane*64+row%64]=make_float2(float(meta>>6),lane==0?d:lane==1?minimum:0);
        else affine[lane*64+row%64]=make_float2(__fmul_rn(d,float(meta&63)),__fmul_rn(minimum,float(meta>>6)));
    }
}
__global__ void prepack_q4_scaled_codes(const unsigned char *source,unsigned char *output,size_t count){
    const size_t block=size_t(blockIdx.x)*4+threadIdx.x/32;const int lane=threadIdx.x%32;if(block>=count)return;
    const auto *input=source+block*144;auto *out=output+block*176;
    if(lane<8)reinterpret_cast<uint4 *>(out+48)[lane]=reinterpret_cast<const uint4 *>(input+16)[lane];
    if(lane<3)reinterpret_cast<unsigned *>(out+36)[lane]=0;
    const uint4 raw=*reinterpret_cast<const uint4 *>(input);
    if(!lane)*reinterpret_cast<unsigned *>(out)=raw.x;
    if(lane<8){
        const int word=(lane*12)>>5,shift=(lane*12)&31;
        const unsigned low=word==0?raw.y:word==1?raw.z:raw.w,high=word==0?raw.z:word==1?raw.w:0;
        const unsigned meta=__funnelshift_r(low,high,shift)&4095u;
        reinterpret_cast<unsigned *>(out+4)[lane]=unsigned(__half_as_ushort(__float2half_rn(float(meta&63))))|
            (unsigned(__half_as_ushort(__float2half_rn(float(meta>>6))))<<16);
    }
}
}
template<bool ByteCodes>
Q6PlaneMatrix prepare_q6_planes_impl(Matrix input,void *output,cudaStream_t stream){
    const size_t raw_bytes=matrix_bytes(input),bytes=ByteCodes?q6_byte_planes_bytes(input):raw_bytes;
    if(input.quant!=Quant::Q6K||input.layout!=BlockLayout::Gguf||input.rows%64||!output||
       reinterpret_cast<uintptr_t>(input.data)%2||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(input.data,raw_bytes,output,bytes))
        throw std::invalid_argument("Q6 planes require complete raw tiles and distinct aligned storage");
    const size_t count=size_t(input.rows)*(input.columns/256),grid=(count+3)/4;
    if(grid>0x7fffffffu)throw std::invalid_argument("Q6 planes grid extent exceeded");
    prepack_q6_planes<ByteCodes><<<unsigned(grid),128,0,stream>>>(static_cast<const unsigned char *>(input.data),
        static_cast<unsigned char *>(output),count,input.columns/256);
    check_cuda(cudaGetLastError(),"owned lossless Q6 plane preparation");
    return {output,input.rows,input.columns,bytes};
}
Q6PlaneMatrix prepare_q6_planes(Matrix input,void *output,cudaStream_t stream){return prepare_q6_planes_impl<false>(input,output,stream);}
Q6PlaneMatrix prepare_q6_byte_planes(Matrix input,void *output,cudaStream_t stream){return prepare_q6_planes_impl<true>(input,output,stream);}
Matrix prepare_q4_pairs(Matrix input, void *output, cudaStream_t stream) {
    matrix_bytes(input);
    if (!output || output == input.data || input.quant != Quant::Q4K || input.layout != BlockLayout::Gguf)
        throw std::invalid_argument("prepare requires distinct raw and packed Q4 storage");
    const size_t count = size_t(input.rows) * input.columns / 256;
    prepack_q4_pairs<<<unsigned((count + 3) / 4), 128, 0, stream>>>(
        static_cast<const unsigned char *>(input.data), static_cast<unsigned char *>(output), count);
    check_cuda(cudaGetLastError(), "native lossless Q4 metadata preparation");
    return {output, input.quant, input.rows, input.columns, BlockLayout::Q4Pairs};
}
size_t prepared_q6_codes_bytes(Matrix input) {
    matrix_bytes(input);
    if (input.quant != Quant::Q6K || input.layout != BlockLayout::Gguf)
        throw std::invalid_argument("Q6 code preparation requires a raw GGUF Q6 matrix");
    return size_t(input.rows) * size_t(input.columns / 256) * Q6CodesBlockBytes;
}
Matrix prepare_q6_codes(Matrix input, void *output, cudaStream_t stream) {
    const size_t raw_bytes = matrix_bytes(input), prepared_bytes = prepared_q6_codes_bytes(input);
    const uintptr_t source = reinterpret_cast<uintptr_t>(input.data);
    const uintptr_t destination = reinterpret_cast<uintptr_t>(output);
    if (!output || source % 2 || destination % 16 ||
        (source <= destination ? destination - source < raw_bytes : source - destination < prepared_bytes))
        throw std::invalid_argument("Q6 code preparation requires aligned distinct nonoverlapping buffers");
    const size_t count = size_t(input.rows) * size_t(input.columns / 256), grid = (count + 3) / 4;
    if (grid > 0x7fffffffu)
        throw std::invalid_argument("Q6 code preparation exceeds CUDA grid extent");
    prepack_q6_codes<<<unsigned(grid), 128, 0, stream>>>(
        static_cast<const unsigned char *>(input.data), static_cast<unsigned char *>(output), count);
    check_cuda(cudaGetLastError(), "native lossless Q6 code preparation");
    return {output, input.quant, input.rows, input.columns, BlockLayout::Q6Codes};
}
size_t prepared_q6_half_codes_bytes(Matrix input){
    matrix_bytes(input);
    if(input.quant!=Quant::Q6K||input.layout!=BlockLayout::Gguf||input.rows%64)
        throw std::invalid_argument("half-code preparation requires complete64-row raw Q6 tiles");
    return size_t(input.rows)*size_t(input.columns/256)*HalfCodeBlockBytes;
}
HalfCodeMatrix prepare_q6_half_codes(Matrix input,void *output,cudaStream_t stream){
    const size_t raw_bytes=matrix_bytes(input),prepared_bytes=prepared_q6_half_codes_bytes(input);
    const uintptr_t source=reinterpret_cast<uintptr_t>(input.data),destination=reinterpret_cast<uintptr_t>(output);
    if(!output||source%2||destination%16||
       (source<=destination?destination-source<raw_bytes:source-destination<prepared_bytes))
        throw std::invalid_argument("half-code preparation requires aligned nonoverlapping buffers");
    const size_t count=size_t(input.rows)*size_t(input.columns/256),grid=(count+3)/4;
    if(grid>0x7fffffffu)throw std::invalid_argument("half-code preparation CUDA grid extent");
    prepack_q6_half_codes<<<unsigned(grid),128,0,stream>>>(static_cast<const unsigned char *>(input.data),
        static_cast<unsigned char *>(output),count,input.columns/256);
    check_cuda(cudaGetLastError(),"native lossless tiled half-code preparation");
    return {output,input.rows,input.columns,prepared_bytes};
}
size_t half_code_workspace_bytes(HalfCodeMatrix weight){
    half_code_matrix_bytes(weight);
    return size_t(3)*32*weight.rows*4+size_t(weight.rows/64)*sizeof(unsigned);
}
size_t prepared_q4_half_codes_bytes(Matrix input){
    matrix_bytes(input);
    if(input.quant!=Quant::Q4K||input.layout!=BlockLayout::Q4Pairs||input.rows%64)
        throw std::invalid_argument("prepared Q4 half codes require complete64-row Q4Pairs tiles");
    return size_t(input.rows)*size_t(input.columns/256)*Q4HalfCodeBlockBytes;
}
template<bool Scaled>
Q4HalfCodeMatrix prepare_q4_half_codes_impl(Matrix input,void *output,cudaStream_t stream){
    const size_t raw_bytes=matrix_bytes(input),bytes=prepared_q4_half_codes_bytes(input);
    if(!output||reinterpret_cast<uintptr_t>(input.data)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(input.data,raw_bytes,output,bytes))
        throw std::invalid_argument("prepared Q4 half codes require aligned nonoverlapping storage");
    const size_t count=size_t(input.rows)*size_t(input.columns/256),grid=(count+3)/4;
    if(grid>0x7fffffffu)throw std::invalid_argument("prepared Q4 half-code grid extent");
    prepack_q4_half_codes<Scaled><<<unsigned(grid),128,0,stream>>>(static_cast<const unsigned char *>(input.data),static_cast<unsigned char *>(output),count,input.columns/256);
    check_cuda(cudaGetLastError(),"owned Q4 half-code cold preparation");return {output,input.rows,input.columns,bytes};
}
Q4HalfCodeMatrix prepare_q4_half_codes(Matrix input,void *output,cudaStream_t stream){return prepare_q4_half_codes_impl<false>(input,output,stream);}
Q4ScaledHalfCodeMatrix prepare_q4_scaled_half_codes(Matrix input,void *output,cudaStream_t stream){return {prepare_q4_half_codes_impl<true>(input,output,stream)};}
size_t prepared_q4_scaled_codes_bytes(Matrix input){
    matrix_bytes(input);if(input.quant!=Quant::Q4K||input.layout!=BlockLayout::Q4Pairs||input.rows%4)
        throw std::invalid_argument("compact scaled Q4 requires four-row Q4Pairs matrix");
    return size_t(input.rows)*size_t(input.columns/256)*176;
}
Q4ScaledCodeMatrix prepare_q4_scaled_codes(Matrix input,void *output,cudaStream_t stream){
    const size_t raw_bytes=matrix_bytes(input),bytes=prepared_q4_scaled_codes_bytes(input);
    if(!output||reinterpret_cast<uintptr_t>(input.data)%16||reinterpret_cast<uintptr_t>(output)%16||detail::split_overlap(input.data,raw_bytes,output,bytes))
        throw std::invalid_argument("compact scaled Q4 preparation requires aligned nonoverlapping storage");
    const size_t count=size_t(input.rows)*size_t(input.columns/256),grid=(count+3)/4;if(grid>0x7fffffffu)throw std::invalid_argument("compact Q4 preparation grid extent");
    prepack_q4_scaled_codes<<<unsigned(grid),128,0,stream>>>(static_cast<const unsigned char *>(input.data),static_cast<unsigned char *>(output),count);
    check_cuda(cudaGetLastError(),"compact scaled Q4 cold preparation");return {output,input.rows,input.columns,bytes};
}
HalfCodeWorkspace initialize_half_code_workspace(HalfCodeMatrix weight,void *scratch,
                                                size_t bytes,cudaStream_t stream){
    const size_t required=half_code_workspace_bytes(weight);
    const auto a=reinterpret_cast<uintptr_t>(weight.data),b=reinterpret_cast<uintptr_t>(scratch);
    if(!scratch||b%16||bytes<required||
       (a<=b?b-a<weight.bytes:a-b<required))
        throw std::invalid_argument("fused scratch requires aligned distinct declared extent");
    const size_t partial_bytes=size_t(3)*32*weight.rows*4;
    check_cuda(cudaMemsetAsync(static_cast<unsigned char *>(scratch)+partial_bytes,0,
                              size_t(weight.rows/64)*sizeof(unsigned),stream),"fused setup tickets");
    return {scratch,bytes,weight.rows,stream,true};
}
}
