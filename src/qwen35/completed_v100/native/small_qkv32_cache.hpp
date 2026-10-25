#pragma once
#include "small_prefill_half_codes.hpp"
#include "small_prefill_q6_planes.hpp"
#include <chrono>
#include <string>
#include <thread>

namespace strata::completed_v100::qwen35::native {
// Optional single-layer cache, borrowed from the same raw PP/TG inventory.
// Setup/retirement must occur outside execution; one thread and stream own it.
class SmallQkv32 {
public:
    SmallQkv32(Matrix raw,cudaStream_t stream,bool complete,bool register_reuse=false,bool prefetch=false,bool planes=false,bool byte_planes=false):raw_(raw),stream_(stream),owner_(std::this_thread::get_id()),complete_(complete),register_reuse_(register_reuse),prefetch_(prefetch),planes_(planes){
        const auto cold=std::chrono::steady_clock::now();
        check_cuda(cudaGetDevice(&device_),"QKV32 cache device");
        const auto caps=capabilities(device_);
        if(caps.major!=7||caps.minor!=0)throw std::invalid_argument("QKV32 half cache requires validated SM70 entry");
        processors_=caps.multiprocessors;
        if(register_reuse&&(!complete||raw.quant!=Quant::Q6K||raw.layout!=BlockLayout::Q6Codes))
            throw std::invalid_argument("register QKV32 requires complete byte-code inventory");
        if(planes&&(!complete||register_reuse||prefetch||raw.quant!=Quant::Q6K||raw.layout!=BlockLayout::Gguf))
            throw std::invalid_argument("Q6 planes require complete raw QKV inventory");
        if(byte_planes&&!planes)throw std::invalid_argument("byte Q6 view requires plane admission");
        const size_t weight_bytes=planes?(byte_planes?q6_byte_planes_bytes(raw):matrix_bytes(raw)):register_reuse?0:prepared_q6_half_codes_bytes(raw);
        // Validate the complete32-token geometry before allocating anything.
        if(raw.rows%64||raw.columns/256<3)throw std::invalid_argument("QKV32 cache requires complete three-way tiles");
        const size_t input_bytes=small_prefill_half_input_bytes(raw.columns,32);
        partial_bytes_=complete?0:size_t(3)*32*raw.rows*4;
        bytes_=weight_bytes+input_bytes+partial_bytes_;
        configure_small_prefill_direct();
        const cudaError_t allocation=cudaMalloc(&arena_,bytes_);
        if(allocation!=cudaSuccess)
            throw std::runtime_error("QKV32 allocation: request_bytes="+std::to_string(bytes_)+
                " owner_live_bytes=0 cuda_error="+cudaGetErrorString(allocation)+
                " cuda_code="+std::to_string(static_cast<int>(allocation)));
        try{
            input_=static_cast<unsigned char *>(arena_)+weight_bytes;
            partials_=static_cast<unsigned char *>(input_)+input_bytes;
            if(planes)plane_weight_=byte_planes?prepare_q6_byte_planes(raw,arena_,stream):prepare_q6_planes(raw,arena_,stream);
            else if(!register_reuse)weight_=prepare_q6_half_codes(raw,arena_,stream);
            if(!complete&&small_prefill_half_codes_split_bytes(weight_,32)!=partial_bytes_)
                throw std::runtime_error("QKV32 cache partial extent mismatch");
            check_cuda(cudaStreamSynchronize(stream),"QKV32 cold preparation completion");
            cold_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-cold).count();
        }catch(...){cudaFree(arena_);arena_=nullptr;throw;}
    }
    ~SmallQkv32(){if(arena_)cudaFree(arena_);}
    SmallQkv32(const SmallQkv32 &)=delete;
    SmallQkv32 &operator=(const SmallQkv32 &)=delete;
    size_t bytes()const{return bytes_;}
    double cold_ms()const{return cold_ms_;}
    unsigned extra_submissions()const{return complete_?1:2;}
    bool matches(Matrix raw)const{
        return raw.data==raw_.data&&raw.rows==raw_.rows&&raw.columns==raw_.columns&&
               raw.quant==raw_.quant&&raw.layout==raw_.layout;
    }
    void *encoding_input(cudaStream_t stream){
        int current=0;check_cuda(cudaGetDevice(&current),"QKV32 encoding lease device");
        if(stream!=stream_||current!=device_||std::this_thread::get_id()!=owner_)
            throw std::invalid_argument("QKV32 encoding lease changed");
        return input_;
    }
    void execute(Matrix raw,const float *normalized,float *output,cudaStream_t stream,bool prepared=false){
        int current=0;check_cuda(cudaGetDevice(&current),"QKV32 execution device");
        if(!matches(raw)||stream!=stream_||current!=device_||std::this_thread::get_id()!=owner_)
            throw std::invalid_argument("QKV32 cache inventory/thread/stream/device lease changed");
        if(!prepared)small_prefill_half_quantize(normalized,input_,raw.columns,32,stream);
        if(planes_)small_prefill_q6_planes_linear(plane_weight_,input_,output,stream);
        else if(register_reuse_)small_prefill_q6_register_linear(raw_,input_,output,stream,prefetch_);
        else if(complete_)small_prefill_half_codes_complete_linear(weight_,input_,output,32,stream,processors_);
        else small_prefill_half_codes_compact_linear(weight_,input_,output,32,partials_,partial_bytes_,stream,processors_);
    }
private:
    Matrix raw_;cudaStream_t stream_;std::thread::id owner_;
    bool complete_=false,register_reuse_=false,prefetch_=false,planes_=false;
    int device_=0,processors_=0;
    void *arena_=nullptr,*input_=nullptr,*partials_=nullptr;
    HalfCodeMatrix weight_;
    Q6PlaneMatrix plane_weight_;
    size_t bytes_=0,partial_bytes_=0;
    double cold_ms_=0;
};
}
