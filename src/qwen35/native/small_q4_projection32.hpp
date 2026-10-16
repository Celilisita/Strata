#pragma once
#include "small_prefill_q4_grouped.hpp"
#include "small_prefill_half_raw.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <algorithm>
#include <array>
#include <thread>

namespace strata::qwen35::native {
// Experimental PP32 stage borrows four raw matrices. No weight copy/cache.
// One prepared input and ordered-partial arena is reused on its setup stream.
class SmallQ4Projection32 {
public:
    SmallQ4Projection32(std::array<Matrix,4> raw,cudaStream_t stream)
        :raw_(raw),stream_(stream),owner_(std::this_thread::get_id()){
        check_cuda(cudaGetDevice(&device_),"Q4 projection stage device");
        const auto caps=capabilities(device_);
        if(caps.major!=7||caps.minor!=0)throw std::invalid_argument("Q4 stage requires validated SM70 service");
        size_t input_bytes=0;
        for(auto w:raw_){
            matrix_bytes(w);
            if(w.quant!=Quant::Q4K||w.layout!=BlockLayout::Q4Pairs||w.columns/256<5||w.rows%4)
                throw std::invalid_argument("Q4 stage requires raw aligned split projection geometry");
            input_bytes=std::max(input_bytes,small_prefill_half_input_bytes(w.columns,32));
            partial_bytes_=std::max(partial_bytes_,size_t(q4_single_partitions(w))*32*w.rows*4);
        }
        if(raw_[0].columns!=raw_[1].columns||raw_[0].columns!=raw_[2].columns||raw_[0].rows%32)
            throw std::invalid_argument("Q4 norm projections must share input geometry");
        if(raw_[1].rows==raw_[2].rows&&raw_[1].rows<64&&raw_[1].columns/256<=32)
            partial_bytes_=std::max(partial_bytes_,size_t(2)*raw_[1].columns/256*32*raw_[1].rows*4);
        configure_small_prefill_q4_pair();bytes_=input_bytes+partial_bytes_;
        check_cuda(cudaMalloc(&arena_,bytes_),"Q4 projection stage arena");
        input_=arena_;partials_=static_cast<unsigned char *>(arena_)+input_bytes;
    }
    ~SmallQ4Projection32(){if(arena_)cudaFree(arena_);}
    SmallQ4Projection32(const SmallQ4Projection32 &)=delete;
    SmallQ4Projection32 &operator=(const SmallQ4Projection32 &)=delete;
    size_t bytes()const{return bytes_;}
    void validate_lease(cudaStream_t stream) const {check_lease(stream);}
    bool matches(const std::array<Matrix,4> &raw)const{
        for(int i=0;i<4;++i)if(!same(raw[i],raw_[i]))return false;
        return true;
    }
    void rebind(std::array<Matrix,4> next,cudaStream_t stream){
        check_lease(stream);
        for(int i=0;i<4;++i){matrix_bytes(next[i]);
            if(next[i].rows!=raw_[i].rows||next[i].columns!=raw_[i].columns||next[i].quant!=raw_[i].quant||next[i].layout!=raw_[i].layout)
                throw std::invalid_argument("Q4 shared projection geometry or format changed");}
        raw_=next;prepared_columns_=0;
    }
    const void *prepared_input(int columns,cudaStream_t stream){
        check_lease(stream);
        if(prepared_columns_!=columns)throw std::invalid_argument("Q4 borrowed prepared input changed");
        return input_;
    }
    void prepare(const float *values,int columns,cudaStream_t stream){
        check_lease(stream);
        if(!values||(columns!=raw_[0].columns&&columns!=raw_[3].columns))
            throw std::invalid_argument("Q4 projection input geometry changed");
        small_prefill_half_raw_quantize(values,input_,Quant::Q4K,columns,32,stream);
        prepared_columns_=columns;
    }
    void prepare_rms(const float *x,const float *w,float *normalized,const float *gate,
                     int features,int heads,float eps,void *q6,cudaStream_t stream){
        check_lease(stream);const int columns=features*heads;
        if(columns!=raw_[0].columns&&columns!=raw_[3].columns)
            throw std::invalid_argument("fused Q4 input geometry changed");
        rms_norm_quantized32(x,w,normalized,gate,features,heads,eps,input_,q6,stream);
        prepared_columns_=columns;
    }
    void linear(Matrix raw,float *output,cudaStream_t stream){
        check_lease(stream);bool found=false;
        for(auto w:raw_)found=found||same(raw,w);
        if(!found||prepared_columns_!=raw.columns)
            throw std::invalid_argument("Q4 projection inventory or prepared input lease changed");
        small_prefill_q4_single_linear(raw,input_,output,32,stream,partials_,partial_bytes_);
    }
    void small_pair(float *a,float *b,cudaStream_t stream){
        check_lease(stream);
        if(prepared_columns_!=raw_[1].columns)throw std::invalid_argument("Q4 pair prepared input changed");
        small_prefill_q4_small_pair_linear(raw_[1],raw_[2],input_,a,b,stream,partials_,partial_bytes_);
    }
private:
    static bool same(Matrix a,Matrix b){
        return a.data==b.data&&a.rows==b.rows&&a.columns==b.columns&&a.quant==b.quant&&a.layout==b.layout;
    }
    void check_lease(cudaStream_t stream)const{
        int current=0;check_cuda(cudaGetDevice(&current),"Q4 projection execution device");
        if(current!=device_||stream!=stream_||std::this_thread::get_id()!=owner_)
            throw std::invalid_argument("Q4 projection stream/thread/device lease changed");
    }
    std::array<Matrix,4> raw_;cudaStream_t stream_;std::thread::id owner_;
    int device_=0,prepared_columns_=0;
    void *arena_=nullptr,*input_=nullptr,*partials_=nullptr;
    size_t bytes_=0,partial_bytes_=0;
};
}
