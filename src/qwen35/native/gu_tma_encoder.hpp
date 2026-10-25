#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include <cuda.h>
#include <cudaTypedefs.h>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native::tma_detail {
inline PFN_cuTensorMapEncodeTiled_v12000 encoder(){
    static const auto function=[](){
        void *address=nullptr;cudaDriverEntryPointQueryResult status{};
        check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",&address,12000,
                   cudaEnableDefault,&status),"TensorMap encoder entry");
        if(status!=cudaDriverEntryPointSuccess||!address)throw std::runtime_error("TensorMap encoder unavailable");
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(address);
    }();return function;
}
}
