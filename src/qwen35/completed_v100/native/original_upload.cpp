#include "strata/qwen35/completed_v100/native/original_upload.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include <cmath>
#include <cstring>
namespace strata::completed_v100::qwen35::native {
Matrix upload_original_nvfp4(const strata::GgufFile& file,const strata::TensorInfo& tensor,
    DeviceStorage& owner,cudaStream_t stream) {
    if(tensor.type!=40||tensor.shape.size()!=2||tensor.shape[0]%64||tensor.shape[0]>65536||tensor.shape[1]>1048576)
        throw std::invalid_argument("Unadmitted original NVFP4 extent");
    const std::string suffix=".weight";
    if(tensor.name.size()<suffix.size()||tensor.name.substr(tensor.name.size()-suffix.size())!=suffix)
        throw std::invalid_argument("Original NVFP4 matrix has no weight role");
    const auto role=tensor.name.substr(0,tensor.name.size()-suffix.size());
    const auto* scale=file.find(role+".scale");
    if(!scale||scale->type!=0||scale->shape!=std::vector<uint64_t>{1}||file.find(role+".input_scale"))
        throw std::invalid_argument("Original NVFP4 scale/input-scale boundary is unadmitted: "+role);
    (void)::strata::qwen35::tensor_bytes(file, *scale);
    float value;std::memcpy(&value,file.tensor_data(*scale),4);
    if(!(value>0)||!std::isfinite(value))throw std::invalid_argument("Invalid original NVFP4 global scale");
    const size_t raw=size_t(tensor.elements()/64)*36;
    if(file.data_start()>file.file_size()||tensor.offset>file.file_size()-file.data_start()||
       raw>file.file_size()-file.data_start()-tensor.offset)
        throw std::invalid_argument("Original NVFP4 payload exceeds file extent");
    auto* memory=static_cast<unsigned char*>(owner.allocate(OriginalNvfp4HeaderBytes+raw));
    check_cuda(cudaMemcpyAsync(memory,file.tensor_data(*scale),4,cudaMemcpyHostToDevice,stream),"Original NVFP4 scale upload");
    initialize_nvfp4_header(memory,stream);
    check_cuda(cudaMemcpyAsync(memory+OriginalNvfp4HeaderBytes,file.tensor_data(tensor),raw,cudaMemcpyHostToDevice,stream),"Original NVFP4 block upload");
    return {memory,Quant::NVFP4,int(tensor.shape[1]),int(tensor.shape[0]),BlockLayout::Nvfp4Scale};
}
}
