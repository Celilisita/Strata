#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "fp4_compensated_service.hpp"
#include <cmath>
#include <cstring>

namespace strata::qwen35::native {
Matrix load_nvfp4_matrix(const strata::GgufFile &file,const strata::TensorInfo &tensor,
                        DeviceStorage &owner,Nvfp4View &view,cudaStream_t stream){
    int device;check_cuda(cudaGetDevice(&device),"NVFP4 inventory device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0||tensor.type!=40||tensor.shape.size()!=2||tensor.shape[0]%256)
        throw std::invalid_argument("NVFP4 inventory requires admitted SM120 block matrix");
    const int rows=int(tensor.shape[1]),columns=int(tensor.shape[0]);
    const auto *scale=file.find(tensor.name.substr(0,tensor.name.size()-7)+".scale");
    if(!scale||scale->type!=0||scale->elements()!=1)throw std::invalid_argument("NVFP4 tensor requires FP32 global scale");
    float global;std::memcpy(&global,file.tensor_data(*scale),4);
    if(!(global>0)||!std::isfinite(global))throw std::invalid_argument("invalid NVFP4 global scale");
    const size_t raw_bytes=strata::qwen35::tensor_bytes(file,tensor),values=nvfp4_value_bytes(rows,columns);
    if(raw_bytes!=nvfp4_gguf_bytes(rows,columns))throw std::invalid_argument("NVFP4 GGUF directory extent");
    auto *data=static_cast<unsigned char *>(owner.allocate(values+nvfp4_scale_bytes(rows,columns)));
    auto *raw=owner.allocate(raw_bytes);auto *invalid=static_cast<unsigned *>(owner.allocate(4));
    try{
        check_cuda(cudaMemsetAsync(invalid,0,4,stream),"NVFP4 inventory invalid reset");
        check_cuda(cudaMemsetAsync(data+values,0,nvfp4_scale_bytes(rows,columns),stream),"NVFP4 inventory padded scales");
        check_cuda(cudaMemcpyAsync(raw,file.tensor_data(tensor),raw_bytes,cudaMemcpyHostToDevice,stream),"NVFP4 inventory upload");
        nvfp4_repack(raw,data,data+values,rows,columns,invalid,stream);
        unsigned bad;check_cuda(cudaMemcpyAsync(&bad,invalid,4,cudaMemcpyDeviceToHost,stream),"NVFP4 payload validation");
        check_cuda(cudaStreamSynchronize(stream),"NVFP4 repack source lease");
        if(bad)throw std::runtime_error("NVFP4 inventory contains invalid E4M3 block scales");
    }catch(...){cudaStreamSynchronize(stream);owner.release(raw);owner.release(invalid);owner.release(data);throw;}
    owner.release(raw);owner.release(invalid);
    BlockLayout layout=BlockLayout::Nvfp4Planes;
    Nvfp4View canonical{data,data+values,rows,columns,global,1};
    const bool qkvz=compensated_ffn_loading()&&qkvz_loading(canonical,tensor.name);
    if(compensated_ffn_loading()&&((compensated_ffn_shape(canonical)&&compensated_ffn_tensor(tensor.name))||
        (compensated_projection_requested()&&compensated_projection_shape(canonical)&&compensated_projection_tensor(tensor.name))||qkvz)){
        auto *packed=static_cast<unsigned char*>(owner.allocate(values+nvfp4_scale_bytes(rows,columns)));
        try{
            fp4_pack_values(canonical,packed,stream);
            check_cuda(cudaMemcpyAsync(packed+values,data+values,nvfp4_scale_bytes(rows,columns),
                cudaMemcpyDeviceToDevice,stream),"packed FFN immutable SF plane");
            check_cuda(cudaStreamSynchronize(stream),"packed FFN publication lease");
        }catch(...){cudaStreamSynchronize(stream);owner.release(packed);owner.release(data);throw;}
        owner.release(data);data=packed;layout=BlockLayout::Nvfp4WarpFragments;compensated_ffn_publish(data);if(qkvz)qkvz_publish({data,data+values,rows,columns,global,1});
    }
    view={data,data+values,rows,columns,global,1};return {data,Quant::NVFP4,rows,columns,layout};
}
}
