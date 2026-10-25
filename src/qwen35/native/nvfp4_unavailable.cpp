#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include <stdexcept>
#include "strata/qwen35/native/request_layout.hpp"
namespace strata::qwen35::native {
namespace {[[noreturn]] void unavailable(){throw std::runtime_error("native NVFP4 requires CUDA toolkit >=12.8 and admitted SM120 hardware");}}
Matrix load_nvfp4_matrix(const strata::GgufFile &,const strata::TensorInfo &,DeviceStorage &,Nvfp4View &,cudaStream_t){unavailable();}
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &,const std::vector<Nvfp4View> &,int,int,cudaStream_t){unavailable();}
std::unique_ptr<DenseGemm> make_nvfp4_runtime(DenseGemm &,const std::vector<Nvfp4View> &,int,int,cudaStream_t,Matrix){unavailable();}
void nvfp4_project_head(DenseGemm &,Matrix,const float *,float *,int,cudaStream_t){unavailable();}
void nvfp4_configure_head_request(DenseGemm &,bool){}
TinyHeadCounters nvfp4_head_counters(DenseGemm &){return {};}
bool nvfp4_request_layout_change_required(DenseGemm &,int){return false;}
void nvfp4_prepare_request_layout(DenseGemm &,int,bool,cudaStream_t){}
bool nvfp4_shared_projections_enabled(){return false;}
void nvfp4_project_set(DenseGemm &,const float *,std::initializer_list<Nvfp4Projection>,int,cudaStream_t){unavailable();}
void nvfp4_project(DenseGemm &,Matrix,const float *,float *,int,cudaStream_t){unavailable();}
void nvfp4_ffn(DenseGemm &,Matrix,Matrix,Matrix,const float *,const Buffers &,int,cudaStream_t){unavailable();}
bool nvfp4_ffn_fused(DenseGemm &,Matrix,Matrix,Matrix,const float *,const Buffers &,int,cudaStream_t,bool){unavailable();}
}
