#pragma once
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "strata/qwen35/completed_v100/native/gdn.hpp"
#include <memory>
#include <string>
namespace strata::completed_v100::qwen35::native {
using PrefetchGdnKernel=void(*)(GdnInputs,GdnShape,float*,float*,void*,size_t,cudaStream_t);
class PrefetchCore;
class PrefetchNvfp4:public Nvfp4Gemm {
    std::unique_ptr<PrefetchCore> core_;
public:
    PrefetchNvfp4(size_t,int,int);
    ~PrefetchNvfp4();
    void select(std::string&);
    void reset(cudaStream_t);
    bool prepare(Matrix,Matrix,Matrix,Buffers,int,GdnInputs,GdnShape,float*,float*,void*,size_t,PrefetchGdnKernel,cudaStream_t);
    bool consume(Matrix,Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
    unsigned long long recorded()const;
    unsigned long long builds()const;
    unsigned long long hits()const;
};
void pp_prefetch_select(DenseGemm*,std::string&);
void pp_prefetch_reset(DenseGemm*,cudaStream_t);
bool try_pp_prefetch_gdn(DenseGemm*,Matrix,Matrix,Matrix,Buffers,int,GdnInputs,GdnShape,float*,float*,void*,size_t,PrefetchGdnKernel,cudaStream_t);
bool try_pp_prefetch_ffn(DenseGemm*,Matrix,Matrix,Matrix,const float*,float*,size_t,float*,int,cudaStream_t);
unsigned long long pp_prefetch_recorded(DenseGemm*);
unsigned long long pp_prefetch_builds(DenseGemm*);
unsigned long long pp_prefetch_hits(DenseGemm*);
}
