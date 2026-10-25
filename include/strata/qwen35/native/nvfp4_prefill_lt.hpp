#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace strata::qwen35::native {
struct Nvfp4LtInfo {
    bool ready=false;
    int rows=0,columns=0,tokens=0,algorithm=-1,tile=-1,stages=-1,splitk=-1,reduction=-1,swizzle=-1,custom=-1,sm=0;
    size_t library=0,workspace=0;
};
class Nvfp4PrefillLt {
    void* implementation_=nullptr;
public:
    Nvfp4PrefillLt(int rows,int columns,int tokens);
    ~Nvfp4PrefillLt();
    Nvfp4PrefillLt(const Nvfp4PrefillLt&)=delete;
    Nvfp4LtInfo info()const;
    bool supports(int rows,int columns,int tokens)const;
    void project(const void* weights,const void* input,float* output,const float* alpha,const float* zero,
        void* workspace,size_t bytes,cudaStream_t);
};
}
