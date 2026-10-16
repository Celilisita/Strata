#include "strata/qwen35/native/mtp_draft_link.hpp"
#include "strata/qwen35/native/ffn.hpp"
namespace strata::qwen35::native {
namespace {
__global__ void link(const int *picked,const unsigned *invalid,int vocabulary,
                     int *next,int *record,unsigned *flags){
    if(!threadIdx.x){
        const int id=*picked;const unsigned bad=*invalid||id<0||id>=vocabulary;
        *record=id;*flags=bad;*next=bad?0:id;
    }
}
}
void mtp_draft_link(const int *picked,const unsigned *invalid,int vocabulary,
                    int *next,int *record,unsigned *flags,cudaStream_t stream){
    if(!picked||!invalid||!next||!record||!flags||vocabulary<1)
        throw std::invalid_argument("invalid GPU MTP draft link");
    link<<<1,32,0,stream>>>(picked,invalid,vocabulary,next,record,flags);
    check_cuda(cudaGetLastError(),"GPU MTP draft ID and validity record");
}
}
