#pragma once
#include "pp_prefetch.hpp"
#include "pp_prefetch_gdn.hpp"
#include <array>
#include <vector>
namespace strata::completed_v100::qwen35::native {
class PrefetchCore {
public:
    struct Lease {
        Matrix gate{},up{},down{};Buffers scratch{};int maximum=0;
        GdnInputs input{};GdnShape shape{};float* attention=nullptr;float* state=nullptr;
        void* work=nullptr;size_t work_bytes=0;
    } lease;
    using Key=std::array<uintptr_t,23>;
    struct Entry {
        Key key{};cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        Entry()=default;Entry(const Entry&)=delete;Entry& operator=(const Entry&)=delete;
        Entry(Entry&& x)noexcept:key(x.key),graph(x.graph),executable(x.executable){x.graph=nullptr;x.executable=nullptr;}
        ~Entry(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
    };
    bool enabled=false,pending=false;
    cudaStream_t main=nullptr,auxiliary=nullptr;cudaEvent_t fork=nullptr,ready=nullptr;
    cublasHandle_t handle=nullptr;void* vendor=nullptr;size_t vendor_bytes=0;const float *one=nullptr,*zero=nullptr;
    std::vector<Entry> entries;
    unsigned long long calls=0,built=0,reused=0;
    explicit PrefetchCore(Nvfp4BlasLease);~PrefetchCore();
    void reset(cudaStream_t);
    void bind(cudaStream_t);
    void begin(cudaStream_t);
    void gdn(cudaStream_t);
    void ffn(const float*,float*,float*,int,cudaStream_t);
    static void callback(void* owner,cudaStream_t stream){static_cast<PrefetchCore*>(owner)->begin(stream);}
    Key key()const;
};
}
