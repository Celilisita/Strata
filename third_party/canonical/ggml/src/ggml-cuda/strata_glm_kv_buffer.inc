#include "strata_glm_kv_window.cuh"
#include <limits>
#include <stdexcept>

struct StrataGlmKvBufferType {int device;uint32_t resident;std::string name;};
// Stable object identity survives MSVC /OPT:ICF folding of get_name functions.
static ggml_backend_buffer_type strata_glm_kv_types[GGML_CUDA_MAX_DEVICES]{};
struct StrataGlmKvBuffer {
    int device;uint32_t resident;void* cpu=nullptr;unsigned char* mapped=nullptr;
    std::vector<StrataGlmKvWindow> tensors;
};
static const char* strata_glm_kv_buft_name(ggml_backend_buffer_type_t b) {
    return static_cast<StrataGlmKvBufferType*>(b->context)->name.c_str();
}
static bool strata_glm_kv_buft(ggml_backend_buffer_type_t b) {
    if (!b) return false;
    for (auto& type : strata_glm_kv_types) if (b == &type) return type.context != nullptr;
    return false;
}
bool strata_glm_kv_window_view(const ggml_tensor* t,StrataGlmKvWindow& result) {
    const ggml_tensor* owner=t;
    while(owner&&owner->view_src)owner=owner->view_src;
    if(!owner||!owner->buffer||!strata_glm_kv_buft(owner->buffer->buft))return false;
    auto* c=static_cast<StrataGlmKvBuffer*>(owner->buffer->context);
    const auto address=reinterpret_cast<uintptr_t>(t->data);
    for(const auto& v:c->tensors) {
        const auto begin=reinterpret_cast<uintptr_t>(v.host);
        if(address>=begin&&address-begin<uint64_t(v.logical_bytes)){result=v;return true;}
    }
    return false;
}
void strata_glm_kv_window_written(const ggml_tensor* t,void* stream) {
    StrataGlmKvWindow view{};
    if(!strata_glm_kv_window_view(t,view))return;
    // CPY writes canonical packed bytes to authoritative host storage. Never
    // publish an old cached row after that write; misses read the new host row.
    CUDA_CHECK(cudaMemsetAsync(view.tags,0xff,size_t(view.resident_tokens)*sizeof(int64_t),
        static_cast<cudaStream_t>(stream)));
}
static void strata_glm_kv_invalidate(StrataGlmKvBuffer& c) {
    ggml_cuda_set_device(c.device);
    for(auto& v:c.tensors)CUDA_CHECK(cudaMemset(v.tags,0xff,size_t(v.resident_tokens)*sizeof(int64_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
}
static void strata_glm_kv_free(ggml_backend_buffer_t b) {
    auto* c=static_cast<StrataGlmKvBuffer*>(b->context);ggml_cuda_set_device(c->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    for(auto& v:c->tensors) {
        strata_glm_kv_attention_update(c->device,v,true);
        CUDA_CHECK(cudaFree(v.device));CUDA_CHECK(cudaFree(v.tags));
    }
    CUDA_CHECK(cudaFreeHost(c->cpu));delete c;
}
static void* strata_glm_kv_base(ggml_backend_buffer_t b) {
    return static_cast<StrataGlmKvBuffer*>(b->context)->mapped;
}
static ggml_status strata_glm_kv_init(ggml_backend_buffer_t b,ggml_tensor* t) {
    if(t->view_src)return GGML_STATUS_SUCCESS;
    auto* c=static_cast<StrataGlmKvBuffer*>(b->context);ggml_cuda_set_device(c->device);
    if((t->type!=GGML_TYPE_F16&&t->type!=GGML_TYPE_Q8_0)||
       t->ne[0]%ggml_blck_size(t->type)!=0||
       t->ne[2]!=1||t->ne[3]!=1||t->ne[1]<1||!ggml_is_contiguous(t))
        return GGML_STATUS_FAILED;
    StrataGlmKvWindow v{};v.host=static_cast<unsigned char*>(t->data);v.row_bytes=int64_t(t->nb[1]);
    v.resident_tokens=std::min<int64_t>(t->ne[1],c->resident);v.logical_bytes=int64_t(ggml_nbytes(t));
    v.type=t->type;v.row_elements=t->ne[0];
    const size_t physical=size_t(v.row_bytes)*size_t(v.resident_tokens);
    auto e=cudaMalloc(reinterpret_cast<void**>(&v.device),physical);
    if(e!=cudaSuccess){(void)cudaGetLastError();return GGML_STATUS_ALLOC_FAILED;}
    e=cudaMalloc(reinterpret_cast<void**>(&v.tags),size_t(v.resident_tokens)*sizeof(int64_t));
    if(e!=cudaSuccess){(void)cudaGetLastError();cudaFree(v.device);return GGML_STATUS_ALLOC_FAILED;}
    CUDA_CHECK(cudaMemset(v.tags,0xff,size_t(v.resident_tokens)*sizeof(int64_t)));
    c->tensors.push_back(v);strata_glm_kv_attention_update(c->device,v,false);
    GGML_LOG_INFO("strata_glm_kv_window tensor=%s logical_rows=%lld resident_rows=%lld host_bytes=%lld device_bytes=%zu\n",
        t->name,(long long)t->ne[1],(long long)v.resident_tokens,(long long)v.logical_bytes,physical);
    return GGML_STATUS_SUCCESS;
}
[[noreturn]] static void strata_glm_kv_range_error(const char* method,const char* why,size_t off,size_t bytes) {
    throw std::runtime_error(std::string("GLM KV CPU range method=")+method+" reason="+why+
        " offset="+std::to_string(off)+" bytes="+std::to_string(bytes));
}
static StrataGlmKvBuffer& strata_glm_kv_context(ggml_backend_buffer_t b,const char* method,size_t off,size_t bytes) {
    if(!b||!b->context||!strata_glm_kv_buft(b->buft))
        strata_glm_kv_range_error(method,"invalid_buffer",off,bytes);
    auto& c=*static_cast<StrataGlmKvBuffer*>(b->context);
    if(!c.cpu||!c.mapped)strata_glm_kv_range_error(method,"missing_allocation",off,bytes);
    return c;
}
static unsigned char* strata_glm_kv_cpu(ggml_backend_buffer_t b,const ggml_tensor* t,
        size_t off,size_t bytes,const char* method) {
    auto& c=strata_glm_kv_context(b,method,off,bytes);
    size_t start=0;
    if(t) {
        const uintptr_t address=reinterpret_cast<uintptr_t>(t->data);
        const uintptr_t mapped=reinterpret_cast<uintptr_t>(c.mapped);
        if(address<mapped||address-mapped>b->size)
            strata_glm_kv_range_error(method,"tensor_outside_allocation",off,bytes);
        start=size_t(address-mapped);
        const size_t tensor_bytes=ggml_nbytes(t);
        if(off>tensor_bytes||bytes>tensor_bytes-off)
            strata_glm_kv_range_error(method,"tensor_extent",off,bytes);
    }
    if(start>b->size||off>b->size-start||bytes>b->size-start-off)
        strata_glm_kv_range_error(method,"allocation_extent",off,bytes);
    // Only the actual CPU allocation can be dereferenced by std::memcpy.
    // mapped/tensor->data are used solely to validate a bounded integer offset.
    return static_cast<unsigned char*>(c.cpu)+start+off;
}
static size_t strata_glm_kv_2d_span(const char* method,size_t off,size_t bytes,size_t copies,size_t stride) {
    if(!copies||!bytes)return 0;
    if(copies>1&&stride>(std::numeric_limits<size_t>::max()-bytes)/(copies-1))
        strata_glm_kv_range_error(method,"stride_overflow",off,bytes);
    return (copies-1)*stride+bytes;
}
static void strata_glm_kv_external(const void* data,const char* method,size_t off,size_t bytes) {
    if(bytes&&!data)strata_glm_kv_range_error(method,"null_external_pointer",off,bytes);
    if(bytes&&reinterpret_cast<uintptr_t>(data)>std::numeric_limits<uintptr_t>::max()-(bytes-1))
        strata_glm_kv_range_error(method,"external_pointer_overflow",off,bytes);
}
static void strata_glm_kv_set(ggml_backend_buffer_t b,ggml_tensor* t,const void* data,size_t off,size_t bytes) {
    auto& c=strata_glm_kv_context(b,"set",off,bytes);
    auto* cpu=strata_glm_kv_cpu(b,t,off,bytes,"set");strata_glm_kv_external(data,"set",off,bytes);
    if(!bytes)return;
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());
    std::memcpy(cpu,data,bytes);strata_glm_kv_invalidate(c);
}
static void strata_glm_kv_get(ggml_backend_buffer_t b,const ggml_tensor* t,void* data,size_t off,size_t bytes) {
    auto& c=strata_glm_kv_context(b,"get",off,bytes);
    auto* cpu=strata_glm_kv_cpu(b,t,off,bytes,"get");strata_glm_kv_external(data,"get",off,bytes);
    if(!bytes)return;
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());std::memcpy(data,cpu,bytes);
}
static void strata_glm_kv_memset(ggml_backend_buffer_t b,ggml_tensor* t,uint8_t value,size_t off,size_t bytes) {
    auto& c=strata_glm_kv_context(b,"memset",off,bytes);
    auto* cpu=strata_glm_kv_cpu(b,t,off,bytes,"memset");if(!bytes)return;
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());
    std::memset(cpu,value,bytes);strata_glm_kv_invalidate(c);
}
static void strata_glm_kv_clear(ggml_backend_buffer_t b,uint8_t value) {
    auto& c=strata_glm_kv_context(b,"clear",0,b?b->size:0);
    auto* cpu=strata_glm_kv_cpu(b,nullptr,0,b->size,"clear");
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());
    std::memset(cpu,value,b->size);strata_glm_kv_invalidate(c);
}
static void strata_glm_kv_set_2d(ggml_backend_buffer_t b,ggml_tensor* t,const void* data,
        size_t off,size_t bytes,size_t copies,size_t stride_tensor,size_t stride_data) {
    auto& c=strata_glm_kv_context(b,"set_2d",off,bytes);
    const size_t tensor_span=strata_glm_kv_2d_span("set_2d",off,bytes,copies,stride_tensor);
    const size_t data_span=strata_glm_kv_2d_span("set_2d",off,bytes,copies,stride_data);
    auto* cpu=strata_glm_kv_cpu(b,t,off,tensor_span,"set_2d");strata_glm_kv_external(data,"set_2d",off,data_span);
    if(!copies||!bytes)return;
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());
    for(size_t i=0;i<copies;++i)std::memcpy(cpu+i*stride_tensor,static_cast<const char*>(data)+i*stride_data,bytes);
    strata_glm_kv_invalidate(c);
}
static void strata_glm_kv_get_2d(ggml_backend_buffer_t b,const ggml_tensor* t,void* data,
        size_t off,size_t bytes,size_t copies,size_t stride_tensor,size_t stride_data) {
    auto& c=strata_glm_kv_context(b,"get_2d",off,bytes);
    const size_t tensor_span=strata_glm_kv_2d_span("get_2d",off,bytes,copies,stride_tensor);
    const size_t data_span=strata_glm_kv_2d_span("get_2d",off,bytes,copies,stride_data);
    auto* cpu=strata_glm_kv_cpu(b,t,off,tensor_span,"get_2d");strata_glm_kv_external(data,"get_2d",off,data_span);
    if(!copies||!bytes)return;
    ggml_cuda_set_device(c.device);CUDA_CHECK(cudaDeviceSynchronize());
    for(size_t i=0;i<copies;++i)std::memcpy(static_cast<char*>(data)+i*stride_data,cpu+i*stride_tensor,bytes);
}
static const ggml_backend_buffer_i strata_glm_kv_buffer_iface={
    strata_glm_kv_free,strata_glm_kv_base,strata_glm_kv_init,strata_glm_kv_memset,
    strata_glm_kv_set,strata_glm_kv_get,strata_glm_kv_set_2d,strata_glm_kv_get_2d,nullptr,strata_glm_kv_clear,nullptr};
static ggml_backend_buffer_t strata_glm_kv_alloc(ggml_backend_buffer_type_t b,size_t bytes) {
    auto& type=*static_cast<StrataGlmKvBufferType*>(b->context);ggml_cuda_set_device(type.device);
    auto* c=new StrataGlmKvBuffer{type.device,type.resident};
    auto e=cudaHostAlloc(&c->cpu,bytes,cudaHostAllocMapped|cudaHostAllocPortable);
    if(e==cudaSuccess)e=cudaHostGetDevicePointer(reinterpret_cast<void**>(&c->mapped),c->cpu,0);
    if(e!=cudaSuccess){(void)cudaGetLastError();if(c->cpu)cudaFreeHost(c->cpu);delete c;return nullptr;}
    return ggml_backend_buffer_init(b,strata_glm_kv_buffer_iface,c,bytes);
}
static ggml_backend_buffer_type_t strata_glm_kv_buffer_type(ggml_backend_dev_t dev,uint32_t resident) {
    if(resident!=32768)return nullptr;
    static std::mutex mutex;std::lock_guard<std::mutex> guard(mutex);
    for(int i=0;i<ggml_backend_cuda_get_device_count();++i)if(ggml_backend_cuda_buffer_type(i)->device==dev) {
        auto& b=strata_glm_kv_types[i];if(!b.context) {
            b.iface=ggml_backend_cuda_buffer_type_interface;b.iface.get_name=strata_glm_kv_buft_name;
            b.iface.alloc_buffer=strata_glm_kv_alloc;b.iface.get_alloc_size=nullptr;
            b.device=dev;b.context=new StrataGlmKvBufferType{i,resident,"STRATA_GLM_KV_STREAM"+std::to_string(i)};
        }
        return &b;
    }
    return nullptr;
}
