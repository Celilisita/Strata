#include "device_cache_budget.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include <iostream>
#include <stdexcept>
using namespace strata::deepseek_v4;
int main(){try{
    if(pp_cache_slots_with_headroom(1284)<6)throw std::runtime_error("headroom capacity invalid");
    llama_backend_init();auto* device=ggml_backend_dev_by_name("CUDA0");if(!device)throw std::runtime_error("CUDA0 missing");
    auto* backend=ggml_backend_dev_init(device,nullptr);if(!backend)throw std::runtime_error("CUDA backend missing");
    const auto before=device_memory_budget(backend);
    auto* buffer=ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend),64ull<<20);
    if(!buffer)throw std::runtime_error("budget fixture allocation failed");
    ggml_backend_buffer_clear(buffer,0);ggml_backend_synchronize(backend);
    const auto after=device_memory_budget(backend);
#ifdef _WIN32
    if(!before.wddm_available||!after.wddm_available||!after.local_budget||after.local_usage<before.local_usage+(64ull<<20))
        throw std::runtime_error("selected adapter WDDM usage did not include live CUDA allocation");
#endif
    std::cout<<"{\"whole_model_loaded\":false,\"passed\":true,\"wddm_available\":"<<(after.wddm_available?"true":"false")
        <<",\"local_budget\":"<<after.local_budget<<",\"local_usage_before\":"<<before.local_usage<<",\"local_usage_after\":"<<after.local_usage
        <<",\"cuda_total\":"<<after.cuda_total<<",\"pp_slots_after_headroom\":"<<pp_cache_slots_with_headroom(1284)<<"}\n";
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
