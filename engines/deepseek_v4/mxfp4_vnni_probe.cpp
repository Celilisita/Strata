#include "mtp_sidecar.hpp"
#include "mxfp4_cpu_service.hpp"
#include "mxfp4_fast_dot.hpp"
#include "checkpoint.hpp"
#include "benchmark_fault_guard.hpp"
#include "ggml-cpu.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::deepseek_v4;
static void select_kernel(bool fast) {
#ifdef _WIN32
    _putenv_s("STRATA_DSV4_CPU_VNNI",fast?"1":"0");
#else
    setenv("STRATA_DSV4_CPU_VNNI",fast?"1":"0",1);
#endif
}
struct Error {double rms=0,absolute=0;};
static Error compare(const std::vector<float>& a,const std::vector<float>& b) {
    double error=0,norm=0;Error result;
    for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("Nonfinite CPU contribution");
        const double d=double(a[i])-b[i];error+=d*d;norm+=double(a[i])*a[i];result.absolute=std::max(result.absolute,std::abs(d));}
    result.rms=std::sqrt(error/std::max(norm,1e-30));return result;
}
static double elapsed(std::chrono::steady_clock::time_point t){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();}
int main(int argc,char** argv)try {
    benchmark_fault_guard();if(argc!=4)throw std::runtime_error("Expected original MTP GGUF, real activation fixture, new JSON report");
    llama_backend_init();if(!mxfp4_fast_dot_supported())throw std::runtime_error("AVX512 VNNI unavailable; production fallback remains active");
    auto* device=ggml_backend_dev_by_name("CUDA0");if(!device)throw std::runtime_error("CUDA0 missing");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    MtpSidecar mtp(argv[1],backend.get());const auto fixture=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(fixture.metadata.type!=GGML_TYPE_F32||fixture.data.size()<8192*sizeof(float))throw std::runtime_error("Two actual activations absent");
    std::array<float,8192> x{};std::memcpy(x.data(),fixture.data.data(),sizeof(x));
    Mxfp4CpuBank bank{mtp.tensor("blk.43.ffn_gate_exps.weight"),mtp.tensor("blk.43.ffn_up_exps.weight"),mtp.tensor("blk.43.ffn_down_exps.weight"),10};
    // Reference and candidate have independently selected, fixed kernels.
    select_kernel(false);Mxfp4CpuService reference(4);select_kernel(true);
    std::array<std::unique_ptr<Mxfp4CpuService>,3> candidates;const int threads[]{4,8,16};
    for(int i=0;i<3;++i)candidates[i]=std::make_unique<Mxfp4CpuService>(threads[i]);
    // Pre-touch the small 3.2GiB sidecar once, separating page faults from CPU timing.
    uint64_t touched=0;for(auto* t:{bank.gate,bank.up,bank.down}){const auto* bytes=static_cast<const volatile uint8_t*>(t->data);for(size_t i=0;i<ggml_nbytes(t);i+=4096)touched+=bytes[i];}
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);const auto* quant=ggml_get_type_traits_cpu(traits->vec_dot_type);
    std::vector<float> dot_expected,dot_actual;std::array<uint8_t,8704> q{};
    for(auto* t:{bank.gate,bank.up,bank.down}){const int n=int(t->ne[0]);quant->from_float(x.data(),q.data(),n);quant->from_float(x.data()+4096,q.data()+4352,n);
        for(int expert=0;expert<256;expert+=17)for(int row=0;row<int(t->ne[1]);row+=251){
            const auto* weight=static_cast<const char*>(t->data)+expert*t->nb[2]+row*t->nb[1];float candidate[2];mxfp4_fast_dot(n,weight,q.data(),q.data()+4352,candidate);
            for(int position=0;position<2;++position){float ref;traits->vec_dot(n,&ref,0,weight,0,q.data()+position*4352,0,1);dot_expected.push_back(ref);dot_actual.push_back(candidate[position]);}}}
    const auto dot_error=compare(dot_expected,dot_actual);
    // Different FP32 dot reductions can cross a subsequent Q8_0 rounding tie.
    // Bound dots tightly above, and complete post-GLU contributions separately.
    std::vector<float> expected(49152),actual(49152);double rotated_rms=0,rotated_absolute=0;bool passed=dot_error.rms<=2e-6&&dot_error.absolute<=.001;
    std::vector<float> single_expected(24576),single_actual(24576);const int32_t single_ids[]{0,17,63,128,191,255};
    for(auto& candidate:candidates){reference.compute(bank,x.data(),single_ids,6,single_expected.data());candidate->compute(bank,x.data(),single_ids,6,single_actual.data());
        auto e=compare(single_expected,single_actual);passed&=e.rms<=1e-4&&e.absolute<=.01;}
    std::ofstream out(argv[3]);out<<"{\"whole_model_loaded\":false,\"main_head_loaded\":false,\"real_mtp_weights\":true,\"vnni_supported\":true,\"cases\":[";
    for(int common=0;common<=6;++common) {
        std::array<int32_t,12> ids{};for(int i=0;i<6;++i){ids[i]=i;ids[i+6]=i<common?i:i+6;}
        uint32_t masks[]{63,63};reference.compute_pair(bank,x.data(),ids.data(),masks,expected.data());
        if(common)out<<',';out<<"{\"common_experts\":"<<common<<",\"workers\":[";
        for(int k=0;k<3;++k){candidates[k]->compute_pair(bank,x.data(),ids.data(),masks,actual.data());auto error=compare(expected,actual);bool ok=error.rms<=1e-4&&error.absolute<=.01;
            // Partial ranks must retain original output ranks and zero GPU rows.
            uint32_t partial[]{21,42};reference.compute_pair(bank,x.data(),ids.data(),partial,expected.data());candidates[k]->compute_pair(bank,x.data(),ids.data(),partial,actual.data());
            auto partial_error=compare(expected,actual);ok&=partial_error.rms<=1e-4&&partial_error.absolute<=.01;passed&=ok;
            reference.compute_pair(bank,x.data(),ids.data(),masks,expected.data());
            std::array<double,9> baseline{},fast{};
            // Rotate through all 256 experts; do not time an L3-resident fixture.
            for(int r=0;r<11;++r){for(int i=0;i<6;++i){ids[i]=(r*19+i)%256;ids[i+6]=(r*19+(i<common?i:i+6))%256;}
                const bool alternate=r&1;auto a=std::chrono::steady_clock::now();if(alternate)candidates[k]->compute_pair(bank,x.data(),ids.data(),masks,actual.data());else reference.compute_pair(bank,x.data(),ids.data(),masks,expected.data());const double first=elapsed(a);
                a=std::chrono::steady_clock::now();if(alternate)reference.compute_pair(bank,x.data(),ids.data(),masks,expected.data());else candidates[k]->compute_pair(bank,x.data(),ids.data(),masks,actual.data());const double second=elapsed(a);
                auto rotated_error=compare(expected,actual);rotated_rms=std::max(rotated_rms,rotated_error.rms);rotated_absolute=std::max(rotated_absolute,rotated_error.absolute);passed&=rotated_error.rms<=1e-4&&rotated_error.absolute<=.01;
                if(r>=2){baseline[r-2]=alternate?second:first;fast[r-2]=alternate?first:second;}}
            std::array<double,9> partial_baseline{},partial_fast{};
            for(int r=0;r<11;++r){for(int i=0;i<6;++i){ids[i]=(r*23+i)%256;ids[i+6]=(r*23+(i<common?i:i+6))%256;}
                auto a=std::chrono::steady_clock::now();reference.compute_pair(bank,x.data(),ids.data(),partial,expected.data());const double first=elapsed(a);
                a=std::chrono::steady_clock::now();candidates[k]->compute_pair(bank,x.data(),ids.data(),partial,actual.data());const double second=elapsed(a);
                const auto e=compare(expected,actual);ok&=e.rms<=1e-4&&e.absolute<=.01;passed&=ok;
                if(r>=2){partial_baseline[r-2]=first;partial_fast[r-2]=second;}}
            std::sort(partial_baseline.begin(),partial_baseline.end());std::sort(partial_fast.begin(),partial_fast.end());
            // Subsequent candidate comparisons must use the current full ranks.
            reference.compute_pair(bank,x.data(),ids.data(),masks,expected.data());
            std::sort(baseline.begin(),baseline.end());std::sort(fast.begin(),fast.end());if(k)out<<',';
            out<<"{\"threads\":"<<threads[k]<<",\"baseline_4_threads_ms\":"<<baseline[4]<<",\"candidate_ms\":"<<fast[4]<<",\"partial_baseline_4_threads_ms\":"<<partial_baseline[4]<<",\"partial_candidate_ms\":"<<partial_fast[4]<<",\"relative_rms\":"<<error.rms<<",\"max_abs\":"<<error.absolute<<",\"partial_relative_rms\":"<<partial_error.rms<<",\"passed\":"<<(ok?"true":"false")<<'}';
        }out<<"]}";out.flush();
    }
    out<<"],\"page_touch_checksum\":"<<touched<<",\"row_dot_relative_rms\":"<<dot_error.rms<<",\"row_dot_max_abs\":"<<dot_error.absolute<<",\"rotating_full_contribution_max_relative_rms\":"<<rotated_rms<<",\"rotating_full_contribution_max_abs\":"<<rotated_absolute<<",\"post_glu_q8_rms_tolerance\":0.0001,\"passed\":"<<(passed?"true":"false")<<"}\n";if(!out)throw std::runtime_error("CPU receipt write failed");return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
