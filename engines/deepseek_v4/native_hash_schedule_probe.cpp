#include "native_read_ahead.hpp"
#include "cpu_weight_fixture.hpp"
#include "ggml-cuda.h"
#include "llama-model.h"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <memory>
using namespace strata::deepseek_v4;
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("Expected expert shard and report");CpuWeightFixture fixture(argv[1]);
    auto* backend=ggml_backend_cuda_init(0);if(!backend)throw std::runtime_error("CUDA absent");
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));using Reserve=bool(*)(ggml_backend_t);
    auto reserve=reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_reserve"));if(!reserve||!reserve(backend))throw std::runtime_error("Mapped prefetch reserve absent");
    auto* packet=router_prefetch_packet(backend);if(!packet)throw std::runtime_error("Mapped producer absent");
    auto model=std::unique_ptr<llama_model>(llama_model_create(LLM_ARCH_DEEPSEEK4,llama_model_default_params()));if(!model)throw std::runtime_error("Metadata model factory absent");model->layers.resize(43);model->hparams.swiglu_clamp_exp.fill(10);
    for(auto& l:model->layers){l.ffn_gate_exps=const_cast<ggml_tensor*>(fixture.bank.gate);l.ffn_up_exps=const_cast<ggml_tensor*>(fixture.bank.up);l.ffn_down_exps=const_cast<ggml_tensor*>(fixture.bank.down);}
    Mxfp4CpuService cpu(8);auto real=std::make_unique<DeviceMissPacket>();std::atomic<bool> stop{false},active{true};
    auto ready=[](const DeviceMissPacket* p){std::atomic_thread_fence(std::memory_order_acquire);return p->sequence;};
    NativeReadAhead reads(backend,*model,cpu,real.get(),ready,stop,active);
    auto publish=[&](bool both){const auto next=packet->hash_sequence+1;for(int layer=0;layer<3;++layer){RouterPrefetchRoutes r{};r.epoch=1;r.layer=layer;r.tokens=both?2:1;r.source=both?2:1;
        r.masks[0]=63;r.masks[1]=both?63:0;r.selected[0]=r.selected[1]=-1;for(int i=0;i<12;++i)r.ids[i]=i;packet->hash[next&1][layer]=r;}
        std::atomic_thread_fence(std::memory_order_release);packet->hash_sequence=next;};
    publish(false);if(!reads.poll(0))throw std::runtime_error("Pending token front window missed");
    real->layer=43;real->sequence=1;real->native_request=2;reads.observe(*real);
    if(!reads.poll(1))throw std::runtime_error("Pending token head window didnotresume");
    publish(true);if(!reads.poll(1))throw std::runtime_error("Two-position first hash didnotread");
    for(int layer=0;layer<3;++layer){real->layer=layer;real->sequence=layer+2;real->native_request=3;real->cpu_mask=4095;
        for(int i=0;i<12;++i)real->ids[i]=i;reads.observe(*real);if(layer<2&&!reads.poll(layer+2))throw std::runtime_error("Next exact hash window missed");}
    const auto metrics=reads.metrics();const bool window_limited=std::getenv("STRATA_DSV4_PREFETCH_WINDOW_AWARE")!=nullptr;
    const bool passed=metrics.jobs==5&&metrics.hash_jobs==5&&metrics.hash_bytes>0&&
        (!window_limited||(metrics.selected_one_jobs==2&&metrics.selected_two_jobs==3))&&
        metrics.observed_windows==3&&metrics.layer_window_count[0]==1&&metrics.layer_window_count[1]==1&&metrics.layer_window_count[2]==1;
    std::ofstream out(argv[2]);out<<"{\"whole_model_loaded\":false,\"main_model_object_is_metadata_only\":true,\"real_mxfp4_pages_read\":true,\"gpu_prediction_not_run_in_queue_fixture\":true,\"foreground_tickets_simulated\":true,\"hash_jobs\":"<<metrics.hash_jobs<<",\"hash_bytes\":"<<metrics.hash_bytes
        <<",\"observed_windows\":"<<metrics.observed_windows<<",\"completed_jobs\":"<<metrics.completed_jobs
        <<",\"selected_one_jobs\":"<<metrics.selected_one_jobs<<",\"selected_two_jobs\":"<<metrics.selected_two_jobs
        <<",\"canceled_after_completion\":"<<metrics.canceled_after_completion
        <<",\"first_three_window_ns\":["<<metrics.layer_window_ns[0]<<','<<metrics.layer_window_ns[1]<<','<<metrics.layer_window_ns[2]
        <<"],\"first_three_read_ns\":["<<metrics.layer_read_ns[0]<<','<<metrics.layer_read_ns[1]<<','<<metrics.layer_read_ns[2]
        <<"],\"front_then_MTP_CPU_then_head_then_three_hash_layers_passed\":"<<(passed?"true":"false")<<",\"passed\":"<<(passed?"true":"false")<<'}';return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
