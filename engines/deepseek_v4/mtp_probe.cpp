#include "mtp_sidecar.hpp"
#include "mtp_math.hpp"
#include "mtp_drafter.hpp"
#include "mtp_shared_weights.hpp"
#include "mtp_native_probe.hpp"
#include "mtp_stage_probe.hpp"
#include "benchmark_fault_guard.hpp"
#include "cuda_pool_retirement.hpp"
#include "mxfp4_cpu_service.hpp"
#include "checkpoint.hpp"
#include "ggml-alloc.h"
#include "llama.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
using namespace strata::deepseek_v4;
void mtp_pool_fixture(ggml_backend_t,size_t);
int main(int argc,char** argv)try {
    benchmark_fault_guard();
    if(argc==2&&std::string(argv[1])=="--fatal-fixture")GGML_ABORT("Strata bounded fatal fixture: diagnostic preserved");
    if(argc!=6)throw std::runtime_error("Expected MTP GGUF, small activation fixture, new report, target head shard, target metadata GGUF");
    llama_backend_init();auto* device=ggml_backend_dev_by_name("CUDA0");if(!device)throw std::runtime_error("CUDA0 missing");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    MtpSidecar mtp(argv[1],backend.get());
    const auto fixture=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(fixture.metadata.type!=GGML_TYPE_F32||fixture.data.size()<2*4096*sizeof(float))throw std::runtime_error("Real small activation fixture missing");
    std::array<float,8192> x{};std::memcpy(x.data(),fixture.data.data(),sizeof(x));
    Mxfp4CpuBank bank{mtp.tensor("blk.43.ffn_gate_exps.weight"),mtp.tensor("blk.43.ffn_up_exps.weight"),mtp.tensor("blk.43.ffn_down_exps.weight"),10};
    Mxfp4CpuService cpu(4);std::vector<float> expected(12*4096),actual(expected.size());
    std::ofstream out(argv[3]);out<<"{\"whole_model_loaded\":false,\"original_mtp_real_weights\":true,\"dense_device_bytes\":"<<mtp.dense_device_bytes()<<",\"pair_cases\":[";
    bool passed=true;
    for(int common=0;common<=6;++common) {
        std::array<int32_t,12> ids{};for(int i=0;i<6;++i){ids[i]=i;ids[i+6]=i<common?i:i+6;}
        const std::array<uint32_t,2> masks{63,63};
        cpu.compute(bank,x.data(),ids.data(),6,expected.data());cpu.compute(bank,x.data()+4096,ids.data()+6,6,expected.data()+6*4096);
        size_t count=cpu.compute_pair(bank,x.data(),ids.data(),masks.data(),actual.data());
        double squared=0,reference=0,max_abs=0;for(size_t i=0;i<actual.size();++i){const double e=actual[i]-expected[i];squared+=e*e;reference+=double(expected[i])*expected[i];max_abs=std::max(max_abs,std::abs(e));}
        const double relative=std::sqrt(squared/std::max(reference,1e-30));const bool ok=count==size_t(12-common)&&std::isfinite(relative)&&relative<=1e-5;passed&=ok;
        auto now=[](){return std::chrono::steady_clock::now();};std::array<double,5> single{},pair{};
        for(int r=0;r<5;++r){auto a=now();cpu.compute(bank,x.data(),ids.data(),6,expected.data());cpu.compute(bank,x.data()+4096,ids.data()+6,6,expected.data()+6*4096);single[r]=std::chrono::duration<double,std::milli>(now()-a).count();
            a=now();cpu.compute_pair(bank,x.data(),ids.data(),masks.data(),actual.data());pair[r]=std::chrono::duration<double,std::milli>(now()-a).count();}
        std::sort(single.begin(),single.end());std::sort(pair.begin(),pair.end());
        if(common)out<<',';out<<"{\"shared_experts\":"<<common<<",\"union_experts\":"<<count<<",\"logical_expert_bytes\":"<<count*13369344ull<<",\"single_twice_ms\":"<<single[2]<<",\"pair_ms\":"<<pair[2]<<",\"relative_rms\":"<<relative<<",\"max_abs\":"<<max_abs<<",\"passed\":"<<(ok?"true":"false")<<'}';
    }
    out<<"]";out.flush();
    // GPU front/head mathematics use authentic small activations, not a main
    // model residual trajectory. Acceptance and complete drafts are not measured.
    std::unique_ptr<ggml_context,decltype(&ggml_free)> meta(ggml_init({4096*ggml_tensor_overhead()+ggml_graph_overhead_custom(4096,false),nullptr,true}),ggml_free);
    auto* c=meta.get();auto* embed=ggml_new_tensor_2d(c,GGML_TYPE_F32,4096,2);auto* hidden=ggml_new_tensor_3d(c,GGML_TYPE_F32,4096,4,2);ggml_set_input(embed);ggml_set_input(hidden);
    auto* front=mtp_front(c,mtp,embed,hidden);auto hc=mtp_hc_pre(c,mtp,front,"ffn");auto* post=mtp_hc_post(c,hc.mixed,front,hc);auto* result=mtp_head_hidden(c,mtp,post);ggml_set_output(result);
    auto* graph=ggml_new_graph_custom(c,4096,false);ggml_build_forward_expand(graph,result);
    std::unique_ptr<ggml_gallocr,decltype(&ggml_gallocr_free)> allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())),ggml_gallocr_free);
    if(!ggml_gallocr_alloc_graph(allocator.get(),graph))throw std::runtime_error("MTP front allocation failed");
    std::array<float,32768> residual{};for(int t=0;t<2;++t)for(int s=0;s<4;++s)for(int i=0;i<4096;++i)residual[(t*4+s)*4096+i]=x[t*4096+i]*(1+s*.025f);
    ggml_backend_tensor_set(embed,x.data(),0,sizeof(x));ggml_backend_tensor_set(hidden,residual.data(),0,sizeof(residual));
    if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MTP front compute failed");
    std::array<float,8192> output{};ggml_backend_tensor_get(result,output.data(),0,sizeof(output));
    const bool finite=std::all_of(output.begin(),output.end(),[](float f){return std::isfinite(f);});passed&=finite;
    const char* pool_fixture=std::getenv("STRATA_DSV4_POOL_FIXTURE");
    if(pool_fixture&&!std::strcmp(pool_fixture,"1")) {
        auto compute_same_input=[&]{ggml_backend_tensor_set(embed,x.data(),0,sizeof(x));ggml_backend_tensor_set(hidden,residual.data(),0,sizeof(residual));
            if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Pool graph compute failed");};
        // Establish an instantiated graph with explicitly refreshed inputs.
        for(int repeat=0;repeat<5;++repeat)compute_same_input();
        ggml_backend_tensor_get(result,output.data(),0,sizeof(output));mtp_pool_fixture(backend.get(),2ull<<30);
        compute_same_input();std::array<float,8192> after_loan{};ggml_backend_tensor_get(result,after_loan.data(),0,sizeof(after_loan));
        double loan_squared=0,loan_norm=0;for(size_t i=0;i<output.size();++i){const double e=double(output[i])-after_loan[i];loan_squared+=e*e;loan_norm+=double(output[i])*output[i];}
        const double loan_rms=std::sqrt(loan_squared/std::max(loan_norm,1e-30));
        const auto reg=ggml_backend_dev_backend_reg(device);auto retire=reinterpret_cast<CudaPoolRetire>(ggml_backend_reg_get_proc_address(reg,"strata_cuda_pool_retire"));
        CudaPoolRetirement state;if(!retire||!retire(backend.get(),&state)||state.released_bytes<(2ull<<30))throw std::runtime_error("Returned CUDA pool loan was not retired");
        for(int repeat=0;repeat<5;++repeat)compute_same_input();
        std::array<float,8192> rebuilt{};ggml_backend_tensor_get(result,rebuilt.data(),0,sizeof(rebuilt));double squared=0,norm=0;
        for(size_t i=0;i<output.size();++i){if(!std::isfinite(rebuilt[i]))throw std::runtime_error("Reconstructed graph nonfinite");const double e=double(output[i])-rebuilt[i];squared+=e*e;norm+=double(output[i])*output[i];}
        const double relative=std::sqrt(squared/std::max(norm,1e-30));passed&=relative<=1e-6&&loan_rms<=1e-6;
        out<<",\"cuda_pool_retirement\":{\"released_bytes\":"<<state.released_bytes<<",\"returned_loan_relative_rms\":"<<loan_rms<<",\"graphs_destroyed\":"<<state.graphs_destroyed<<",\"reconstructed_relative_rms\":"<<relative<<"}";
    }
    MtpSharedWeights shared(argv[4],backend.get());
    std::array<int32_t,2> input_ids{5,11498},positions{0,1};
    out<<",\"bootstrap_draft_forward\":[";
    for(int n=1;n<=2;++n){MtpDrafter drafter(mtp,backend.get(),shared.embedding,shared.head,cpu,n);std::array<double,5> samples{};std::vector<int32_t> draft;
        for(int r=0;r<7;++r){drafter.reset();const auto start=std::chrono::steady_clock::now();draft=drafter.forward(residual.data(),input_ids.data(),positions.data());if(r>=2)samples[r-2]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}
        drafter.reset();std::array<int32_t,2> next_positions{127,128};
        std::vector<float> history(127*16384);std::vector<int32_t> history_ids(127,input_ids[0]),history_positions(127);
        for(int row=0;row<127;++row){std::copy_n(residual.data(),16384,history.data()+row*16384);history_positions[row]=row;}
        drafter.catch_up(history.data(),history_ids.data(),history_positions.data(),127);
        draft=drafter.forward(residual.data(),input_ids.data(),next_positions.data());passed&=drafter.history_rows()==127;
        bool rejected_discontinuity=false;try{drafter.forward(residual.data(),input_ids.data(),next_positions.data());}catch(const std::runtime_error&){rejected_discontinuity=true;}passed&=rejected_discontinuity;
        const auto stage_before=drafter.stage_metrics();
        for(int r=1;r<=16;++r){next_positions={127+r*n,128+r*n};draft=drafter.forward(residual.data(),input_ids.data(),next_positions.data());}
        const auto stage_after=drafter.stage_metrics();
        std::sort(samples.begin(),samples.end());if(n>1)out<<',';out<<"{\"positions\":"<<n<<",\"draft_token_ids\":[";for(int t=0;t<n;++t){if(t)out<<',';out<<draft[t];passed&=draft[t]>=0&&draft[t]<129280;}
        out<<"],\"median_ms\":"<<samples[2]<<",\"steady_input_ms_per_call\":"<<(stage_after.host_ms[0]-stage_before.host_ms[0])/16
           <<",\"steady_input_sub_ms_per_call\":[";
        for(size_t part=0;part<8;++part){if(part)out<<',';out<<(stage_after.input_sub_ms[part]-stage_before.input_sub_ms[part])/16;}
        out<<"]}";out.flush();}
    out<<"],\"unified_native_cache\":";passed&=mtp_native_probe(argv[5],mtp,backend.get(),shared.embedding,shared.head,residual.data(),input_ids.data(),out);
    out<<",\"background_staging\":";passed&=mtp_stage_probe(mtp,backend.get(),out);
    out<<",\"gpu_front_hc_own_head_finite\":"<<(finite?"true":"false")<<",\"actual_draft_logits_executed\":true,\"bootstrap_input_is_operator_fixture\":true,\"persistent_mtp_kv_integrated\":true,\"mtp_acceptance_measured\":false,\"passed\":"<<(passed?"true":"false")<<"}\n";
    if(!out)throw std::runtime_error("MTP report write failed");std::cout<<(passed?"passed":"failed")<<'\n';return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
