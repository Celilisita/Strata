#pragma once
#include "../deepseek_v4/cpu_expert_service.hpp"
#include "cpu_gu_q8_metrics.hpp"
#include "../deepseek_v4/device_cache_protocol.hpp"
#include <array>
#include <string>
#include <vector>
struct llama_model;
namespace strata::glm5_next {
class HybridProvider {
public:
    HybridProvider(const llama_model&,const std::string& curve_path,const char* gpu_description,bool four_positions=false);
    int threads()const{return threads_;}
    // Explicit diagnostic preparation is quiescent; compute remains on the serial miss worker.
    deepseek_v4::CpuExpertService& partial_down_diagnostic_service() noexcept{return service_;}
    const auto& pair_jobs()const{return pair_jobs_;}
    const auto& pair_ns()const{return pair_ns_;}
    bool four_positions_prepared()const{return four_positions_;}
    const auto& four_jobs()const{return four_jobs_;}
    const auto& four_ns()const{return four_ns_;}
    const auto& four_capacity_before()const{return four_before_;}
    const auto& four_capacity_after()const{return four_after_;}
    double four_prepare_ms()const{return four_prepare_ms_;}
    CpuGuQ8PreparationMetrics gu_q8_preparation()const {
        auto out=gu_q8_preparation_;out.runtime_active=service_.gu_q8_reuse_runtime_active();return out;
    }
    bool gu_q8_reuse_requested()const noexcept{return service_.gu_q8_reuse_runtime_requested();}
    bool gu_q8_reuse_active()const noexcept{return service_.gu_q8_reuse_runtime_active();}
    bool set_gu_q8_reuse_active(bool enabled)noexcept{return service_.set_gu_q8_reuse_runtime_active(enabled);}
    bool prefer_cpu(int layer,int missing)const;
    bool prefer_cpu_pair(int layer,int missing,int unique)const;
    bool cpu_only_layer(int layer)const;
    uint64_t compute_pair(int layer,const int32_t*,const uint8_t*,int,const float*,float*);
    uint64_t compute_four(int layer,const int32_t*,const uint8_t*,int,const float*,float*);
    uint64_t compute(int layer,const int32_t* ids,int count,const float* activation,float* output);
private:
    const llama_model& model_;
    const int threads_;
    deepseek_v4::CpuExpertService service_;
    const bool four_positions_;
    std::array<uint64_t,17*17> pair_jobs_{},pair_ns_{};
    std::array<uint64_t,33*33> four_jobs_{},four_ns_{};
    deepseek_v4::CpuExpertService::RoutedRowsCapacity four_before_{},four_after_{};
    double four_prepare_ms_=0;
    CpuGuQ8PreparationMetrics gu_q8_preparation_;
    struct Curve {std::array<double,3> cpu{},cpu_control{},gpu{};bool ready=false,wide=false;};
    std::vector<Curve> curves_;
    struct PairCurve {int count=0,unique=0;double cpu=0,cpu_control=0,gpu=0,upload=0;bool wide=false;};
    std::vector<std::vector<PairCurve>> pair_curves_;
};
bool hybrid_requested();
}
