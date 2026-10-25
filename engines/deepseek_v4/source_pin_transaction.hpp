#pragma once
#include "expert_source_pin_backend.hpp"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::deepseek_v4 {
// Startup only, before readers; source inventory <=4096 bounds all API reports.
struct SourcePinTransaction {
    uint64_t sync_checks=0,sync_failures=0,rollback_ranges=0,rollback_bytes=0,api_calls=0,known_oom_clears=0,known_fallback_clears=0;
    uint32_t stop_error=0,final_sync_error=0,final_sync_pending=0;
    const char* stop_reason="none";
    const char* first_failure="none";
    SourcePinApiResult first_result{};
    bool known_oom=false;

    void api(const char* operation,SourcePinApiResult result) {
        ++api_calls;
        if((result.primary||result.pending)&&std::strcmp(first_failure,"none")==0) {
            first_failure=operation;first_result=result;
        }
        if(result.primary||result.pending) {
            std::fprintf(stderr,"strata_source_pin_api op=%s stage=%s primary=%u pending=%u\n",operation,result.stage,result.primary,result.pending);
            std::fflush(stderr);
        }
    }
    void report()const {
        std::fprintf(stderr,"strata_source_pin_channels calls=%llu first_op=%s first_primary=%u first_pending=%u known_oom_clears=%llu known_register_801_clears=%llu final_sync_primary=%u final_sync_pending=%u first_stage=%s\n",
            (unsigned long long)api_calls,first_failure,first_result.primary,first_result.pending,
            (unsigned long long)known_oom_clears,(unsigned long long)known_fallback_clears,final_sync_error,final_sync_pending,first_result.stage);
        std::fflush(stderr);
    }
    template<class Clear> void consume(uint32_t expected,Clear& clear) {
        uint32_t observed=0;const bool cleared=clear(expected,observed);
        std::fprintf(stderr,"strata_source_pin_clear known_error=%u observed=%u accepted=%d\n",expected,observed,int(cleared));
        std::fflush(stderr);
        if(!cleared)throw std::runtime_error("source pin known error consume failed: "+std::to_string(observed));
        if(observed==2)++known_oom_clears;
        if(observed==801)++known_fallback_clears;
    }
    template<class Clear> bool pending(SourcePinApiResult result,Clear& clear) {
        if(!result.pending)return false;
        if(result.pending!=2||!known_oom)throw std::runtime_error(
            "source pin unknown pending CUDA error: "+std::to_string(result.pending));
        consume(2,clear);return true;
    }
    template<class Clear> void allocation(const char* operation,SourcePinApiResult result,Clear& clear) {
        api(operation,result);
        if(std::strcmp(result.stage,"api"))throw std::runtime_error("source pin selected device check failed");
        if(std::strcmp(operation,"register")==0&&result.primary==801&&result.pending==801) {
            consume(801,clear);return; // Exact current NotSupported; next real sync must still pass.
        }
        if(result.primary==2)known_oom=true;
        pending(result,clear);
    }
    template<class Sync> SourcePinApiResult raw_sync(Sync& sync) {
        SourcePinApiResult result{};++sync_checks;sync(result);
        final_sync_error=result.primary;final_sync_pending=result.pending;
        if(result.primary||result.pending)++sync_failures;
        api("sync",result);return result;
    }
    template<class Sync> void begin(Sync& sync) {
        const auto result=raw_sync(sync);
        // A prior pending error remains unconsumed and is always fatal.
        if(result.primary||result.pending)throw std::runtime_error(
            "source pin pre-registration synchronization failed primary="+std::to_string(result.primary)+
            " pending="+std::to_string(result.pending));
    }
    template<class Sync,class Clear> bool check(Sync& sync,Clear& clear,uint32_t& error) {
        auto result=raw_sync(sync);
        auto classify=[&](SourcePinApiResult current) {
            const bool backend_fence=!std::strcmp(current.stage,"cudaDeviceSynchronize")||
                !std::strcmp(current.stage,"cudaStreamSynchronize_main")||!std::strcmp(current.stage,"cudaStreamSynchronize_copy");
            if(std::strcmp(current.stage,"api")&&!backend_fence)
                throw std::runtime_error(std::string("source pin synchronization owner/pending check failed at ")+current.stage);
            if(current.primary&&current.primary!=2)throw std::runtime_error(
                "source pin registration synchronization failed: "+std::to_string(current.primary));
            if(current.primary==2)known_oom=true;
            return pending(current,clear);
        };
        if(classify(result)) {
            // Consume is not recovery. Require a subsequent genuine 0/0 sync.
            result=raw_sync(sync);
            if(classify(result)) {error=2;return false;} // Still unstable: existing owned-tail rollback.
        }
        error=result.primary;return result.primary==0;
    }
    template<class Rollback,class Clear> void discard_tail(Rollback& rollback,Clear& clear) {
        SourcePinApiResult result{};const uint64_t bytes=rollback(result);
        if(bytes||result.primary||result.pending)api("unregister",result);
        if(result.primary)throw std::runtime_error(
            "source pin rollback unregister failed: "+std::to_string(result.primary));
        if(!bytes)throw std::runtime_error("source pin recovery exhausted owned registrations");
        // Callback removed this owner on primary success before any pending-error policy.
        ++rollback_ranges;rollback_bytes+=bytes;
        pending(result,clear);
    }
    template<class Sync,class Rollback,class Clear>
    bool complete(bool registered,uint32_t registration_error,uint32_t alias_error,
                  Sync& sync,Rollback& rollback,Clear& clear) {
        uint32_t error=0;bool stable=check(sync,clear,error);
        if(!stable&&error!=2)throw std::runtime_error(
            "source pin registration synchronization failed: "+std::to_string(error));
        if(!known_oom&&stable)return true;
        stop_reason=registration_error==2?"register_oom":alias_error==2?"alias_oom":"commit_oom";
        stop_error=2;
        if(registered) {discard_tail(rollback,clear);stable=check(sync,clear,error);}
        while(!stable) {
            if(error!=2)throw std::runtime_error(
                "source pin recovery synchronization failed: "+std::to_string(error));
            discard_tail(rollback,clear);stable=check(sync,clear,error);
        }
        return false; // Verified prefix retains the unchanged staging fallback.
    }
};
} // namespace strata::deepseek_v4
