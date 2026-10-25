#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif
namespace strata::core::native_miss_pickup {
inline bool enabled() {
    const char* value=std::getenv("STRATA_NATIVE_MISS_PICKUP_SPIN");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::invalid_argument("STRATA_NATIVE_MISS_PICKUP_SPIN must be0 or1");
}
struct NativeIdle {
    static void pause() noexcept {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
        _mm_pause();
#elif defined(_M_ARM64)
        __yield();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield");
#else
        std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
    }
    static void yield() {std::this_thread::yield();}
};
// Only replaces the no-request idle action. Caller still checks stop and
// acquire-loads request on EVERY poll. No clock read on the default path.
template<class Clock=std::chrono::steady_clock,class Idle=NativeIdle>
class IdleWait {
    bool enabled_,started_=false;
    uint32_t polls_=0;
    typename Clock::time_point since_{};
public:
    explicit IdleWait(bool enabled):enabled_(enabled) {}
    void observed() noexcept {started_=false;polls_=0;}
    void idle() {
        if(!enabled_) {Idle::yield();return;}
        if(!started_) {since_=Clock::now();started_=true;polls_=0;}
        Idle::pause();
        if((++polls_&63u)!=0)return;
        if(Clock::now()-since_<std::chrono::milliseconds(5))return;
        Idle::yield();started_=false;polls_=0;
    }
};
} // namespace strata::core::native_miss_pickup
