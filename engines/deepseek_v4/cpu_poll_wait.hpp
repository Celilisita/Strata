#pragma once
#include <thread>
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
namespace strata::deepseek_v4 {
inline void cpu_poll_pause(){
#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
}
