#pragma once
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <csignal>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace strata::deepseek_v4 {
inline void benchmark_fault_guard() {
#ifdef _WIN32
    _set_abort_behavior(0, _CALL_REPORTFAULT);
    // CUDA DLL shutdown can spin while an aborted graph still names mapped
    // host tickets. Preserve the exact fatal text and terminate this benchmark
    // process directly, avoiding both WER dumps and loader-detach deadlocks.
    ggml_set_abort_callback([](const char* message){std::fprintf(stderr,"%s\n",message);std::fflush(stderr);TerminateProcess(GetCurrentProcess(),3);});
    std::signal(SIGABRT,[](int){std::fflush(stderr);TerminateProcess(GetCurrentProcess(),3);});
#endif
}
}
