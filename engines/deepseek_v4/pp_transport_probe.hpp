#pragma once
#include "pp_transport.hpp"
#include "pp_stream_probe_graph.hpp"
#include "pp_stream_metrics.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <ostream>
#include <string>

namespace strata::deepseek_v4 {
struct PpTransportProbeBinding {
    ggml_backend_t backend;
    ExpertSourcePin& pin;
    std::array<ggml_tensor*,3> sources;
    PpTransportProbeBinding(ggml_backend_t b, ExpertSourcePin& p, const std::array<ggml_tensor*,3>& s)
        : backend(b), pin(p), sources(s) {
        bind_pp_transport(backend, [](const void* input, size_t n, void* data) {
            auto& self = *static_cast<PpTransportProbeBinding*>(data);
            const auto address = reinterpret_cast<uintptr_t>(input);
            for (const auto* tensor : self.sources) {
                const auto start = reinterpret_cast<uintptr_t>(tensor->data);
                if (address >= start && address - start < ggml_nbytes(tensor) &&
                    n <= ggml_nbytes(tensor) - (address - start)) return self.pin.segment(input, n);
            }
            return ExpertSourcePin::Segment{0, false};
        }, this);
    }
    ~PpTransportProbeBinding() { bind_pp_transport(backend, nullptr, nullptr); }
};
inline void pp_probe_ranges_mode(const char* mode, const char* name="STRATA_DSV4_PP_SOURCE_RANGES") {
#ifdef _WIN32
    if (_putenv_s(name, mode)) throw std::runtime_error("PP range mode update failed");
#else
    if (setenv(name, mode, 1)) throw std::runtime_error("PP range mode update failed");
#endif
}
inline bool pp_transport_operator_probe(PpProbeGraph& graph, std::ostream& out, bool fixed_edges=false) {
    const char* flag=fixed_edges?"STRATA_DSV4_PP_FIXED_EDGES":"STRATA_DSV4_PP_SOURCE_RANGES";
    const char* previous = std::getenv(flag);
    const std::string restore = previous ? previous : "1";
    std::array<std::array<double,3>,2> times{};
    std::array<uint64_t,2> payload{},staged{},copies{},waits{};
    for (int mode = 0; mode < 2; ++mode) {
        pp_probe_ranges_mode(mode ? "1" : "0",flag);
        const auto before = pp_stream_backend_metrics(graph.backend);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute_async(graph.backend, graph.graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("PP source range execution failed");
            ggml_backend_synchronize(graph.backend);
            times[mode][repeat] = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        const auto after = pp_stream_backend_metrics(graph.backend);
        payload[mode] = after.payload_bytes-before.payload_bytes;
        staged[mode] = after.staged_bytes-before.staged_bytes;
        copies[mode] = after.source_cpu_copy_ns-before.source_cpu_copy_ns;
        waits[mode] = after.staging_wait_ns-before.staging_wait_ns;
    }
    pp_probe_ranges_mode(restore.c_str(),flag);
    auto a=times[0],b=times[1];std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    const bool passed=payload[0]==payload[1]&&(fixed_edges?staged[1]==staged[0]&&waits[1]==0:staged[1]<staged[0])&&b[1]<=a[1]*1.05;
    out<<"{\"control_ms\":["<<times[0][0]<<','<<times[0][1]<<','<<times[0][2]
       <<"],\"ranges_ms\":["<<times[1][0]<<','<<times[1][1]<<','<<times[1][2]
       <<"],\"speedup\":"<<a[1]/b[1]<<",\"payload_bytes\":["<<payload[0]<<','<<payload[1]
       <<"],\"staged_bytes\":["<<staged[0]<<','<<staged[1]
       <<"],\"staging_wait_ns\":["<<waits[0]<<','<<waits[1]
       <<"],\"cpu_copy_ns\":["<<copies[0]<<','<<copies[1]<<"],\"passed\":"<<(passed?"true":"false")<<'}';
    return passed;
}
}
