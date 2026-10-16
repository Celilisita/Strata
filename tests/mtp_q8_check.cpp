#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_mtp_q8.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::kernels;
static void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
struct Device {
    void* data = nullptr;
    explicit Device(size_t size) { check(cudaMalloc(&data, size)); }
    ~Device() { cudaFree(data); }
    template<class T> T* as() { return static_cast<T*>(data); }
};
int main(int argc, char** argv) {
    try {
        if (argc != 2) return 2;
        strata::GgufFile source(argv[1]);
        constexpr int H = 2560, FF = 640, NT = 3, K = 3, HIT = NT * K;
        const size_t matrix = size_t(FF) * (H / 32) * 34, blob = 3 * matrix;
        const strata::TensorInfo* tensors[3]{};
        const char* roles[] = {"gate", "up", "down"};
        for (const auto& tensor : source.tensors()) for (int r = 0; r < 3; ++r)
            if (tensor.name == std::string("blk.48.ffn_") + roles[r] + "_exps.weight") tensors[r] = &tensor;
        for (int r = 0; r < 3; ++r) {
            if (!tensors[r] || tensors[r]->type != 8 || tensors[r]->shape.size() != 3 || tensors[r]->shape[2] != 512 ||
                tensors[r]->shape[0] != uint64_t(r == 2 ? FF : H) || tensors[r]->shape[1] != uint64_t(r == 2 ? H : FF))
                throw std::runtime_error("real Q8_0 tensor contract differs");
        }
        Device experts(blob * 512), x(NT * H * sizeof(float)), xq(NT * (H / 32) * 36);
        Device refs(HIT * sizeof(NativeMtpQ8Ref)), ids(HIT * sizeof(int32_t));
        Device scratch(native_mtp_q8_scratch_bytes(HIT, FF)), got(HIT * H * sizeof(float));
        Device gate(FF * sizeof(float)), up(FF * sizeof(float)), hidden((FF / 32) * 36), expected(HIT * H * sizeof(float));
        const int chosen[] = {0, 7, 511};
        std::vector<uint8_t> host(blob), back(blob);
        for (int expert : chosen) {
            for (int r = 0; r < 3; ++r) std::memcpy(host.data() + r * matrix, source.tensor_data(*tensors[r]) + size_t(expert) * matrix, matrix);
            check(cudaMemcpy(experts.as<uint8_t>() + size_t(expert) * blob, host.data(), blob, cudaMemcpyHostToDevice));
            check(cudaMemcpy(back.data(), experts.as<uint8_t>() + size_t(expert) * blob, blob, cudaMemcpyDeviceToHost));
            if (host != back) throw std::runtime_error("real GPU expert payload differs bitwise");
        }
        std::vector<float> input(NT * H);
        std::mt19937 rng(59); std::normal_distribution<float> normal(0.f, 1.f);
        for (auto& value : input) value = normal(rng);
        int32_t route[HIT]; for (int i = 0; i < HIT; ++i) route[i] = chosen[i % K];
        check(cudaMemcpy(x.data, input.data(), input.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(ids.data, route, sizeof(route), cudaMemcpyHostToDevice));
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        native_quantize_q8_1(x.as<float>(), xq.data, H, NT, stream);
        native_mtp_q8_refs(ids.as<int32_t>(), HIT, K, experts.as<uint8_t>(), blob, refs.as<NativeMtpQ8Ref>(), stream);
        check(cudaStreamSynchronize(stream));
        NativeMtpQ8Ref host_refs[HIT]; check(cudaMemcpy(host_refs, refs.data, sizeof(host_refs), cudaMemcpyDeviceToHost));
        for (int i = 0; i < HIT; ++i)
            if (host_refs[i].blob != reinterpret_cast<unsigned long long>(experts.as<uint8_t>() + size_t(route[i]) * blob) ||
                host_refs[i].dst != i || host_refs[i].token != i / K) throw std::runtime_error("route reference key or row differs");
        auto grouped = [&] { native_mtp_q8(refs.as<NativeMtpQ8Ref>(), HIT, H, FF, xq.data, scratch.data, got.as<float>(), stream); };
        auto canonical = [&] {
            for (int i = 0; i < HIT; ++i) {
                auto* weights = experts.as<uint8_t>() + size_t(route[i]) * blob;
                auto* activation = xq.as<uint8_t>() + size_t(i / K) * (H / 32) * 36;
                native_mmvq(8, weights, activation, gate.as<float>(), H, FF, 1, stream);
                native_mmvq(8, weights + matrix, activation, up.as<float>(), H, FF, 1, stream);
                native_swiglu_quantize_q8_1(gate.as<float>(), up.as<float>(), hidden.data, FF, 1, stream);
                native_mmvq(8, weights + 2 * matrix, hidden.data, expected.as<float>() + size_t(i) * H, FF, H, 1, stream);
            }
        };
        grouped(); canonical(); check(cudaStreamSynchronize(stream));
        std::vector<float> a(HIT * H), b(a.size());
        check(cudaMemcpy(a.data(), got.data, a.size() * 4, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), expected.data, b.size() * 4, cudaMemcpyDeviceToHost));
        double difference = 0, magnitude = 0, maximum = 0, reference_max = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("nonfinite expert output");
            const double error = std::fabs(double(a[i]) - b[i]);
            difference += error; magnitude += std::fabs(b[i]); maximum = std::max(maximum, error); reference_max = std::max(reference_max, std::fabs(double(b[i])));
        }
        auto timing = [&](auto operation) {
            for (int i = 0; i < 3; ++i) operation();
            cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
            check(cudaEventRecord(begin, stream)); for (int i = 0; i < 50; ++i) operation();
            check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end)); float ms; check(cudaEventElapsedTime(&ms, begin, end));
            cudaEventDestroy(begin); cudaEventDestroy(end); return double(ms) * 1000 / 50;
        };
        const double grouped_us = timing(grouped), canonical_us = timing(canonical);
        const double relative = difference / std::max(magnitude, 1e-30), max_relative = maximum / std::max(reference_max, 1e-30);
        const bool pass = relative < 1e-4 && max_relative < 5e-4;
        std::printf("{\"pass\":%s,\"experts\":[0,7,511],\"tokens\":3,\"routes\":9,\"payload_exact\":true,\"refs_exact\":true,\"L1_relative\":%.9g,\"max_relative\":%.9g,\"grouped_us\":%.3f,\"canonical_sequence_us\":%.3f,\"timing_scope\":\"operator-only; not an end-to-end baseline\"}\n",
                    pass ? "true" : "false", relative, max_relative, grouped_us, canonical_us);
        check(cudaStreamDestroy(stream)); return pass ? 0 : 1;
    } catch (const std::exception& error) { std::fprintf(stderr, "Q8 MTP check: %s\n", error.what()); return 1; }
}
