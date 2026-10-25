#pragma once
#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/nvfp4.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "strata/qwen35/spec.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace strata::qwen35::native {
inline bool q36_startup_body_plan_requested(const std::string &variant,
                                           const std::string &kv) {
    const char *control = std::getenv("STRATA_DENSE_Q36_STARTUP_BODY_PLAN");
    const std::string mode = control ? control : "0";
    if (mode != "0" && mode != "1")
        throw std::invalid_argument("Q36 startup body plan must be0or1");
    if (mode == "0") return false;
    if (variant != "qwen36_27b" || kv != "ram8k-reuse")
        throw std::invalid_argument("Q36 startup body plan requires its RAM profile");
    const char *rows = std::getenv("STRATA_DENSE_RAM_GPU_ROWS");
    if (!rows || std::string(rows) != "40960")
        throw std::invalid_argument("Q36 startup body plan requires RAM40960");
    return true;
}

inline bool q36_plan_enabled_profile_flag(const char *name) {
    const char *value = std::getenv(name);
    return !value || std::string(value) == "1";
}

// This is a cold, launch-free shape-plan experiment. It never admits Q8 eager
// readiness, reinterprets weight values, changes a request profile, or captures
// a Graph. Actual execution still chooses its existing route in PREPARE.
template<class Inventory, class IsHead>
void prepare_q36_body_8192_plans(Nvfp4Gemm &gemm, Nvfp4Input input,
                                const Inventory &inventory, IsHead is_head,
                                const strata::qwen35::Spec &spec,
                                cudaStream_t stream, bool fused,
                                bool gu_available) {
    const auto total_start = std::chrono::steady_clock::now();
    auto require = [](bool ok, const char *message) {
        if (!ok) throw std::invalid_argument(message);
    };
    require(spec.n_layers == 64 && spec.layers.size() == 64 &&
            spec.n_embd == 5120 && spec.n_ff == 17408 &&
            spec.n_vocab == 248320 && spec.n_mtp_layers == 1 &&
            spec.context_length == 262144,
            "Q36 startup body plan actual model geometry");
    require(input.capacity_rows == 8192 && input.capacity_columns >= 17408 &&
            input.values && input.scales,
            "Q36 startup body plan existing encoded arena");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Q36 body plan device");
    const auto caps = capabilities(device);
    require(caps.major == 12 && caps.minor == 0,
            "Q36 startup body plan requires SM120");
    cudaStreamCaptureStatus capture{};
    check_cuda(cudaStreamIsCapturing(stream, &capture), "Q36 body plan capture lease");
    require(capture == cudaStreamCaptureStatusNone,
            "Q36 startup body planning must precede Graph capture");
    const char *half = std::getenv("STRATA_NVFP4_HALF_FFN");
    require(fused && gu_available &&
            (!half || std::string(half) == "1" || std::string(half) == "resident") &&
            q36_plan_enabled_profile_flag("STRATA_NVFP4_GU_TMA") &&
            q36_plan_enabled_profile_flag("STRATA_NVFP4_GU_MERGED"),
            "Q36 startup body plan requires existing fused GU profile");

    using Shape = std::pair<int, int>; // rows, columns; token extent is8192.
    const std::map<Shape, size_t> expected = {
        {{1024, 5120}, 32}, {{5120, 6144}, 64}, {{5120, 17408}, 64},
        {{6144, 5120}, 48}, {{10240, 5120}, 48}, {{12288, 5120}, 16}
    };
    std::map<Shape, size_t> actual;
    std::map<Shape, Nvfp4View> plans;
    size_t skipped_head = 0, skipped_gu = 0, selected = 0;
    for (const auto &item : inventory) {
        const auto &entry = item.second;
        const auto view = entry.view;
        require(view.values && view.scales && view.weight_scale > 0 &&
                view.input_scale > 0 && std::isfinite(view.weight_scale) &&
                std::isfinite(view.input_scale), "Q36 body plan actual weight view");
        if (is_head(view)) {
            require(view.rows == 248320 && view.columns == 5120,
                    "Q36 body plan external shared head geometry");
            ++skipped_head;
            continue;
        }
        if (view.rows == 17408 && view.columns == 5120) {
            require(entry.packed && entry.declared_packed,
                    "Q36 body plan fused GU physical inventory");
            ++skipped_gu;
            continue;
        }
        const Shape shape{view.rows, view.columns};
        require(expected.count(shape) == 1,
                "Q36 body plan unexpected body projection geometry");
        ++actual[shape];
        ++selected;
        // Packing changes value nibbles only. prepare() reads geometry and
        // scale pointers, never values; do not call canonical()/unpack here.
        plans.emplace(shape, view);
    }
    require(inventory.size() == 401 && skipped_head == 1 && skipped_gu == 128 &&
            selected == 272 && actual == expected,
            "Q36 body plan complete400 body plus external head inventory");

    size_t free_before = 0, free_after = 0, total = 0;
    const size_t reserve = std::max(DeviceStorage::reserve_bytes(),
                                    size_t(4) * 1024 * 1024 * 1024);
    check_cuda(cudaMemGetInfo(&free_before, &total), "Q36 body plan initial memory");
    require(free_before >= reserve, "Q36 body planning lacks initial GPU reserve");
    const auto start = std::chrono::steady_clock::now();
    for (const auto &item : plans) gemm.prepare(item.second, input, 8192);
    const double planning_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    check_cuda(cudaMemGetInfo(&free_after, &total), "Q36 body plan final memory");
    const double startup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_start).count();
    std::cerr << "DENSE_Q36_STARTUP_BODY_PLAN enabled=1 tokens=8192"
              << " selected_body=" << selected << " skipped_head=" << skipped_head
              << " skipped_gu=" << skipped_gu
              << " unique_geometry_requests=" << plans.size()
              << " prepare_calls=" << plans.size()
              << " existing_tiny_only_by_bound_ctor=1"
              << " planning_ms=" << planning_ms << " startup_ms=" << startup_ms
              << " free_before=" << free_before
              << " free_after=" << free_after << " reserve_bytes=" << reserve
              << " reserve_admitted=" << int(free_after >= reserve)
              << " project_calls=0 graph_nodes=0 prefix_tokens=0\n";
    require(free_after >= reserve, "Q36 body planning lacks final GPU reserve");
}

// Private service-to-existing-Runtime hook; no new provider or owner.
void nvfp4_plan_q36_body_8192(DenseGemm &, const strata::qwen35::Spec &, cudaStream_t);
}
