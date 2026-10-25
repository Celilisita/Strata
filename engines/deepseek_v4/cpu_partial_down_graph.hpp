#pragma once

#include "cpu_partial_down.hpp"
#include "ggml-backend.h"
#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace strata::deepseek_v4 {

// Only the captured L24/50762 CPU route domain is admitted by this prototype.
// These are occurrence indices; repeated keys retain their actual position row.
inline bool partial_down_job_shape(const PartialDownJob& job) noexcept {
    constexpr int32_t ids[] = {57,234,27,106,256,234,129,27,271,57};
    constexpr uint8_t positions[] = {0,0,0,0,0,1,1,1,1,1};
    constexpr uint8_t ranks[] = {0,1,2,3,4,8,11,12,14,15};
    constexpr uint8_t remaining[] = {2,3,4,6,7,8};
    if (job.layer != 24 || job.positions_count != 2 || job.gu_routes != 10 ||
        job.down_routes != 6 || !job.activations || !job.ids || !job.positions ||
        !job.original_ranks || !job.down_rows || !job.output || !job.publish_mid ||
        job.output_elements < 6*4096) return false;
    return std::memcmp(job.ids, ids, sizeof(ids)) == 0 &&
        std::memcmp(job.positions, positions, sizeof(positions)) == 0 &&
        std::memcmp(job.original_ranks, ranks, sizeof(ranks)) == 0 &&
        std::memcmp(job.down_rows, remaining, sizeof(remaining)) == 0;
}

inline bool partial_down_disjoint(const void* a, std::size_t an,
                                   const void* b, std::size_t bn) noexcept {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    if (!a || !b || an > std::numeric_limits<uintptr_t>::max()-x ||
        bn > std::numeric_limits<uintptr_t>::max()-y) return false;
    return x+an <= y || y+bn <= x;
}

class CpuPartialDownGraph {
public:
    ggml_tensor *mid = nullptr, *gather_indices = nullptr, *down_ids = nullptr;
    ggml_tensor *compact_mid = nullptr, *output = nullptr;
    bool evaluation_active = false;

    CpuPartialDownGraph(ggml_context* ctx, ggml_cgraph* graph, ggml_tensor* original_down,
                        ggml_tensor* original_ids, ggml_tensor* bank,
                        std::size_t gu_routes, std::size_t down_routes) {
        mid = original_down ? original_down->src[1] : nullptr;
        if (gu_routes != 10 || down_routes != 6 || !original_down ||
            original_down->op != GGML_OP_MUL_MAT_ID || original_down->src[0] != bank ||
            original_down->src[2] != original_ids || !bank || bank->type != GGML_TYPE_IQ4_XS ||
            bank->ne[0] != 2048 || bank->ne[1] != 4096 || bank->ne[2] != 288 || bank->ne[3] != 1 ||
            !mid || mid->op != GGML_OP_GLU || mid->type != GGML_TYPE_F32 ||
            (ggml_get_glu_op(mid) != GGML_GLU_OP_SWIGLU &&
             ggml_get_glu_op(mid) != GGML_GLU_OP_SWIGLU_CLAMP) ||
            mid->ne[0] != 2048 || mid->ne[1] != 1 || mid->ne[2] != 10 || mid->ne[3] != 1 ||
            !ggml_is_contiguous(mid))
            throw std::runtime_error("partial Down requires the original contiguous canonical GU10 boundary");

        gather_indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 6);
        down_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 6);
        ggml_set_input(gather_indices);
        ggml_set_input(down_ids);
        auto* rows = ggml_reshape_2d(ctx, mid, 2048, 10);
        auto* gathered = ggml_get_rows(ctx, rows, gather_indices);
        compact_mid = ggml_reshape_3d(ctx, gathered, 2048, 1, 6);
        output = ggml_mul_mat_id(ctx, bank, compact_mid, down_ids);
        // Reuse the original raw Down op parameters, bank and canonical CPU leaf.
        std::memcpy(output->op_params, original_down->op_params, sizeof(output->op_params));
        ggml_set_name(output, "partial-down-original-six");
        ggml_set_output(output);
        ggml_graph_clear(graph);
        ggml_build_forward_expand(graph, output);
        bool saw_mid = false;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto* node = ggml_graph_node(graph, i);
            saw_mid |= node == mid;
            if (node == original_down)
                throw std::runtime_error("partial Down still reaches the original ten-route Down");
        }
        if (!saw_mid || !ggml_is_contiguous(output) || ggml_nbytes(output) != 6*4096*sizeof(float))
            throw std::runtime_error("partial Down graph boundary or six-route output differs");
    }

    struct Evaluation {
        const PartialDownJob* job = nullptr;
        CpuPartialDownGraph* graph = nullptr;
        uint32_t publication_callbacks = 0;
        bool published = false;
        std::exception_ptr error;
    };

    void begin(ggml_backend_sched_t scheduler, Evaluation& evaluation, const PartialDownJob& job) {
        if (evaluation_active) throw std::runtime_error("partial Down graph is already computing");
        evaluation.job = &job;
        evaluation.graph = this;
        evaluation_active = true;
        // This private Graph owns the sole scheduler callback slot. No other
        // callback is installed on it; begin/end require serial service ownership.
        ggml_backend_sched_set_eval_callback(scheduler, observe, &evaluation);
    }

    void end(ggml_backend_sched_t scheduler) noexcept {
        ggml_backend_sched_set_eval_callback(scheduler, nullptr, nullptr);
        evaluation_active = false;
    }

private:
    static bool observe(ggml_tensor* tensor, bool ask, void* opaque) noexcept {
        auto& state = *static_cast<Evaluation*>(opaque);
        if (ask) return tensor == state.graph->mid;
        try {
            if (tensor != state.graph->mid || ++state.publication_callbacks != 1 ||
                !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer))
                throw std::runtime_error("partial Down canonical mid callback differs");
            // Existing scheduler has completed and synchronized the CPU prefix.
            // The callback copies selected rows before returning; no tensor or
            // packet pointer is retained by this adapter after compute returns.
            if (!state.job->publish_mid(state.job->publication_context,
                    static_cast<const float*>(tensor->data), 10, 2048))
                throw std::runtime_error("partial Down mid publication was rejected");
            state.published = true;
            return true;
        } catch (...) {
            state.error = std::current_exception();
            return false;
        }
    }
};

} // namespace strata::deepseek_v4
