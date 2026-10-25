// Included inside strata::deepseek_v4 after the original serial service methods.
void CpuExpertService::prepare_partial_down_diagnostic(int layer, size_t gu_routes, size_t down_routes) {
    if (impl_->model.arch != LLM_ARCH_GLM5_NEXT || impl_->threads != 16 || layer != 24 ||
        gu_routes != 10 || down_routes != 6 || size_t(layer) >= impl_->graphs.size() ||
        impl_->graphs[layer].empty() || impl_->runtime_gu_q8_requested || impl_->diagnostic_gu_q8 ||
        impl_->diagnostic_derived || impl_->diagnostic_iq3_r4 || impl_->diagnostic_exact_lut)
        throw std::runtime_error("partial Down preparation requires original L24 native16 GU10/Down6");
    if (!impl_->partial_down_graph) {
        if (impl_->row_activation.size() < gu_routes*4096) impl_->row_activation.resize(gu_routes*4096);
        impl_->partial_down_graph = std::make_unique<Graph>(impl_->model, impl_->cpu,
            impl_->threads, layer, uint32_t(gu_routes), true, false, false, down_routes);
    }
}

bool CpuExpertService::partial_down_preflight(const PartialDownJob& job) const noexcept {
    if (!impl_ || impl_->threads != 16 || impl_->model.arch != LLM_ARCH_GLM5_NEXT ||
        !partial_down_job_shape(job) || !impl_->partial_down_graph ||
        !impl_->partial_down_graph->partial_down ||
        impl_->partial_down_graph->partial_down->evaluation_active ||
        impl_->row_activation.size() < 10*4096 || impl_->runtime_gu_q8_requested ||
        impl_->diagnostic_gu_q8 || impl_->diagnostic_derived ||
        impl_->diagnostic_iq3_r4 || impl_->diagnostic_exact_lut) return false;
    constexpr size_t output_bytes = 6*4096*sizeof(float);
    if (!partial_down_disjoint(job.output, output_bytes, job.activations, 2*4096*sizeof(float)) ||
        !partial_down_disjoint(job.output, output_bytes, job.ids, 10*sizeof(int32_t)) ||
        !partial_down_disjoint(job.output, output_bytes, job.positions, 10) ||
        !partial_down_disjoint(job.output, output_bytes, job.original_ranks, 10) ||
        !partial_down_disjoint(job.output, output_bytes, job.down_rows, 6)) return false;
    const auto& graph = *impl_->partial_down_graph;
    const auto& partial = *graph.partial_down;
    for (const auto* tensor : {graph.activation, graph.ids, partial.mid, partial.compact_mid,
                              partial.gather_indices, partial.down_ids, partial.output})
        if (!tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) ||
            !partial_down_disjoint(job.output, output_bytes, tensor->data, ggml_nbytes(tensor))) return false;
    return true;
}

PartialDownReceipt CpuExpertService::compute_partial_down_diagnostic(const PartialDownJob& job) {
    // This method is entered only after the caller commits partialdecision. A
    // later guard/callback/compute failure must go to that ticket's error/drain.
    if (!job.partial_decision_committed || !partial_down_preflight(job))
        throw std::runtime_error("committed partial Down job differs from its prepared preflight contract");
    auto& graph = *impl_->partial_down_graph;
    auto& partial = *graph.partial_down;
    std::array<int32_t,6> gather_rows{}, down_keys{};
    for (size_t row = 0; row < job.gu_routes; ++row)
        std::memcpy(impl_->row_activation.data()+row*4096,
            job.activations+size_t(job.positions[row])*4096, 4096*sizeof(float));
    for (size_t row = 0; row < job.down_routes; ++row) {
        gather_rows[row] = job.down_rows[row];
        down_keys[row] = job.ids[job.down_rows[row]];
    }
    ggml_backend_tensor_set(graph.activation, impl_->row_activation.data(), 0, ggml_nbytes(graph.activation));
    ggml_backend_tensor_set(graph.ids, job.ids, 0, ggml_nbytes(graph.ids));
    ggml_backend_tensor_set(partial.gather_indices, gather_rows.data(), 0, ggml_nbytes(partial.gather_indices));
    ggml_backend_tensor_set(partial.down_ids, down_keys.data(), 0, ggml_nbytes(partial.down_ids));
    CpuPartialDownGraph::Evaluation evaluation;
    partial.begin(graph.scheduler, evaluation, job);
    ggml_status status = GGML_STATUS_FAILED;
    try {
        strata::glm5_next::with_cpu_caller(impl_->threads, [&] {
            status = ggml_backend_sched_graph_compute(graph.scheduler, graph.context->gf);
            ggml_backend_sched_synchronize(graph.scheduler);
        });
    } catch (...) {
        // Callback state remains alive until the original backend is quiescent.
        ggml_backend_sched_synchronize(graph.scheduler);
        partial.end(graph.scheduler);
        throw;
    }
    partial.end(graph.scheduler);
    if (evaluation.error) std::rethrow_exception(evaluation.error);
    if (status != GGML_STATUS_SUCCESS || evaluation.publication_callbacks != 1 || !evaluation.published)
        throw std::runtime_error("partial Down failed or did not publish its one canonical GU boundary");
    ggml_backend_tensor_get(graph.down, job.output, 0, ggml_nbytes(graph.down));
    return {job.gu_routes, job.down_routes, 2048, evaluation.publication_callbacks, evaluation.published};
}

PartialDownCapacity CpuExpertService::partial_down_capacity() const {
    PartialDownCapacity out;
    if (!impl_->partial_down_graph) return out;
    const auto& graph = *impl_->partial_down_graph;
    const auto& partial = *graph.partial_down;
    out.prepared_graphs = 1;
    out.graph_nodes = size_t(ggml_graph_n_nodes(graph.context->gf));
    for (auto* t = ggml_get_first_tensor(graph.context->ctx0); t;
         t = ggml_get_next_tensor(graph.context->ctx0, t)) ++out.graph_tensors;
    out.graph_context_used_bytes = ggml_used_mem(graph.context->ctx0);
    out.scheduler_cpu_buffer_bytes = ggml_backend_sched_get_buffer_size(graph.scheduler, impl_->cpu);
    out.activation_capacity_bytes = impl_->row_activation.capacity()*sizeof(float);
    out.original_mid_bytes = ggml_nbytes(partial.mid);
    out.compact_mid_bytes = ggml_nbytes(partial.compact_mid);
    out.output_bytes = ggml_nbytes(partial.output);
    out.gather_index_bytes = ggml_nbytes(partial.gather_indices);
    out.down_id_bytes = ggml_nbytes(partial.down_ids);
    return out;
}
