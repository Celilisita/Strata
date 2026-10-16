#pragma once
struct ggml_backend_cuda_context;
struct ggml_tensor;
bool strata_cache_lease_lookup_cuda(ggml_backend_cuda_context&, ggml_tensor*);
bool strata_cache_lease_guard_cuda(ggml_backend_cuda_context&, ggml_tensor*);
