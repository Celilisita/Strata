// Candidate-only CUDA shared-memory preference, scoped to the worker/device.
template<bool Prepared>
void configure_attention_shared() {
    int device = -1;
    const auto identity = cudaGetDevice(&device);
    if (identity != cudaSuccess) throw std::runtime_error(cudaGetErrorString(identity));
    static thread_local int configured_device = -1;
    if (configured_device == device) return;
    cudaError_t configured;
    if constexpr (Prepared)
        configured = cudaFuncSetAttribute(tile<true, 2, false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    else
        configured = cudaFuncSetAttribute(tile<true, 2, false, false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (configured != cudaSuccess) throw std::runtime_error(cudaGetErrorString(configured));
    configured_device = device;
}
