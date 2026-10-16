size_t native_mtp_q8_scratch_bytes(int hits, int n_ff) {
    if (hits < 1 || hits > 65535 || n_ff < 32 || n_ff % 32 || n_ff > INT32_MAX / hits)
        throw std::invalid_argument("invalid Q8 MTP scratch extent");
    return size_t(hits) * (size_t(2) * n_ff * sizeof(float) + size_t(n_ff / 32) * sizeof(Q81Block));
}
void native_mtp_q8_refs(const int32_t* ids, int hits, int k, const uint8_t* experts,
                        size_t blob_bytes, NativeMtpQ8Ref* refs, void* stream) {
    if (hits < 1 || hits > 65535 || k < 1 || hits % k || blob_bytes == 0)
        throw std::invalid_argument("invalid Q8 MTP resident routes");
    validate_pointer(ids); validate_pointer(experts); validate_pointer(refs); validate_stream(stream);
    mtp_q8_refs_kernel<<<(hits + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, hits, k, experts, blob_bytes, refs);
    launch_check();
}
void native_mtp_q8(const NativeMtpQ8Ref* refs, int hits, int n_in, int n_ff,
                    const void* x_q8_1, void* scratch, float* out, void* stream) {
    native_mtp_q8_scratch_bytes(hits, n_ff);
    validate_shape(n_in, 1); validate_pointer(refs); validate_pointer(x_q8_1);
    validate_pointer(scratch); validate_pointer(out); validate_stream(stream);
    auto* gu = static_cast<float*>(scratch);
    auto* hq = reinterpret_cast<Q81Block*>(gu + size_t(2) * hits * n_ff);
    const auto cs = static_cast<cudaStream_t>(stream);
    const size_t matrix_bytes = size_t(n_ff) * (n_in / 32) * sizeof(Q80Block);
    mtp_q8_rows_kernel<2><<<dim3((2 * n_ff + 7) / 8, hits), 256, 0, cs>>>(
        refs, matrix_bytes, static_cast<const Q81Block*>(x_q8_1), gu, n_in, n_ff, hits);
    launch_check();
    // Each row is a multiple of 32; flattening preserves quantization blocks.
    native_swiglu_quantize_q8_1(gu, gu + size_t(hits) * n_ff, hq, hits * n_ff, 1, stream);
    mtp_q8_rows_kernel<1><<<dim3((n_in + 7) / 8, hits), 256, 0, cs>>>(
        refs, 2 * matrix_bytes, hq, out, n_ff, n_in, hits);
    launch_check();
}
