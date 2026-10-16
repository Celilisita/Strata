// GPU row mask uses the common Q8_1 payload and finite-value contract.
void native_quantize_q8_1(const float* x, void* output, int width, int columns, void* stream,
                          const unsigned long long* addresses) {
    if (!addresses) { native_quantize_q8_1(x, output, width, columns, stream); return; }
    validate_shape(width, columns); validate_pointer(x); validate_pointer(output); validate_stream(stream);
    const int count = width * columns;
    const unsigned blocks = unsigned((size_t(count) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_quantize_q8_1_kernel<true>, dim3(blocks), dim3(QUANT_THREADS), 0,
        static_cast<cudaStream_t>(stream), x, static_cast<Q81Block*>(output), count, width, addresses);
    launch_check();
}
