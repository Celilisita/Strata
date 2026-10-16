#pragma once

// Dispatch to another instantiation of the existing canonical MMVQ templates.
// Fused variants and geometry experiments keep their original dispatch.
template <ggml_type type>
static bool strata_mxfp4_load_dispatch(
        const void* vx, const void* vy, const int32_t* ids, ggml_cuda_mm_fusion_args_device fusion, float* dst,
        int columns, int rows, int tokens, uint3 channels_y, uint3 channel_ratio, uint3 sample_ratio,
        int row_x, int col_y, int col_dst, int channel_x, int channel_y, int channel_dst,
        int sample_x, int sample_y, int sample_dst, int ids_stride,
        int channels_dst, int samples_dst, int warp_size, int cc, mmvq_parameter_table_id table, cudaStream_t stream) {
    if constexpr (type != GGML_TYPE_MXFP4) return false;
    else {
        if (!ids || channels_dst != 6 || samples_dst != 1 || (reinterpret_cast<uintptr_t>(vx) & 3) ||
            row_x != columns / 32 || channel_x < row_x * rows || (channel_x & 3) ||
            fusion.gate || fusion.x_bias || fusion.gate_bias ||
            fusion.x_scale || fusion.gate_scale || fusion.shared_up ||
            !strata::deepseek_v4::mxfp4_load_enabled(cc, columns, rows, tokens) ||
            strata_mmvq_wide_rows(type, cc, columns, rows, 1)) return false;
        if (tokens == 1) {
            const auto dims = calc_launch_params<type>(1, rows, channels_dst, samples_dst, warp_size, table);
            const ggml_cuda_kernel_launch_params launch(dims.first, dims.second, 0, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, 1, false, false, false, true>, launch,
                vx, vy, ids, fusion, dst, columns, channels_y, row_x, col_y, col_dst,
                channel_ratio, channel_x, channel_y, channel_dst,
                sample_ratio, sample_x, sample_y, sample_dst, ids_stride);
        } else {
            constexpr int rows_per_block = 2;
            const dim3 grid((rows + rows_per_block - 1) / rows_per_block, channels_dst);
            const dim3 block(warp_size, tokens);
            const ggml_cuda_kernel_launch_params launch(grid, block, 0, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false, true>, launch,
                vx, vy, ids, fusion, dst, columns, channels_y, rows, row_x, col_y, col_dst,
                channel_x, channel_y, channel_dst, tokens, ids_stride);
        }
        return true;
    }
}
