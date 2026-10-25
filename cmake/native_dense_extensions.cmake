# General Dense kernels and state operations. The base target owns all existing
# DenseModel / NativeMtp implementations; these are additional translation units.
target_sources(strata_qwen35_native PRIVATE
  src/qwen35/native/compact_attention.cu
  src/qwen35/native/fp16pv_attention.cu
  src/qwen35/native/pv_reuse_attention.cu
  src/qwen35/native/row_inverse_attention.cu
  src/qwen35/native/rescale_skip_attention.cu
  src/qwen35/native/conv_gdn_fused.cu
  src/qwen35/native/gdn_delta_half.cu
  src/qwen35/native/gdn_float_value_prepare.cu
  src/qwen35/native/gdn_float_value_scan.cu
  src/qwen35/native/gdn_register_prepare.cu
  src/qwen35/native/gdn_weighted_qk.cu
  src/qwen35/native/kreg_prepare.cu
  src/qwen35/native/ram_block_sort.cu
  src/qwen35/native/ram_block_sort_policy.cpp
  src/qwen35/native/ram40_score.cu
  src/qwen35/native/ram_six.cu
  src/qwen35/native/page_row.cu
  src/qwen35/native/prefix_copy.cu
  src/qwen35/native/state_snapshot_copy.cu
  src/qwen35/native/t1_tc.cu
  src/qwen35/native/regp.cu
  src/qwen35/native/rmsconcat.cu
  src/qwen35/native/cold_head_binding.cpp
  src/qwen35/native/mtp_nvfp4_weights.cpp
  src/qwen35/native/swiglu_vector.cu
  src/qwen35/native/tiny32.cu
  src/qwen35/native/tiny_mma.cu
  src/qwen35/native/warp_logical.cu
  src/qwen35/native/q6_warp_logical.cu)
target_sources(strata_qwen35_dense_runtime PRIVATE
  src/qwen35/native/mtp_gumbel.cu
  src/qwen35/native/mtp_prob.cu)
# partial_gemm extends the existing cuBLAS/CUTLASS fallback interface. It does
# not define a second NVFP4 Runtime or replace the original GEMM factory.
target_sources(strata_qwen35_cutlass_swiglu PRIVATE src/qwen35/native/partial_gemm.cu)
target_include_directories(strata_qwen35_native PRIVATE src/qwen35/native)
target_include_directories(strata_qwen35_dense_runtime PRIVATE src/qwen35/native)
target_include_directories(strata PRIVATE src/qwen35/native)
# Keep the selected service-source include as well as the live target include.
set_property(SOURCE src/qwen35/native/serve.cpp APPEND PROPERTY INCLUDE_DIRECTORIES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/qwen35/native")
target_compile_options(strata_qwen35_cutlass_swiglu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
