# Native FP4 instructions are isolated from general Dense kernels. These source
# files contain unguarded SM120 PTX and must not inherit an older GPU target.
add_library(strata_qwen35_dense_sm120 OBJECT
  src/qwen35/native/fp4_compensated.cu
  src/qwen35/native/gutail_encode.cu
  src/qwen35/native/nvfp4_ops.cu
  src/qwen35/native/tiny_nvfp4.cu
  src/qwen35/native/fused_nvfp4.cu
  src/qwen35/native/fp4_packed.cu
  src/qwen35/native/fp4_unpack_vector.cu
  src/qwen35/native/fp4_gu_fused.cu
  src/qwen35/native/fp4_gu_tma.cu
  src/qwen35/native/fp4_gu_merge.cu
  src/qwen35/native/fp4_residual_norm.cu
  src/qwen35/native/inputnorm_fused.cu
  src/qwen35/native/half_tail.cu
  src/qwen35/native/fp8_pp.cu
  src/qwen35/native/fp8_dual_cta.cu
  src/qwen35/native/fp8_p_fusion.cu
  src/qwen35/native/gdn_output_elision.cu)
target_include_directories(strata_qwen35_dense_sm120 PRIVATE
  include src/qwen35/native third_party/ggml)
target_link_libraries(strata_qwen35_dense_sm120 PRIVATE CUDA::cudart)
set_target_properties(strata_qwen35_dense_sm120 PROPERTIES
  CUDA_ARCHITECTURES 120a CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
  CUDA_RUNTIME_LIBRARY Shared POSITION_INDEPENDENT_CODE ON)
target_compile_options(strata_qwen35_dense_sm120 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Exactly one native runtime, one float-output GEMM and one half-output GEMM.
# No prebuilt object or private archive is pulled from the qualification tree.
target_sources(strata_qwen35_native PRIVATE
  $<TARGET_OBJECTS:strata_qwen35_dense_sm120>
  src/qwen35/native/nvfp4_gemm.cpp
  src/qwen35/native/half_output_gemm.cpp
  src/qwen35/native/nvfp4_weights.cpp
  src/qwen35/native/q8_pp_sgemm.cu
  src/qwen35/native/nvfp4_runtime.cpp)
target_link_libraries(strata_qwen35_native PUBLIC CUDA::cublasLt CUDA::cublas)
foreach(_target IN ITEMS strata_native_sampling strata_qwen35_native
    strata_qwen35_cublas strata_qwen35_cutlass_swiglu
    strata_qwen35_gdn_vectorized strata_qwen35_dense_runtime)
  set_target_properties(${_target} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  target_compile_definitions(${_target} PRIVATE $<$<PLATFORM_ID:Windows>:NOMINMAX>)
endforeach()
target_compile_definitions(strata_qwen35_dense_sm120 PRIVATE $<$<PLATFORM_ID:Windows>:NOMINMAX>)
