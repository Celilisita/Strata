# Completed V100 operator providers, sharing the mature model/state ABI.
# The parent chooses available toolchain/ISA; compiling is not runtime acceptance.
include_guard(GLOBAL)
option(STRATA_COMPLETED_V100_KERNELS "Build completed V100 operator provider archive" OFF)
if(NOT STRATA_COMPLETED_V100_KERNELS)
  return()
endif()
if(NOT STRATA_ENABLE_CUDA OR NOT TARGET strata_completed_dense_common)
  message(FATAL_ERROR "Completed V100 kernels require the unified CUDA common target")
endif()
add_library(strata_completed_v100_kernels STATIC
  src/qwen35/native/attention_online_t3.cu
  src/qwen35/native/attention_verify_online.cu
  src/qwen35/native/pp_gqa.cu
  src/qwen35/native/pp_prefetch_gdn.cu
  src/qwen35/native/q6_soa.cu
  src/qwen35/native/q6_head_policy.cu
  src/qwen35/native/prefill_down_tensor.cu)
set_target_properties(strata_completed_v100_kernels PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
target_include_directories(strata_completed_v100_kernels PUBLIC include
  PRIVATE src/qwen35/native third_party/ggml)
target_include_directories(strata_completed_v100_kernels PRIVATE
  "${STRATA_CUTLASS_DIR}/include")
target_compile_definitions(strata_completed_v100_kernels PRIVATE NOMINMAX)
target_compile_options(strata_completed_v100_kernels PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
if(MSVC)
  target_compile_options(strata_completed_v100_kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
target_link_libraries(strata_completed_v100_kernels PUBLIC
  strata_completed_dense_common CUDA::cudart CUDA::cublas)
# Online attention needs only the mature partial workspace/merge functions.
if(TARGET strata_qwen35_native)
  target_link_libraries(strata_completed_v100_kernels PUBLIC strata_qwen35_native)
else()
  target_sources(strata_completed_v100_kernels PRIVATE src/qwen35/native/attention_verify.cu)
endif()
# pp_prefetch_gdn borrows the already completed subwarp16 workspace contract.
# Keep one symbol owner whether the older SM70 archive is enabled or not.
if(TARGET strata_completed_sm70_provider)
  target_link_libraries(strata_completed_v100_kernels PUBLIC strata_completed_sm70_provider)
else()
  target_sources(strata_completed_v100_kernels PRIVATE src/qwen35/native/gdn_subwarp16.cu)
  if(TARGET strata_qwen35_gdn_vectorized)
    target_link_libraries(strata_completed_v100_kernels PUBLIC strata_qwen35_gdn_vectorized)
  else()
    target_sources(strata_completed_v100_kernels PRIVATE src/qwen35/native/gdn_vectorized.cu)
  endif()
endif()
# No model dispatch/profile/default is modified by compiling this archive.
