# Original-block V100 quantized provider; distinct from final223 SM120 planes.
include_guard(GLOBAL)
option(STRATA_COMPLETED_ORIGINAL_PROVIDER "Build frozen original-block provider" OFF)
if(NOT STRATA_COMPLETED_ORIGINAL_PROVIDER)
  return()
endif()
if(NOT STRATA_ENABLE_CUDA OR NOT TARGET strata_completed_dense_common)
  message(FATAL_ERROR "Original provider requires CUDA and completed common modules")
endif()
if(NOT TARGET strata_completed_sm70_provider OR NOT TARGET strata_completed_v100_kernels)
  message(FATAL_ERROR "Original provider requires the completed SM70/kernel archives")
endif()
add_library(strata_completed_original_provider STATIC
  src/qwen35/native/nvfp4_cublas_prefill.cu
  src/qwen35/native/nvfp4_decode.cu
  src/qwen35/native/nvfp4_decode8.cu
  src/qwen35/native/nvfp4_gemm.cu
  src/qwen35/native/nvfp4_interleaved.cu
  src/qwen35/native/nvfp4_native_m8.cu
  src/qwen35/native/nvfp4_packed.cu
  src/qwen35/native/nvfp4_pair.cu
  src/qwen35/native/nvfp4_pair_swiglu.cu
  src/qwen35/native/nvfp4_prefill_fused.cu
  src/qwen35/native/nvfp4_prefill_gdn.cu
  src/qwen35/native/nvfp4_prefill_lt.cpp
  src/qwen35/native/nvfp4_prefill_lt.cu
  src/qwen35/native/nvfp4_prefill_pair.cu
  src/qwen35/native/nvfp4_projection.cu
  src/qwen35/native/nvfp4_q8_activation.cu
  src/qwen35/native/nvfp4_q8_compact.cu
  src/qwen35/native/nvfp4_q8_fast.cu
  src/qwen35/native/nvfp4_q8_fastsoa.cu
  src/qwen35/native/nvfp4_q8_halfscale.cu
  src/qwen35/native/nvfp4_q8_soa.cu
  src/qwen35/native/nvfp4_repack.cu
  src/qwen35/native/nvfp4_wmma.cu
  src/qwen35/native/original_cutlass.cu
  src/qwen35/native/original_fused.cu
  src/qwen35/native/original_upload.cpp
  src/qwen35/native/p2_vector_encode.cu
  src/qwen35/native/pp_prefetch.cpp
  src/qwen35/native/pp_prefetch_core.cpp
  src/qwen35/native/q35_shared_prepare.cpp
  src/qwen35/native/q8_power2_prefill.cu
  src/qwen35/native/q8_projection.cu
  src/qwen35/native/raw_q8_blocked.cu
  src/qwen35/native/raw_q8_packed.cu
  src/qwen35/native/raw_q8_projection.cu
  src/qwen35/native/raw_q8_wide.cu)
target_include_directories(strata_completed_original_provider PUBLIC include
  PRIVATE src/qwen35/native "${STRATA_CUTLASS_DIR}/include")
set_target_properties(strata_completed_original_provider PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(strata_completed_original_provider PRIVATE NOMINMAX)
target_compile_options(strata_completed_original_provider PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_link_libraries(strata_completed_original_provider PUBLIC
  strata_completed_dense_common strata_completed_sm70_provider strata_completed_v100_kernels
  strata_artifact CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
if(MSVC)
  target_compile_options(strata_completed_original_provider PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
# Call-site selection remains explicit at the raw-F32 boundary; default native
# Qwen execution continues to use its registered final223 provider and layout.
