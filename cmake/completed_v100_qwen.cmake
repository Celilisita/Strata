# Completed V100 normal Qwen executor, isolated from the final223 owner.
include_guard(GLOBAL)
option(STRATA_COMPLETED_V100_QWEN "Build completed V100 Qwen normal model entry" OFF)
if(NOT STRATA_COMPLETED_V100_QWEN)
  return()
endif()
if(NOT STRATA_ENABLE_CUDA)
  message(FATAL_ERROR "Completed V100 Qwen requires CUDA")
endif()
if(NOT TARGET strata_completed_dense_common)
  # Register the already extracted common source under the existing opt-in.
  set(STRATA_COMPLETED_V100_COMMON ON)
  include("${CMAKE_CURRENT_LIST_DIR}/completed_v100_qwen_common.cmake")
endif()
if(NOT TARGET strata_native_sampling)
  add_library(strata_native_sampling STATIC src/kernels/cuda/radix_sampler.cu)
  target_include_directories(strata_native_sampling PUBLIC include)
  set_target_properties(strata_native_sampling PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(strata_native_sampling PUBLIC CUDA::cudart)
endif()
include("${CMAKE_CURRENT_LIST_DIR}/completed_v100_qwen_sources.cmake")
add_library(strata_qwen_v100_completed STATIC ${STRATA_COMPLETED_V100_QWEN_SOURCES})
target_include_directories(strata_qwen_v100_completed PUBLIC include
  PRIVATE src/qwen35/completed_v100/native third_party/ggml "${STRATA_CUTLASS_DIR}/include")
set_target_properties(strata_qwen_v100_completed PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(strata_qwen_v100_completed PRIVATE NOMINMAX)
target_compile_options(strata_qwen_v100_completed PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_link_libraries(strata_qwen_v100_completed PUBLIC strata_completed_dense_common
  strata_native_sampling strata_artifact CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
if(MSVC)
  target_compile_options(strata_qwen_v100_completed PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
add_executable(strata-qwen-v100-native src/program/qwen_v100_main.cpp)
target_link_libraries(strata-qwen-v100-native PRIVATE strata_qwen_v100_completed)
# No link to final223/model runtime, old executable, object, or provider archive.
