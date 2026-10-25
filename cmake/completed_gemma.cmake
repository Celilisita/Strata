# Completed V100 Gemma mathematics/state in the unified source project.
# Target ISA is the parent toolchain choice; compilation is not model acceptance.
include_guard(GLOBAL)
option(STRATA_COMPLETED_GEMMA "Build completed Gemma math/state and native entry" OFF)
if(NOT STRATA_ENABLE_CUDA)
  if(STRATA_COMPLETED_GEMMA)
    message(FATAL_ERROR "Completed Gemma source requires CUDA")
  endif()
  return()
endif()
# These extracted definitions are also the one owner for existing native Qwen
# targets. Gemma remains opt-in and cannot gate their ordinary link closure.
if(NOT STRATA_COMPLETED_GEMMA AND NOT STRATA_COMPLETED_SM70_PROVIDERS
   AND NOT TARGET strata_qwen35 AND NOT TARGET strata_qwen35_native)
  return()
endif()
add_library(strata_completed_dense_common STATIC
  src/qwen35/tensor_bytes.cpp
  src/qwen35/native/cuda_error.cpp
  src/qwen35/native/elementwise_common.cu
  src/runtime/capacity_storage.cpp)
target_include_directories(strata_completed_dense_common PUBLIC include
  PRIVATE src/qwen35/native)
set_target_properties(strata_completed_dense_common PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(strata_completed_dense_common PUBLIC CUDA::cudart)
target_compile_definitions(strata_completed_dense_common PRIVATE NOMINMAX)
if(TARGET strata_qwen35)
  target_link_libraries(strata_qwen35 PUBLIC strata_completed_dense_common)
endif()
if(TARGET strata_qwen35_native)
  target_link_libraries(strata_qwen35_native PUBLIC strata_completed_dense_common)
endif()
if(MSVC)
  target_compile_options(strata_completed_dense_common PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
if(NOT STRATA_COMPLETED_GEMMA)
  return()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/completed_gemma_sources.cmake")
if(NOT TARGET strata_native_sampling)
  add_library(strata_native_sampling STATIC src/kernels/cuda/radix_sampler.cu)
  target_include_directories(strata_native_sampling PUBLIC include)
  set_target_properties(strata_native_sampling PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(strata_native_sampling PUBLIC CUDA::cudart)
endif()
add_library(strata_gemma_dense STATIC ${STRATA_GEMMA_SOURCES}
  src/gemma4/service.cpp
  src/gemma4/native/model_sampling.cpp
  src/gemma4/native/model_sampling.cu)
target_include_directories(strata_gemma_dense PUBLIC include
  PRIVATE src/gemma4/native src/qwen35/native third_party/ggml)
target_include_directories(strata_gemma_dense BEFORE PRIVATE
  third_party/fmha-cutlass-override third_party/fmha third_party/cutlass/include)
set_target_properties(strata_gemma_dense PROPERTIES
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_definitions(strata_gemma_dense PRIVATE NOMINMAX)
target_compile_options(strata_gemma_dense PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
target_link_libraries(strata_gemma_dense PUBLIC
  strata_completed_dense_common strata_native_sampling CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
if(MSVC)
  target_compile_options(strata_gemma_dense PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()
add_executable(strata-gemma-native src/program/gemma_main.cpp)
target_link_libraries(strata-gemma-native PRIVATE strata_gemma_dense)
