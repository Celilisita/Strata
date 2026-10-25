# Consolidated production sources. Include after native_families and P20 registration.
include_guard(GLOBAL)
if(NOT TARGET strata_qwen35moe_runtime)
  return()
endif()
function(strata_completed_moe_sources target)
  get_target_property(_existing ${target} SOURCES)
  set(_known)
  foreach(_source IN LISTS _existing)
    cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _absolute)
    list(APPEND _known "${_absolute}")
  endforeach()
  foreach(_source IN LISTS ARGN)
    cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" NORMALIZE OUTPUT_VARIABLE _absolute)
    if(NOT _absolute IN_LIST _known)
      target_sources(${target} PRIVATE "${_absolute}")
      list(APPEND _known "${_absolute}")
    endif()
  endforeach()
endfunction()
strata_completed_moe_sources(strata_qwen35moe_runtime
  src/qwen35moe/runtime_prefetch.cpp
  src/qwen35moe/expert_prefetch.cpp
  src/qwen35moe/runtime_copy_runs.cpp
  src/qwen35moe/runtime_host_pin.cpp
  src/qwen35moe/runtime_batch_moe_compute.cpp
  src/qwen35moe/runtime_moe_nvfp4.cpp
  src/qwen35moe/runtime_moe_nvfp4_device.cpp
  src/qwen35moe/runtime_nvfp4_admission.cpp
  src/qwen35moe/runtime_gdn_graph.cpp)
strata_completed_moe_sources(strata_qwen35moe_cuda src/qwen35moe/cuda/expert_copy_stage.cu)
strata_completed_moe_sources(strata_mmq
  src/prefill/native_q6_accum_mmq.cu
  src/prefill/native_grouped_mmq_blackwell.cpp
  src/prefill/native_grouped_mmq_ordinary.cpp)
strata_completed_moe_sources(strata_kernels_cpu src/kernels/cpu/nvfp4_avx2.cpp)
if(MSVC)
  set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mf16c;-mfma")
endif()
set(_strata_sm120 FALSE)
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_arch MATCHES "^120(a|f)?(-real|-virtual)?$")
    set(_strata_sm120 TRUE)
  endif()
endforeach()
if(_strata_sm120)
  include("${CMAKE_CURRENT_LIST_DIR}/native_moe_blackwell.cmake")
  set(_nvfp4_cuda src/prefill/native_grouped_nvfp4.cu)
else()
  set(_nvfp4_cuda src/prefill/native_nvfp4_unavailable.cpp)
endif()
add_library(strata_moe_nvfp4 STATIC ${_nvfp4_cuda} src/prefill/native_grouped_nvfp4_provider.cpp)
target_include_directories(strata_moe_nvfp4 PUBLIC include PRIVATE src/prefill)
target_compile_features(strata_moe_nvfp4 PUBLIC cxx_std_17)
target_link_libraries(strata_moe_nvfp4 PUBLIC CUDA::cudart)
if(_strata_sm120)
  set_target_properties(strata_moe_nvfp4 PROPERTIES CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(strata_moe_nvfp4 PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_120a,code=sm_120a>")
  set_property(SOURCE src/prefill/native_grouped_mmq_blackwell.cpp
    src/qwen35moe/runtime_batch_moe_compute.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_BLACKWELL_MMQ=1)
  add_subdirectory(tools/conversion/moe_nvfp4 EXCLUDE_FROM_ALL)
endif()
target_link_libraries(strata_qwen35moe_runtime PUBLIC strata_moe_nvfp4)
