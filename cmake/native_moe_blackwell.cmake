include_guard(GLOBAL)
# Preserve the measured SM120 MoE providers without changing other GPU builds.
if(NOT _strata_sm120)
  return()
endif()
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR "Native SM120 MoE providers require CUDA12.8 or newer")
endif()
add_library(strata_mmq_blackwell STATIC src/prefill/native_blackwell_mmq.cu)
set_target_properties(strata_mmq_blackwell PROPERTIES CUDA_ARCHITECTURES "120a"
  CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(strata_mmq_blackwell PRIVATE
  ${_strata_gg}/include ${_strata_gg}/src ${_strata_mmq}
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(strata_mmq_blackwell PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
target_link_libraries(strata_mmq_blackwell PUBLIC ggml-base CUDA::cudart)
target_link_libraries(strata_mmq PUBLIC strata_mmq_blackwell)
set_property(SOURCE src/prefill/native_q6_accum_mmq.cu
  src/prefill/native_grouped_mmq_provider.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS STRATA_BLACKWELL_MMQ=1)
set_property(SOURCE src/qwen35moe/runtime_projection_mmq.cpp
  src/qwen35moe/runtime_moe_down_mmq.cpp
  src/qwen35moe/runtime_batch_moe_grouped.cpp APPEND PROPERTY
  COMPILE_DEFINITIONS STRATA_BLACKWELL_MMQ=1)
target_sources(strata_qwen35moe_runtime PRIVATE
  src/qwen35moe/runtime_blackwell_projection.cpp
  src/qwen35moe/runtime_precise_dense.cpp)
