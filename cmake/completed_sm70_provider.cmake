# Completed SM70 kernels; shared Dense state/math ABI remains owned by final223.
include_guard(GLOBAL)
option(STRATA_COMPLETED_SM70_PROVIDERS "Build accepted SM70 provider archive" OFF)
if(NOT STRATA_COMPLETED_SM70_PROVIDERS)
  return()
endif()
if(NOT TARGET strata_completed_dense_common)
  message(FATAL_ERROR "Include completed_gemma.cmake before completed_sm70_provider.cmake")
endif()
add_library(strata_completed_sm70_provider STATIC
  src/qwen35/native/gdn_subwarp16.cu
  src/qwen35/native/gdn_subwarp8.cu
  src/qwen35/native/gdn_subwarp_dual.cu
  src/qwen35/native/attention_short32.cu)
set_target_properties(strata_completed_sm70_provider PROPERTIES
  CUDA_ARCHITECTURES 70 CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(strata_completed_sm70_provider PUBLIC include)
target_link_libraries(strata_completed_sm70_provider PUBLIC strata_completed_dense_common)
if(TARGET strata_qwen35_gdn_vectorized)
  target_link_libraries(strata_completed_sm70_provider PUBLIC strata_qwen35_gdn_vectorized)
else()
  target_sources(strata_completed_sm70_provider PRIVATE
    src/qwen35/native/gdn_vectorized.cu
    src/qwen35/native/gdn.cu
    src/qwen35/native/attention_keys64.cu
    src/qwen35/native/attention_prefill.cu)
endif()
# No Dense selector or default is changed here. Shared model/MTP/KV semantics
# keep their mature owner; original-format providers use separate typed archives.
