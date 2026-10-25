# Reuse the same extracted common owner if the existing families are disabled.
if(NOT TARGET strata_completed_dense_common)
  add_library(strata_completed_dense_common STATIC src/qwen35/tensor_bytes.cpp
    src/qwen35/native/cuda_error.cpp src/qwen35/native/elementwise_common.cu
    src/runtime/capacity_storage.cpp)
  target_include_directories(strata_completed_dense_common PUBLIC include PRIVATE src/qwen35/native)
  set_target_properties(strata_completed_dense_common PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  target_link_libraries(strata_completed_dense_common PUBLIC CUDA::cudart)
  target_compile_definitions(strata_completed_dense_common PRIVATE NOMINMAX)
  if(MSVC)
    target_compile_options(strata_completed_dense_common PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:/utf-8;/bigobj>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  endif()
endif()
