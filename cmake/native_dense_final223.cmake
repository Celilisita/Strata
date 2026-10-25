# This integration publishes the qualified SM120 slice. The incomplete historic
# nvfp4_unavailable.cpp is not offered as a successful older-toolkit build.
if(NOT STRATA_QWEN35_DENSE_FINAL223)
  message(FATAL_ERROR
    "Dense final223 source requires explicit STRATA_QWEN35_DENSE_FINAL223=ON; "
    "this slice is scoped to CUDA 13 / SM120")
endif()
if(CUDAToolkit_VERSION VERSION_LESS "13.0" OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
  message(FATAL_ERROR "Dense final223 requires the qualified CUDA 13 or newer toolchain")
endif()
set(_strata_dense_has_sm120 FALSE)
foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(_arch MATCHES "^120a?(-real|-virtual)?$")
    set(_strata_dense_has_sm120 TRUE)
  endif()
endforeach()
if(NOT _strata_dense_has_sm120)
  message(FATAL_ERROR
    "Dense final223 requires an explicit SM120 architecture; "
    "other hardware providers remain outside this integration slice")
endif()
if(NOT TARGET CUDA::cublasLt)
  message(FATAL_ERROR "Dense final223 requires CUDA::cublasLt from the selected toolkit")
endif()
