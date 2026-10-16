if(STRATA_ENABLE_CUDA AND TARGET strata_kernels)
  add_executable(strata-mtp-q8-check EXCLUDE_FROM_ALL tests/mtp_q8_check.cpp)
  target_link_libraries(strata-mtp-q8-check PRIVATE strata_kernels strata_artifact CUDA::cudart)
endif()
