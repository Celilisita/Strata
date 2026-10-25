# The actual original GDN has this SOURCE option; strata_kernels alone does not inherit it.
set_property(SOURCE "${CMAKE_SOURCE_DIR}/src/kernels/cuda/native_gdn_coalesced.cu"
  APPEND PROPERTY COMPILE_OPTIONS --use_fast_math)
