# DeepSeek-V4.1-Flash: a separate engine path (src/dsv41), ported from Project Maya's
# maya-ds4 under MIT (third_party/project-maya). OFF by default: it adds no source to
# any existing target, and its vendored kernels are their own static library.
option(STRATA_BUILD_DSV41 "Build the DeepSeek-V4.1 engine path (strata-dsv41)" OFF)

if(STRATA_BUILD_DSV41 AND NOT STRATA_HIP_GFX906)
  add_library(strata_dsv41_kernels STATIC
              src/dsv41/vendor/exl3.cu src/dsv41/vendor/sampler.cu
              src/dsv41/vendor/moe_win.cu src/dsv41/vendor/exl3_cpu.cpp)
  if(NOT STRATA_ENABLE_HIP)
    # the pre-sm_80 NVIDIA decode kernels exl3.cu dispatches to
    target_sources(strata_dsv41_kernels PRIVATE src/dsv41/vendor/exl3_legacy.cu)
  endif()
  target_include_directories(strata_dsv41_kernels PUBLIC
                             ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories})
  target_link_libraries(strata_dsv41_kernels PUBLIC ${_strata_gpu_runtime_target})

  add_executable(strata-dsv41
                 src/dsv41/ds4_main.cpp src/dsv41/ds4_model.cu src/dsv41/ds4_kernels.cu
                 src/dsv41/ds4_vision.cu src/dsv41/ds4_gguf.cu src/dsv41/ds4_gguf_cpu.cpp
                 src/dsv41/ds4_iq.cu src/dsv41/ds4_iq_cpu.cpp src/dsv41/ds4_tq.cu
                 src/dsv41/ds4_tq_cpu.cpp src/dsv41/ds4_gg_cpu.cpp src/dsv41/ds4_packinfo.cpp)

  # the commit the binary was built from, readable with --version
  if(NOT STRATA_BUILD_ID AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID)
    file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/BUILD_ID STRATA_BUILD_ID LIMIT_COUNT 1)
  endif()
  if(NOT STRATA_BUILD_ID)
    execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short=9 HEAD
                    OUTPUT_VARIABLE STRATA_BUILD_ID OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  if(NOT STRATA_BUILD_ID)
    set(STRATA_BUILD_ID unknown)
  endif()
  set_source_files_properties(src/dsv41/ds4_main.cpp PROPERTIES
                              COMPILE_DEFINITIONS "DS4_BUILD_ID=\"${STRATA_BUILD_ID}\"")

  target_include_directories(strata-dsv41 PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/include ${_strata_gpu_include_directories}
                             ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)
  target_link_libraries(strata-dsv41 PRIVATE strata_dsv41_kernels
                        ${_strata_gpu_runtime_target} ${_strata_gpu_blas_target})

  find_package(Threads REQUIRED)
  target_link_libraries(strata-dsv41 PRIVATE Threads::Threads)

  if(WIN32 AND STRATA_ENABLE_HIP)
    # cudaMemGetInfo is the DXGI-budgeted hip_compat::mem_get_info on the AMD backend
    target_link_libraries(strata-dsv41 PRIVATE strata_core)
  endif()

  if(STRATA_NATIVE_EXPERTS)
    # Strata's CPU trellis kernels (tq_cpu.hpp) for the experts on the CPU lane
    target_link_libraries(strata-dsv41 PRIVATE strata_kernels_cpu)
    target_compile_definitions(strata-dsv41 PRIVATE STRATA_DSV41_TQ_CPU=1)
  endif()
endif()

if(STRATA_BUILD_DSV41 AND NOT STRATA_ENABLE_CUDA AND NOT STRATA_ENABLE_HIP AND NOT STRATA_HIP_GFX906)
  message(STATUS "STRATA_BUILD_DSV41: no GPU backend; strata-dsv41 is not built")
endif()
