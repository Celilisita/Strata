# Explicit parent integration hook; this file does not change existing families.
function(strata_add_gemma4_sm120 source_root)
  if(TARGET strata_gemma4_sm120)
    return()
  endif()
  if(NOT TARGET strata_qwen35_native OR NOT TARGET strata_qwen35_dense_sm120 OR NOT TARGET CUDA::cublasLt)
    message(FATAL_ERROR "Gemma SM120 requires the one existing generic NVFP4 primitive owner")
  endif()
  if(CUDAToolkit_VERSION VERSION_LESS "13.0" OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "Gemma SM120 NVFP4 requires CUDA 13 or newer")
  endif()
  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/mmq_api.cmake")
  # Exact selected-B runtime archive closure:45TUs, no checker/benchmark.
  set(_sources
    sm120_model_forward.cpp
    original_q6_head.cu
    global_keys1024_attention.cpp
    global_keys1024_layout.cpp
    global_keys1024_ops.cu
    local_gqa2_attention.cpp
    local_gqa2_ops.cu
    sm120_cached_global.cu
    sm120_cached_global.cpp
    whole_history.cu
    whole_history.cpp
    sm120_state.cu
    sm120_math.cu
    linear.cu
    sm120_global_flash.cu
    sm120_global_flash_layout.cpp
    pp_tc_ops.cu
    pp_tc_attention.cpp
    pp_tc_layout.cpp
    attention_blas_ops.cu
    attention_blas.cpp
    tg_local270.cu
    sm120_attention_ops.cu
    original_mmq_budget.cu
    original_quant_projection.cpp
    sm120_projection_plan.cpp
    sm120_spec.cpp
    sm120_budget.cpp
    sm120_attention.cpp
    sm120_sampled_mtp_ops.cu
    sm120_mtp_assistant_split.cu
    sm120_mtp_assistant_quant.cu
    sm120_mtp_assistant_head.cu
    sm120_mtp_assistant_forward.cu
    sm120_mtp_target.cu
    sm120_mtp_assistant_spec.cpp
    sm120_mtp_assistant_load.cpp
    sm120_model_verify.cpp
    sm120_model_load.cpp
    sm120_api.cpp
    nvfp4_ffn.cpp
    nvfp4_projection.cpp
    sm120_mtp_phase_timing.cu
    sm120_sampled_mtp.cpp
    sm120_attention_split.cu
  )
  set(_paths)
  foreach(_source IN LISTS _sources)
    set(_path "${source_root}/src/gemma4/native/${_source}")
    if(NOT EXISTS "${_path}")
      message(FATAL_ERROR "Gemma SM120 missing production source: ${_path}")
    endif()
    list(APPEND _paths "${_path}")
  endforeach()
  # The selected parent is Query512/Keys512; this is not a geometry sweep.
  set(STRATA_GEMMA_PP_TC_QUERY "512" CACHE STRING "Selected Gemma TC query tile")
  if(NOT STRATA_GEMMA_PP_TC_QUERY STREQUAL "512")
    message(FATAL_ERROR "Selected Gemma sources require STRATA_GEMMA_PP_TC_QUERY=512")
  endif()
  # These are the actual parent/leaf consumers, including inactive link closure.
  foreach(_tc_source pp_tc_layout.cpp pp_tc_attention.cpp pp_tc_ops.cu
      global_keys1024_attention.cpp global_keys1024_layout.cpp global_keys1024_ops.cu
      local_gqa2_attention.cpp local_gqa2_ops.cu
      sm120_cached_global.cpp sm120_cached_global.cu)
    set_property(SOURCE "${source_root}/src/gemma4/native/${_tc_source}" APPEND PROPERTY
      COMPILE_DEFINITIONS "STRATA_GEMMA_PP_TC_QUERY=512")
  endforeach()
  foreach(_local_source local_gqa2_attention.cpp local_gqa2_ops.cu)
    set_property(SOURCE "${source_root}/src/gemma4/native/${_local_source}" APPEND PROPERTY
      COMPILE_DEFINITIONS "STRATA_GEMMA_PP_TC_LOCAL_GQA2=1")
  endforeach()
  add_library(strata_gemma4_sm120 STATIC ${_paths})
  if(MSVC)
    foreach(_strict_host global_keys1024_attention.cpp global_keys1024_layout.cpp local_gqa2_attention.cpp)
      set_property(SOURCE "${source_root}/src/gemma4/native/${_strict_host}" APPEND PROPERTY
        COMPILE_OPTIONS /fp:strict)
    endforeach()
  endif()
  target_include_directories(strata_gemma4_sm120 BEFORE PUBLIC "${source_root}/include")
  target_include_directories(strata_gemma4_sm120 PRIVATE "${source_root}/src/gemma4/native")
  set(_primitive_root "${PROJECT_SOURCE_DIR}")
  if(ARGC GREATER 1)
    set(_primitive_root "${ARGV1}")
  endif()
  if(NOT EXISTS "${_primitive_root}/src/qwen35/native/fp4_compensated_service.hpp")
    message(FATAL_ERROR "Gemma tiny requires the registered generic NVFP4 primitive source root")
  endif()
  target_include_directories(strata_gemma4_sm120 PRIVATE "${_primitive_root}/src/qwen35/native")
  if(NOT STRATA_MMQ_KQUANTS OR NOT TARGET strata_mmq OR NOT TARGET strata_kernels)
    message(FATAL_ERROR "Gemma original quant requires existing K-quant MMQ and native MMVQ targets")
  endif()
  get_target_property(_gemma_mmq_sources strata_mmq SOURCES)
  get_target_property(_gemma_mmq_definitions strata_mmq COMPILE_DEFINITIONS)
  # Preserve existing target flags. If Q6 projection dispatch is not already
  # enabled, update only the actual macro consumer, not all template instances.
  if(NOT STRATA_Q6K_EXPERTS AND NOT "STRATA_NATIVE_MOE_Q6_MMQ=1" IN_LIST _gemma_mmq_definitions)
    set(_gemma_q6_dispatch_count 0)
    foreach(_gemma_source IN LISTS _gemma_mmq_sources)
      if(NOT IS_ABSOLUTE "${_gemma_source}")
        set(_gemma_source "${PROJECT_SOURCE_DIR}/${_gemma_source}")
      endif()
      if(_gemma_source MATCHES "/moe_mmq[^/]*\\.cu$" AND EXISTS "${_gemma_source}")
        file(READ "${_gemma_source}" _gemma_dispatch_text)
        string(FIND "${_gemma_dispatch_text}" "defined(STRATA_NATIVE_MOE_Q6_MMQ)" _gemma_q6_macro)
        if(NOT _gemma_q6_macro EQUAL -1)
          set_property(SOURCE "${_gemma_source}" TARGET_DIRECTORY strata_mmq APPEND PROPERTY
            COMPILE_DEFINITIONS STRATA_MMQ_KQUANTS=1 STRATA_NATIVE_MOE_Q6_MMQ=1)
          math(EXPR _gemma_q6_dispatch_count "${_gemma_q6_dispatch_count}+1")
        endif()
      endif()
    endforeach()
    if(NOT _gemma_q6_dispatch_count EQUAL 1)
      message(FATAL_ERROR "Gemma requires one identified Q6 MMQ macro consumer")
    endif()
  endif()
  if(NOT STRATA_DEEPSEEK_V4 AND NOT STRATA_Q6K_EXPERTS)
    set(_gemma_q6 "${_strata_mmq}/template-instances/mmq-instance-q6_k.cu")
    if(NOT _gemma_q6 IN_LIST _gemma_mmq_sources)
      target_sources(strata_mmq PRIVATE "${_gemma_q6}")
    endif()
  endif()
  # Inactive flash remains a symbol dependency of the tested AS/budget source.
  # Keep its selected device-header closure; no ggml host/runtime owner is added.
  set(_gemma_flash_pins
    "fattn-mma-f16.cuh|80a40451e0054acec97c04859281d5fd13292366e54f63bf8009ed2c6c9c7091"
    "fattn-common.cuh|d47e737d80329f4b0862e92109b060a4ef7031041165b812ce0d6792b9466679"
    "mma.cuh|4acce516e9463607122f562a9cdab6aa1f25c37f937980701c20075c0149f615"
    "cp-async.cuh|c0857cb62072ead470b2ec86039a90cd06b1825443fc523564992f5b7d94fec4"
    "common.cuh|cde04ff564d8523f0b2c1789ee43112c1feafc50e2c3757c888169fe0ba4ea56"
    "convert.cuh|c22bcfa617140b6dfd7fb5eb11fc5ae4c25244a435aceb8e7c4c26f4bd578fd0"
    "vecdotq.cuh|200dad1a94624716460b019d7f2bb38777d7e1348109a5cc623d03ce69ba275a"
    "vendors/cuda.h|e7847065d5d741f892d2f1f7aaaa8f418a41e3b983d1e6f09010055d4d1c38d1"
    "../ggml-impl.h|43564db0238aebb7ed68501e346c194866b5dac218d1d37b26baff9f458c00d3"
    "../ggml-common.h|0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c"
    "../../include/ggml.h|12ee71f99db7db9b353bc02b1fbb57c344ee17c01ac5fb7952b41a637a747ea9"
    "../../include/ggml-cuda.h|9cbfcea49f6da4d07f41dc22fa8dbf870825faaa2a68e34e65fe30ea6c41e042"
  )
  foreach(_pin IN LISTS _gemma_flash_pins)
    string(REPLACE "|" ";" _pair "${_pin}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _expected)
    if(NOT EXISTS "${_strata_mmq}/${_name}")
      message(FATAL_ERROR "Gemma flash leaf missing selected dependency: ${_name}")
    endif()
    file(SHA256 "${_strata_mmq}/${_name}" _actual)
    if(NOT _actual STREQUAL _expected)
      message(FATAL_ERROR "Gemma flash leaf selected dependency differs: ${_name}")
    endif()
  endforeach()
  target_include_directories(strata_gemma4_sm120 PRIVATE
    "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}")
  # This inactive TU needs extended-lambda exactly as in its retained recipe.
  set_property(SOURCE "${source_root}/src/gemma4/native/sm120_global_flash.cu" APPEND PROPERTY
    INCLUDE_DIRECTORIES "${_strata_gg}/include" "${_strata_gg}/src" "${_strata_mmq}")
  set_property(SOURCE "${source_root}/src/gemma4/native/sm120_global_flash.cu" APPEND PROPERTY
    COMPILE_OPTIONS --extended-lambda)
  if(STRATA_MMQ_API_SIGNATURE STREQUAL "explicit-prec-src1-q8")
    set_property(SOURCE "${source_root}/src/gemma4/native/original_mmq_budget.cu" APPEND PROPERTY
      COMPILE_DEFINITIONS STRATA_DEEPSEEK_V4_GGML_API=1)
  elseif(NOT STRATA_MMQ_API_SIGNATURE STREQUAL "legacy-q8")
    message(FATAL_ERROR "Gemma original requires the already identified selected MMQ API")
  endif()
  target_link_libraries(strata_gemma4_sm120 PUBLIC
    strata_qwen35_native strata_mmq strata_kernels CUDA::cublas CUDA::cublasLt CUDA::cudart)
  set_target_properties(strata_gemma4_sm120 PROPERTIES
    CUDA_ARCHITECTURES 120a CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_compile_definitions(strata_gemma4_sm120 PRIVATE $<$<PLATFORM_ID:Windows>:NOMINMAX>)
  target_compile_options(strata_gemma4_sm120 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
  if(NOT TARGET strata_native_sampling)
    message(FATAL_ERROR "Gemma native service requires the one existing shared sampler owner")
  endif()
  add_executable(strata-gemma4-sm120
    "${source_root}/src/gemma4/native/sm120_main.cpp"
    "${source_root}/src/gemma4/native/sm120_serve.cpp")
  target_link_libraries(strata-gemma4-sm120 PRIVATE strata_gemma4_sm120 strata_native_sampling)
  set_target_properties(strata-gemma4-sm120 PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_compile_definitions(strata-gemma4-sm120 PRIVATE $<$<PLATFORM_ID:Windows>:NOMINMAX>)
endfunction()
