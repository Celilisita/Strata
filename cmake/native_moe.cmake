if(NOT STRATA_QWEN35_MOE)
  return()
endif()
add_library(strata_qwen35moe STATIC
  src/qwen35moe/spec.cpp
  src/qwen35moe/spec_bind.cpp)
target_include_directories(strata_qwen35moe PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35moe PUBLIC strata_artifact)
add_library(strata_qwen35moe_cuda STATIC
  src/core/native_miss.cu
  src/qwen35moe/cuda/router.cu
  src/qwen35moe/cuda/resident_gather.cu
  src/qwen35moe/cuda/head_select.cu
  src/qwen35moe/cuda/gdn_post.cu
  src/qwen35moe/cuda/gdn_l2.cu
  src/qwen35moe/cuda/moe_math.cu
  src/qwen35moe/cuda/flash_attn.cu
  src/qwen35moe/cuda/flash_attn_tile.cu
  src/qwen35moe/cuda/batch_quant.cu
  src/qwen35moe/cuda/moe_batch_ops.cu
  src/qwen35moe/cuda/attention_tc.cu
  src/qwen35moe/cuda/router_tiled.cu
  src/prefill/precise.cu
  src/prefill/gemm_scaled.cu
  src/kernels/cuda/native_gdn_scan.cu
  src/kernels/cuda/native_gdn_scan_preprocess.cu)
target_include_directories(strata_qwen35moe_cuda PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35moe_cuda PUBLIC CUDA::cudart CUDA::cublas strata_kernels strata_prefill strata_native_sampling)
add_library(strata_qwen35moe_runtime STATIC
  src/qwen35moe/visual_input.cpp
  src/qwen35moe/runtime_visual.cpp
  src/qwen35moe/runtime_session_gdn.cpp
  src/qwen35moe/runtime_session_mtp.cpp
  src/qwen35moe/runtime_session_pair.cpp
  src/qwen35moe/runtime_sessions.cpp
  src/qwen35moe/runtime_sampling_mtp.cpp
  src/qwen35moe/runtime_sampling.cpp
  src/qwen35moe/runtime_moe_graph.cpp
  src/qwen35moe/runtime_dynamic.cpp
  src/qwen35moe/runtime_mtp.cpp
  src/qwen35moe/runtime_speculative.cpp
  src/qwen35moe/runtime_verify.cpp
  src/qwen35moe/runtime_dspark_verify.cpp
  src/qwen35moe/runtime_batch_layer.cpp
  src/qwen35moe/runtime_memory.cpp
  src/qwen35moe/runtime_context.cpp
  src/qwen35moe/expert_storage.cpp
  src/qwen35moe/runtime_cache_resize.cpp
  src/qwen35moe/runtime_cache.cpp
  src/qwen35moe/runtime_host_pin.cpp
  src/prefill/native_tc_provider.cpp
  src/qwen35moe/runtime_batch_memory.cpp
  src/qwen35moe/runtime_prefill_metrics.cpp
  src/qwen35moe/runtime_batch.cpp
  src/qwen35moe/runtime_batch_gdn_scan.cpp
  src/qwen35moe/runtime_batch_gdn.cpp
  src/qwen35moe/runtime_batch_attention.cpp
  src/qwen35moe/runtime_attention_prefill.cpp
  src/qwen35moe/runtime_batch_moe.cpp
  src/qwen35moe/runtime_batch_moe_grouped.cpp
  src/qwen35moe/runtime_batch_moe_cpu.cpp
  src/qwen35moe/runtime_batch_moe_mixed.cpp
  src/qwen35moe/runtime_gdn.cpp
  src/qwen35moe/runtime_attention.cpp
  src/qwen35moe/runtime_moe.cpp
  src/qwen35moe/runtime_run.cpp
  src/qwen35moe/runtime_reset.cpp
  src/qwen35moe/runtime_embedding_batch.cpp
  src/qwen35moe/runtime_projection_mmq.cpp
  src/qwen35moe/runtime_moe_down_mmq.cpp
  src/qwen35moe/runtime_moe_gu_mmq.cpp
  src/qwen35moe/runtime_mtp_kv_rows.cpp)
target_include_directories(strata_qwen35moe_runtime PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35moe_runtime PUBLIC strata_qwen35moe strata_qwen35moe_cuda strata_kernels_cpu)
if(NOT TARGET strata_mmq)
  message(FATAL_ERROR "Mini profile requires the mainline MMQ provider")
endif()
target_sources(strata_mmq PRIVATE
  src/prefill/native_q8_mmq.cu
  src/prefill/native_q4_code_sum_mmq.cu
  src/prefill/native_grouped_mmq_provider.cpp
  src/prefill/native_q6_accum_mmq.cu)
target_link_libraries(strata_qwen35moe_runtime PUBLIC strata_mmq)
target_sources(strata PRIVATE src/qwen35moe/minicpm_dspark_engine.cpp)
target_link_libraries(strata PRIVATE strata_qwen35moe_runtime)
target_compile_definitions(strata PRIVATE STRATA_QWEN35_MOE=1)

set_property(SOURCE src/kernels/cuda/native_mmvq.cu APPEND PROPERTY COMPILE_DEFINITIONS STRATA_QWEN35_KERNEL_EXTENSIONS=1)

set_target_properties(strata_qwen35moe_cuda strata_native_sampling PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(strata_qwen35moe_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
# The qualified Dense sampler was compiled without fast-math. Keep the original
# MoE-only recipe, while a combined executable uses one frozen sampler owner.
if(NOT STRATA_QWEN35_DENSE_FINAL223)
  target_compile_options(strata_native_sampling PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
endif()

target_sources(strata_kernels_cpu PRIVATE src/kernels/cpu/native_expert_views.cpp)

target_sources(strata_qwen35moe_runtime PRIVATE src/qwen35moe/runtime_diagnostic_policy.cpp)
