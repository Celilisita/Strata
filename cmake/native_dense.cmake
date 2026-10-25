if(NOT STRATA_QWEN35_DENSE)
  return()
endif()
add_library(strata_qwen35 STATIC
  src/qwen35/spec.cpp
  src/qwen35/spec_bind.cpp)
target_include_directories(strata_qwen35 PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35 PUBLIC strata_artifact)
add_library(strata_qwen35_native STATIC
  src/qwen35/native/ffn.cpp
  src/qwen35/native/graph.cpp
  src/qwen35/native/ffn_gemm.cpp
  src/qwen35/native/swiglu_half.cu
  src/qwen35/native/swiglu_pair.cu
  src/qwen35/native/attention_gemm.cu
  src/qwen35/native/gdn_rows.cu
  src/qwen35/native/recurrent.cpp
  src/qwen35/native/q8_projection.cu
  src/qwen35/native/recurrent_graph.cpp
  src/qwen35/native/attention_layer.cpp
  src/qwen35/native/weights.cpp
  src/qwen35/native/model.cpp
  src/qwen35/native/model_visual.cpp
  src/qwen35/native/model_cache.cpp
  src/qwen35/native/mtp_position.cu
  src/qwen35/native/tiny_batch.cu
  src/qwen35/native/cache_q8.cu
  src/qwen35/native/attention_cache_only.cpp
  src/qwen35/native/token_ops.cu
  src/qwen35/native/elementwise.cu
  src/qwen35/native/tiled.cu
  src/qwen35/native/dense.cu
  src/qwen35/native/small_prefill_encode_half.cu
  src/qwen35/native/small_prefill_encode_vector.cu
  src/qwen35/native/small_prefill_vector.cu
  src/qwen35/native/small_prefill_direct.cu
  src/qwen35/native/small_prefill_tensor.cu
  src/qwen35/native/small_prefill_scaled.cu
  src/qwen35/native/small_prefill.cu
  src/qwen35/native/vector.cu
  src/qwen35/native/prepack.cu
  src/qwen35/native/gdn.cu
  src/qwen35/native/ops.cu
  src/qwen35/native/small_prefill_q4_grouped.cu
  src/qwen35/native/attention_ops.cu
  src/qwen35/native/attention_decode.cu
  src/qwen35/native/attention_prefill.cu
  src/qwen35/native/attention_verify.cu
  src/qwen35/native/model_head_batch.cpp
  src/qwen35/native/mtp_draft_link.cu
  src/qwen35/native/kv_ram.cpp
  src/qwen35/native/kv_ram_write.cu
  # final223 ram_block_sort.cu owns ram_select / ram_sort_workspace and fallback.
  src/qwen35/native/attention_ram_decode.cu
  src/qwen35/native/attention_ram_verify.cu
  src/qwen35/native/cache_window.cu
  src/qwen35/native/attention_window_decode.cu
  src/qwen35/native/attention_window_verify.cu
  src/qwen35/native/attention_window_prefill.cu
  src/qwen35/native/tiny_rows.cu
  src/qwen35/native/tiny_meta.cu
  src/qwen35/native/gdn_prefix.cu
  src/qwen35/native/mtp_prefix.cpp
  src/qwen35/native/weight_overlap.cpp
  src/qwen35/native/attention_wide.cu
  src/qwen35/native/attention_blocked.cu
  src/qwen35/native/attention_online.cu
  src/qwen35/native/attention_skew.cu
  src/qwen35/native/attention_reuse.cu
  src/qwen35/native/attention_history.cu
  src/qwen35/native/attention_verify_padded.cu
  src/qwen35/native/attention_verify_grouped.cu)
target_include_directories(strata_qwen35_native PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35_native PUBLIC strata_qwen35 CUDA::cudart)
add_library(strata_qwen35_cublas STATIC
  src/qwen35/native/cublas_gemm.cpp)
target_include_directories(strata_qwen35_cublas PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35_cublas PUBLIC strata_qwen35_native CUDA::cublas)
add_library(strata_qwen35_cutlass_swiglu STATIC
  src/qwen35/native/cutlass_swiglu_gemm.cu)
target_include_directories(strata_qwen35_cutlass_swiglu PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35_cutlass_swiglu PUBLIC strata_qwen35_cublas)
add_library(strata_qwen35_gdn_vectorized STATIC
  src/qwen35/native/gdn_vectorized.cu)
target_include_directories(strata_qwen35_gdn_vectorized PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35_gdn_vectorized PUBLIC strata_qwen35_native)
add_library(strata_qwen35_dense_runtime STATIC
  src/qwen35/native/mtp_weights.cpp
  src/qwen35/native/mtp_condition.cpp
  src/qwen35/native/mtp_round.cpp
  src/qwen35/native/mtp_round_gpu.cpp
  src/qwen35/native/mtp_graph.cpp
  src/qwen35/native/mtp_graph_operation.cpp
  src/qwen35/native/attention_gather.cu
  src/qwen35/native/attention_keys64.cu)
target_include_directories(strata_qwen35_dense_runtime PUBLIC include PRIVATE third_party/ggml)
target_link_libraries(strata_qwen35_dense_runtime PUBLIC strata_qwen35_cutlass_swiglu strata_qwen35_gdn_vectorized strata_native_sampling)
set(STRATA_CUTLASS_DIR "" CACHE PATH "Pinned CUTLASS source")
if(NOT EXISTS "${STRATA_CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set STRATA_CUTLASS_DIR to the existing pinned headers")
endif()
target_include_directories(strata_qwen35_cutlass_swiglu PRIVATE "${STRATA_CUTLASS_DIR}/include")
target_sources(strata PRIVATE src/qwen35/native/serve.cpp)
target_link_libraries(strata PRIVATE strata_qwen35_dense_runtime)
target_compile_definitions(strata PRIVATE STRATA_QWEN35_DENSE=1)
option(STRATA_NATIVE_VISUAL_CHECK "Build the focused native visual boundary check" OFF)
if(STRATA_NATIVE_VISUAL_CHECK)
  add_executable(strata-native-visual-check tools/native_visual_check.cpp)
  target_link_libraries(strata-native-visual-check PRIVATE strata_qwen35_native)
endif()

set_target_properties(strata_qwen35_cutlass_swiglu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

set_target_properties(strata_qwen35_native strata_qwen35_dense_runtime strata_qwen35_gdn_vectorized PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)

# Source ownership for the frozen provider is registered after the base targets.
include(${CMAKE_CURRENT_LIST_DIR}/native_dense_extensions.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/native_dense_fp4.cmake)
