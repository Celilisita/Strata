# Shared, controlled dependency defaults. Call before model targets are created.
# Explicit caller paths remain supported; no prebuilt object/archive is imported.
set(_strata_controlled_backend "${CMAKE_CURRENT_LIST_DIR}/../third_party/canonical")
if(EXISTS "${_strata_controlled_backend}/src/llama-moe-cache.cpp")
  set(STRATA_LLAMA_SOURCE_DIR "${_strata_controlled_backend}" CACHE PATH "Pinned shared backend source")
  set(STRATA_GGML_DIR "${_strata_controlled_backend}" CACHE PATH "Shared quantized operator source")
  set(STRATA_DENSE_BACKEND_SOURCE "${_strata_controlled_backend}" CACHE PATH "Shared dense vision source")
endif()
set(STRATA_CUTLASS_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party/cutlass" CACHE PATH "Pinned CUTLASS headers")
unset(_strata_controlled_backend)
