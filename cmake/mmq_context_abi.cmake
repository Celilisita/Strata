# The shared CUDA context contains graph-dependent fields before its pools.
# Match the actual backend definitions; otherwise inline constructors/pool()
# from different consumers can be merged with incompatible object layouts.
if(STRATA_DEEPSEEK_V4 AND TARGET ggml-cuda)
  get_target_property(_strata_backend_defs ggml-cuda COMPILE_DEFINITIONS)
  foreach(_definition IN LISTS _strata_backend_defs)
    if(_definition MATCHES "^GGML_CUDA_USE_GRAPHS(=.*)?$")
      target_compile_definitions(strata_mmq PRIVATE "${_definition}")
    endif()
  endforeach()
  unset(_definition)
  unset(_strata_backend_defs)
endif()
