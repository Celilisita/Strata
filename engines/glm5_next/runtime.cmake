# GLM normal-runtime registration shared by embedded and standalone builds.
# This file adds only source modules and their narrow build dependencies.
target_sources(strata_deepseek_v4_runtime PRIVATE ../glm5_next/kernel_activity.cpp)
if(WIN32)
    set_property(SOURCE ../glm5_next/kernel_activity.cpp TARGET_DIRECTORY strata_deepseek_v4_runtime APPEND PROPERTY
        INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}" "${CUDAToolkit_BIN_DIR}/../extras/CUPTI/include")
endif()

target_sources(strata_deepseek_v4_runtime PRIVATE ../glm5_next/mtp_session.cpp ../glm5_next/mtp_runtime.cpp runtime_decode.cpp)
target_sources(strata_deepseek_v4_runtime PRIVATE ../glm5_next/mtp_generate.cpp ../glm5_next/serve_mtp.cpp)
set_property(SOURCE ../glm5_next/mtp_generate.cpp ../glm5_next/serve_mtp.cpp
    TARGET_DIRECTORY strata_deepseek_v4_runtime APPEND PROPERTY
    INCLUDE_DIRECTORIES "${STRATA_LLAMA_SOURCE_DIR}/vendor")
# Capture belongs to the existing cache owner; do not create a reverse static
# library dependency from llama's DeviceExpertCache to the runtime library.
target_sources(llama PRIVATE ../glm5_next/route_capture.cpp)

target_sources(llama PRIVATE ../glm5_next/cold_gpu_owner.cpp)

set_property(SOURCE "${STRATA_LLAMA_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.c" TARGET_DIRECTORY ggml-cpu APPEND PROPERTY
    COMPILE_DEFINITIONS STRATA_GLM53_CPU_PHASE_PROFILE=1)
set_property(SOURCE "${STRATA_LLAMA_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-cpu.c" TARGET_DIRECTORY ggml-cpu APPEND PROPERTY
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|x64)$")
    foreach(source ggml-cpu.c arch/x86/quants.c)
        set_property(SOURCE "${STRATA_LLAMA_SOURCE_DIR}/ggml/src/ggml-cpu/${source}" TARGET_DIRECTORY ggml-cpu APPEND PROPERTY
            COMPILE_DEFINITIONS STRATA_GLM53_CPU_PAIR_DOT=1)
        set_property(SOURCE "${STRATA_LLAMA_SOURCE_DIR}/ggml/src/ggml-cpu/${source}" TARGET_DIRECTORY ggml-cpu APPEND PROPERTY
            INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")
    endforeach()
endif()

# KDA policy uses the same controlled backend header as its CUDA implementation.
set_property(SOURCE "${STRATA_LLAMA_SOURCE_DIR}/src/models/delta-net-base.cpp"
    TARGET_DIRECTORY llama APPEND PROPERTY
    INCLUDE_DIRECTORIES "${STRATA_LLAMA_SOURCE_DIR}/ggml/src")

if(TARGET strata)
    target_sources(strata PRIVATE ../glm5_next/serve_entry.cpp)
endif()
