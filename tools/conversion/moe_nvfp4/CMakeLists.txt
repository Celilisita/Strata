cmake_minimum_required(VERSION 3.24)
project(StrataMoeNvfp4Converter LANGUAGES CXX CUDA)
find_package(CUDAToolkit 12.8 REQUIRED)
# Reuse the original converter target unchanged; no Strata runtime or GGML needed.
include("${CMAKE_CURRENT_SOURCE_DIR}/gpu/add_nvfp4_converter.cmake")
