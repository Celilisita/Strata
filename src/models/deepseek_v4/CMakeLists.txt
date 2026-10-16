add_library(strata_deepseek_v4_residual STATIC hyper_connections.cpp)
target_include_directories(strata_deepseek_v4_residual PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata_deepseek_v4_residual PRIVATE strata_warnings)

# The native model is not registered as supported until real PP/TG and vision pass.
# This target validates the first residual slice from exported reference checkpoints.
add_executable(strata-deepseek-v4-residual-bench residual_bench.cpp)
target_link_libraries(strata-deepseek-v4-residual-bench PRIVATE strata_deepseek_v4_residual)
