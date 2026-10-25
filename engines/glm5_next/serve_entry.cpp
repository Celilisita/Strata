#include "cpu_placement.hpp"
#include <cstdio>
#include <exception>

int strata_deepseek_v4_serve(int argc, char** argv);

int strata_glm5_next_serve(int argc, char** argv) {
    try {
        strata::glm5_next::configure_cpu_process();
        return strata_deepseek_v4_serve(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "GLM CPU placement: %s\n", error.what());
        return 2;
    }
}
