#include "strata/qwen35moe/runtime.hpp"
#include <stdexcept>
namespace strata::qwen35moe {
void Runtime::check_sample_pick() {
    throw std::logic_error("CPU distribution oracle is excluded from production targets");
}
double Runtime::batch_stage_error(const std::string&, const char*, const float*, size_t) {
    throw std::logic_error("batch oracle files are excluded from production targets");
}
}
