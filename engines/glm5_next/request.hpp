#pragma once
#include "../deepseek_v4/runtime.hpp"
#include <nlohmann/json.hpp>

namespace strata::glm5_next {
using Json = nlohmann::ordered_json;
Json generate(deepseek_v4::Runtime&, const Json& request);
}
