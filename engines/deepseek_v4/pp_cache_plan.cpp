#include "pp_cache_plan.hpp"

#include <cstdlib>
#include <cstring>

namespace strata::deepseek_v4 {

bool pp_cache_bank_eligible(int64_t tokens, int64_t experts, int64_t slots) {
    static const bool enabled = [] {
        const char* value = std::getenv("STRATA_DSV4_PP_CACHE_BANKS");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled && tokens > 32 && experts > 0 && slots >= experts;
}

}
