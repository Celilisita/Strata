#include "cache_growth.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using strata::deepseek_v4::converge_cache_growth;
    for (const int safe_limit : {0, 7, 74, 342, 370}) {
        int current = 0;
        std::vector<int> visited;
        const auto result = converge_cache_growth(370,
            [&](int slots) { current = slots; visited.push_back(slots); },
            [&] { return current <= safe_limit; });
        assert(result.added_slots <= safe_limit);
        assert(current == result.added_slots);
        assert(result.attempts <= 8);
        assert(!visited.empty());
        if (safe_limit == 370) assert(result.added_slots == 370);
        if (safe_limit == 342) assert(result.added_slots >= 339);
    }
    std::cout << "cache growth backoff passed\n";
}
