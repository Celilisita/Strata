#include "cache_cost_audit.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
bool cache_cost_audit_requested() {
    const char* v = std::getenv("STRATA_DSV4_CACHE_COST_TRACE");
    if (!v || !std::strcmp(v, "0")) return false;
    if (!std::strcmp(v, "1")) return true;
    throw std::runtime_error("cache cost trace must be0 or1");
}
CacheCostAudit::CacheCostAudit(size_t keys) : history_(keys) {
    if (!keys || keys > 64*288) throw std::runtime_error("expert fill audit exceeds its fixed bound");
}
void CacheCostAudit::plan() { ++costs_.plans; }
void CacheCostAudit::fill(int key, int evicted, uint32_t incoming, uint32_t outgoing) {
    if (key < 0 || size_t(key) >= history_.size() || evicted < -1 || (evicted >= 0 && size_t(evicted) >= history_.size()))
        throw std::runtime_error("D4 fill audit key differs");
    ++costs_.planned_fills;
    auto& h = history_[key];
    if (h.last_fill) {
        ++costs_.repeated_fills;
        if (h.last_eviction && costs_.plans - h.last_eviction <= 43) ++costs_.refills_within43_miss_plans;
    }
    h.last_fill = costs_.plans;
    if (evicted >= 0) {
        ++costs_.planned_evictions;
        history_[evicted].last_eviction = costs_.plans;
        costs_.incoming_score_not_greater += incoming <= outgoing;
    }
}
}
