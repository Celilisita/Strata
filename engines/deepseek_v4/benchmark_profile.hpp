#pragma once

#include "profile_scope.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace strata::deepseek_v4 {

// Capture a bounded TG window in a full request. Greedy already drains the
// preceding logits; this observer adds no synchronization or model work.
class TgProfileWindow {
public:
    TgProfileWindow(bool measured, int predict) {
        if (!measured) return;
        const char* value = std::getenv("STRATA_DSV4_PROFILE_TG_TOKENS");
        if (!value) return;
        if (!*value || std::strlen(value) > 2)
            throw std::runtime_error("STRATA_DSV4_PROFILE_TG_TOKENS must be in1..32");
        char* end = nullptr;
        const long count = std::strtol(value, &end, 10);
        const char* enabled = std::getenv("STRATA_DSV4_PROFILE");
        if (*end || count < 1 || count > 32 || count >= predict ||
            !enabled || std::strcmp(enabled, "1") != 0)
            throw std::runtime_error("TG capture needs PROFILE=1 and1..32 decode steps below n-predict");
        tokens_ = int(count);
    }

    bool enabled() const noexcept { return tokens_ != 0; }

    void after_greedy(int index) {
        if (!enabled()) return;
        if (index == 0) capture_ = std::make_unique<ProfileScope>("STRATA_DSV4_TG_WINDOW");
        if (index == tokens_) capture_.reset();
    }

private:
    int tokens_ = 0;
    std::unique_ptr<ProfileScope> capture_;
};

}
