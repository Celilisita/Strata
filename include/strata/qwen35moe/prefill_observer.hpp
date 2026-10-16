#pragma once

namespace strata::qwen35moe {
// Borrowed request observer. Reports only layers known complete at existing
// route/head drains; adds no device events, synchronization or model ownership.
struct PrefillObserver {
    void* context = nullptr;
    void (*report)(void*, int, int) = nullptr;
    void notify(int completed, int total) const {
        if (report) report(context, completed, total);
    }
};
} // namespace strata::qwen35moe
