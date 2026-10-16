#pragma once
#include "strata/qwen35moe/runtime.hpp"
#include <utility>

namespace strata::qwen35moe {
// Borrow a parked sequence's KV owners while dispatching the same canonical
// attention operator. Queued kernels retain pointer values; this moves no data.
struct SessionCacheView {
    DeviceBuffer& key;
    DeviceBuffer& value;
    DeviceBuffer* other_key;
    DeviceBuffer* other_value;
    SessionCacheView(DeviceBuffer& k, DeviceBuffer& v, DeviceBuffer* ok = nullptr, DeviceBuffer* ov = nullptr)
        : key(k), value(v), other_key(ok), other_value(ov) { exchange(); }
    ~SessionCacheView() { exchange(); }
    SessionCacheView(const SessionCacheView&) = delete;
    SessionCacheView& operator=(const SessionCacheView&) = delete;
private:
    void exchange() {
        if (other_key) { std::swap(key, *other_key); std::swap(value, *other_value); }
    }
};
} // namespace strata::qwen35moe
