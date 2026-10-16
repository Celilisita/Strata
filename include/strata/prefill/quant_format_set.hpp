#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::prefill {
// A measured provider's format policy. Formats come from tensor metadata;
// this contains no model equations, weight ownership or device transfers.
class QuantFormatSet {
public:
    bool contains(int type) const noexcept {
        return type >= 0 && type < 64 && (mask_ & (uint64_t(1) << type));
    }
    static QuantFormatSet parse(const std::string& csv) {
        QuantFormatSet result; result.mask_ = 0;
        if (csv.empty() || csv.back() == ',') throw std::invalid_argument("empty quant format policy");
        size_t first = 0;
        while (first < csv.size()) {
            const size_t end = csv.find(',', first);
            const std::string field = csv.substr(first, end == std::string::npos ? end : end - first);
            size_t used = 0;
            const int type = std::stoi(field, &used);
            if (used != field.size() || type < 0 || type >= 64)
                throw std::invalid_argument("invalid quant format policy");
            result.mask_ |= uint64_t(1) << type;
            if (end == std::string::npos) break;
            first = end + 1;
        }
        return result;
    }
private:
    uint64_t mask_ = ~uint64_t(0);
};
}
