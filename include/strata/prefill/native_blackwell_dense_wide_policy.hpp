#pragma once
#include "strata/prefill/native_blackwell_dense_int8_policy.hpp"
#include <cstring>
#include <stdexcept>

namespace strata::prefill::mmq {
struct DenseWidePolicy { int q4 = 16, q5 = 16; };
inline DenseWidePolicy blackwell_dense_wide_policy(const char* value) {
    if (!value || !std::strcmp(value,"0")) return {};
    if (!std::strcmp(value,"32")) return {32,32};
    if (!std::strcmp(value,"64")) return {64,64};
    if (!std::strcmp(value,"12:32")) return {32,16};
    if (!std::strcmp(value,"12:64")) return {64,16};
    if (!std::strcmp(value,"13:32")) return {16,32};
    if (!std::strcmp(value,"13:64")) return {16,64};
    if (!std::strcmp(value,"12:32,13:32")) return {32,32};
    if (!std::strcmp(value,"12:32,13:64")) return {32,64};
    if (!std::strcmp(value,"12:64,13:32")) return {64,32};
    if (!std::strcmp(value,"12:64,13:64")) return {64,64};
    throw std::invalid_argument("Dense INT8 output rows require0,32,64 or canonical12:32/64,13:32/64");
}
inline int blackwell_dense_wide_rows(DenseWidePolicy policy, int type, int rows, int width, int outputs) {
    if (!blackwell_dense_int8_selected((1u<<12)|(1u<<13),type,rows,width,outputs)) return 16;
    return type == 12 ? policy.q4 : policy.q5;
}
} // namespace strata::prefill::mmq
