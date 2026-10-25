#pragma once
#include <cstddef>
#include "strata/gemma4/native/vector3.hpp"
namespace strata::gemma4::native::detail {
struct alignas(4) ResidualBlock {float d[3];int sum[3];signed char code[3][32];};
static_assert(sizeof(ResidualBlock)==ResidualVectorBlockBytes);
static_assert(offsetof(ResidualBlock,code)==24&&alignof(ResidualBlock)==4);
}
