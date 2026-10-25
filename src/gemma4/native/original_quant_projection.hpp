#pragma once
#include <cstddef>
namespace strata::gemma4::native {
// One F32-to-Q8 activation arena, model-owned; weights stay raw GGUF blocks.
std::size_t original_projection_pool_bound_bytes();
std::size_t original_projection_input_bytes(int batch,int columns);
}
