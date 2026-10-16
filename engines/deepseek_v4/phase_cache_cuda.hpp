#pragma once
#include "ggml-backend.h"
namespace strata::deepseek_v4 {
void phase_swap_rows(ggml_backend_t,void*,void*,void*,size_t,int,int);
}
