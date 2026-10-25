#pragma once
#include "ggml-backend.h"
namespace strata::deepseek_v4 {
void phase_swap_rows_mixed(ggml_backend_t,void*,size_t,void*,size_t,void*,size_t,int,int);
void phase_swap_rows(ggml_backend_t,void*,void*,void*,size_t,int,int);
}
