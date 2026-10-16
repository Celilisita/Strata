#pragma once
#include <cstdlib>
#include <cstring>

// Experimental launch geometry only. The dot product, reduction, activation
// quantizer and fusion body remain the canonical templates in mmvq.cu.
inline bool strata_mmvq_wide_rows(ggml_type type, int cc, int columns, int rows, int tokens) {
    const char* option = std::getenv("STRATA_DSV4_MMVQ_WIDE");
    if (!option || std::strcmp(option, "1") != 0 || cc != 1200 || rows % 4 || tokens != 1)
        return false;
    if (type == GGML_TYPE_MXFP4)
        return columns == 2048 && rows == 4096;
    if (type == GGML_TYPE_Q8_0)
        return columns == 1024 && rows == 32768;
    return false;
}
