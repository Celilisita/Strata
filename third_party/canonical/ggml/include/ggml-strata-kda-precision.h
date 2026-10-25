#pragma once
#include "ggml.h"

// Private MUL_MAT-only graph contract. Slots 0/1/3 remain accumulator
// precision, hint and source precision. Slot 7 is unused by MUL_MAT upstream.
// Other op types (including PAD, which uses slot 7) never match this tag.
#define GGML_STRATA_KDA_F32_SLOT 7
#define GGML_STRATA_KDA_F32_MAGIC 0x4b444633

static inline bool ggml_strata_kda_f32_tagged(const struct ggml_tensor * node) {
    return node && node->op == GGML_OP_MUL_MAT &&
        node->op_params[GGML_STRATA_KDA_F32_SLOT] == GGML_STRATA_KDA_F32_MAGIC;
}

static inline bool ggml_strata_kda_f32_requested(const struct ggml_tensor * node) {
    return ggml_strata_kda_f32_tagged(node) && node->op_params[0] == GGML_PREC_F32 &&
        node->type == GGML_TYPE_F32 && node->src[0] && node->src[1] &&
        node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32;
}

static inline void ggml_strata_kda_mark_f32(struct ggml_tensor * node) {
    GGML_ASSERT(node && node->op == GGML_OP_MUL_MAT && node->type == GGML_TYPE_F32);
    GGML_ASSERT(node->src[0] && node->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(node->src[1] && node->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(node->op_params[GGML_STRATA_KDA_F32_SLOT] == 0 || ggml_strata_kda_f32_tagged(node));
    GGML_ASSERT(ggml_prec_set_acc(node, GGML_PREC_F32));
    node->op_params[GGML_STRATA_KDA_F32_SLOT] = GGML_STRATA_KDA_F32_MAGIC;
}
