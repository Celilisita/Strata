#pragma once
#include "ggml.h"
#include "cpu_row_dot.h"
#include "cpu_vnni_dot.h"
#include "cpu_iq4_packed.h"
#include "cpu_iq4_down_pair.h"
#include "cpu_r16_down_schedule.h"
#include "cpu_gu_fusion.h"
#include "cpu_iq3_dual_lut.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
// Backend-only opt-in. Only two actual columns of an IQ3_S expert share its
// quant decoding; each activation retains the canonical independent sum.
struct strata_glm_cpu_dot_options { bool pair, wide, graph_snapshot, static_down, row_dot, vnni_dot, pair4, iq4_packed, iq4_down_pair, pair_schedule, gu_fusion, gu_dual_lut; };
static inline bool strata_glm_cpu_dot_env(const char* name) {
    const char* flag=getenv(name);return flag&&strcmp(flag,"1")==0;
}
static inline bool strata_glm_cpu_pair_dot(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options) {
    if(bank->type!=GGML_TYPE_IQ3_S||bank->ne[2]!=288||ids->ne[0]!=1||ids->ne[1]<2||ids->ne[1]>16||
       !((bank->ne[0]==4096&&bank->ne[1]==2048)||(bank->ne[0]==2048&&bank->ne[1]==4096)))return false;
    return options->graph_snapshot?options->pair:strata_glm_cpu_dot_env("STRATA_GLM53_CPU_PAIR_DOT");
}
void ggml_vec_dot_iq3_s_q8_K_pair(int n,float* s0,float* s1,const void* x,const void* y0,const void* y1);

int ggml_vec_dot_iq3_s_q8_K_wide_available(void);
void ggml_vec_dot_iq3_s_q8_K_wide(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_iq3_s_q8_K_wide_pair(int,float*,float*,const void*,const void*,const void*);
static inline struct strata_glm_cpu_dot_options strata_glm_cpu_dot_options_from_env(void) {
    // Caller reads once per graph before publishing existing pool work. Exact1
    // opts in; unset/0/other values retain the original worker getenv behavior.
    struct strata_glm_cpu_dot_options options={false,false,false,false,false,false,false,false,false,false,false,false};
    options.graph_snapshot=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_GRAPH_FLAGS");
    options.static_down=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_STATIC_DOWN");
    if(options.graph_snapshot) {
        options.pair=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_PAIR_DOT");
        options.wide=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_IQ3_WIDE")&&ggml_vec_dot_iq3_s_q8_K_wide_available();
        options.row_dot=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_ROW_DOT")&&strata_glm_cpu_row_dot_available();
        options.vnni_dot=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_VNNI_DOT")&&strata_glm_cpu_vnni_capabilities()==7;
        options.pair4=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_PAIR4_DOT");
        options.iq4_packed=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_IQ4_PACKED_DECODE")&&strata_glm_cpu_iq4_packed_available();
        options.iq4_down_pair=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_IQ4_DOWN_PAIR")&&strata_glm_cpu_iq4_down_pair_available();
        options.pair_schedule=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_PAIR_SCHEDULE");
        options.gu_fusion=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_GU_FUSION");
        options.gu_dual_lut=strata_glm_cpu_dot_env("STRATA_GLM53_CPU_GU_DUAL_LUT");
    }
    return options;
}
static inline bool strata_glm_cpu_pair4_shape(const struct ggml_tensor* bank,const struct ggml_tensor* ids,int threads) {
    return threads==16&&bank->type==GGML_TYPE_IQ3_S&&bank->ne[0]==4096&&bank->ne[1]==2048&&
        bank->ne[2]==288&&bank->ne[3]==1&&ggml_is_contiguous(bank)&&ids->ne[0]==1&&
        ids->ne[1]>=17&&ids->ne[1]<=32&&ids->ne[2]==1&&ids->ne[3]==1;
}
static inline bool strata_glm_cpu_pair4_enabled(const struct strata_glm_cpu_dot_options* options) {
    return options->graph_snapshot&&options->pair4&&options->pair&&!options->wide&&!options->row_dot&&!options->vnni_dot;
}
static inline bool strata_glm_cpu_vnni_dot_shape(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options,int threads) {
    return options->graph_snapshot&&options->vnni_dot&&options->pair&&!options->wide&&!options->row_dot&&threads==16&&
        bank->type==GGML_TYPE_IQ3_S&&bank->ne[2]==288&&bank->ne[3]==1&&
        ((bank->ne[0]==4096&&bank->ne[1]==2048)||(bank->ne[0]==2048&&bank->ne[1]==4096))&&
        ids->ne[0]==1&&ids->ne[1]>=2&&ids->ne[1]<=16&&ids->ne[2]==1&&ids->ne[3]==1;
}
static inline bool strata_glm_cpu_row_dot_shape(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options,int threads) {
    return options->graph_snapshot&&options->row_dot&&threads==16&&bank->type==GGML_TYPE_IQ3_S&&
        bank->ne[0]==4096&&bank->ne[1]==2048&&bank->ne[2]==288&&bank->ne[3]==1&&
        ids->ne[0]>=1&&ids->ne[0]<=8&&ids->ne[1]>=1&&ids->ne[0]*ids->ne[1]<=16&&ids->ne[2]==1&&ids->ne[3]==1;
}
static inline bool strata_glm_cpu_static_down(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options,int threads) {
    return options->static_down&&threads==16&&bank->ne[0]==2048&&bank->ne[1]==4096&&
        bank->ne[2]==288&&bank->ne[3]==1&&ids->ne[0]==1&&ids->ne[1]>=1&&ids->ne[1]<=16&&
        ids->ne[2]==1&&ids->ne[3]==1&&(bank->type==GGML_TYPE_IQ4_XS||bank->type==GGML_TYPE_Q6_K);
}
static inline bool strata_glm_cpu_iq3_wide(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options) {
    if(bank->type!=GGML_TYPE_IQ3_S||bank->ne[2]!=288||ids->ne[0]<1||ids->ne[0]>8||ids->ne[1]<1||ids->ne[0]*ids->ne[1]>16||
       !((bank->ne[0]==4096&&bank->ne[1]==2048)||(bank->ne[0]==2048&&bank->ne[1]==4096)))return false;
    return options->graph_snapshot?options->wide:
        strata_glm_cpu_dot_env("STRATA_GLM53_CPU_IQ3_WIDE")&&ggml_vec_dot_iq3_s_q8_K_wide_available();
}

static inline bool strata_glm_cpu_iq4_packed_shape(const struct ggml_tensor* bank,const struct ggml_tensor* ids,
        const struct strata_glm_cpu_dot_options* options,int threads) {
    return options->graph_snapshot&&options->iq4_packed&&threads==16&&bank->type==GGML_TYPE_IQ4_XS&&
        bank->ne[0]==2048&&bank->ne[1]==4096&&bank->ne[2]==288&&bank->ne[3]==1&&ggml_is_contiguous(bank)&&
        ids->ne[0]==1&&ids->ne[1]>=1&&ids->ne[1]<=16&&ids->ne[2]==1&&ids->ne[3]==1;
}

// Original Down geometry only. Pair sharing does not alter its old chunk plan.
static inline bool strata_glm_cpu_iq4_down_pair_shape(const struct ggml_tensor * bank,
        const struct ggml_tensor * ids, const struct strata_glm_cpu_dot_options * options, int threads) {
    return options->graph_snapshot && options->iq4_down_pair && !options->iq4_packed && threads == 16 &&
        bank->type == GGML_TYPE_IQ4_XS && bank->ne[0] == 2048 && bank->ne[1] == 4096 &&
        bank->ne[2] == 288 && bank->ne[3] == 1 && ggml_is_contiguous(bank) &&
        ids->ne[0] == 1 && ids->ne[1] >= 2 && ids->ne[1] <= 16 && ids->ne[2] == 1 && ids->ne[3] == 1;
}
