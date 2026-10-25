#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::glm5_next {
inline constexpr uint64_t partial_down_bridge_magic=0x5044324341503031ull;
inline constexpr int32_t partial_down_mid_tag=0x50444d31;
inline constexpr int partial_down_tag_word=GGML_MAX_OP_PARAMS/sizeof(int32_t)-1;
inline constexpr uint32_t partial_down_stock=1,partial_down_selected=2;
inline constexpr uint32_t partial_down_whole_mask=1088,partial_down_role_mask=33027;
inline constexpr uint32_t partial_down_cpu_mask=22556,partial_down_missing_mask=56671;
inline constexpr int32_t partial_down_keys[2]{57,234};
inline constexpr uint8_t partial_down_mid_rows[4]{0,1,5,9};
inline constexpr uint8_t partial_down_ranks[4]{0,1,8,15};
inline constexpr uint8_t partial_down_cpu_rows[6]{2,3,4,6,7,8};
inline constexpr uint8_t partial_down_cpu_ranks[6]{2,3,4,11,12,14};
inline constexpr int32_t partial_down_cpu_ids[10]{57,234,27,106,256,234,129,27,271,57};
inline constexpr uint8_t partial_down_cpu_positions[10]{0,0,0,0,0,1,1,1,1,1};
inline constexpr uint8_t partial_down_gu_ranks[10]{0,1,2,3,4,8,11,12,14,15};
#ifdef __CUDACC__
#define STRATA_PD_HD __host__ __device__
#else
#define STRATA_PD_HD
#endif
STRATA_PD_HD inline constexpr unsigned partial_down_rank(unsigned i){return i<2?i:i==2?8:15;}
STRATA_PD_HD inline constexpr unsigned partial_down_gu_rank(unsigned i){return i<5?i:i==5?8:i<8?i+5:i+6;}
STRATA_PD_HD inline constexpr int partial_down_cpu_id(unsigned i){
    return i==0||i==9?57:i==1||i==5?234:i==2||i==7?27:i==3?106:i==4?256:i==6?129:271;
}
#undef STRATA_PD_HD

// One mapped, same-owner phase. Each publication word uses system release/acquire.
struct alignas(64) PartialDownPhase {
    uint64_t owner_epoch=0,bank_generation=0;
    uint32_t ticket=0,decision_seq=0,mid_ready_seq=0,mid_consumed_seq=0,error_seq=0;
    uint32_t decision=0,partial_mask=0,cpu_down_mask=0;
    int32_t down_keys[2]{-1,-1};
    uint32_t reader_error=0,reserved=0;
};
// Stored in a real external CUDA buffer. It is not a host-buffer tensor alias.
struct alignas(64) PartialDownBridge {
    uint64_t magic=partial_down_bridge_magic,owner_epoch=0,bank_generation=0;
    uintptr_t phase=0,metadata=0;
    uint32_t first_row=0,layer=24;
    uint64_t reserved[2]{};
};
struct PartialDownGraphBinding {ggml_tensor* bridge=nullptr;ggml_tensor* indices=nullptr;};
static_assert(sizeof(PartialDownPhase)==64&&alignof(PartialDownPhase)==64,"partial phase ABI");
static_assert(sizeof(PartialDownBridge)==64&&alignof(PartialDownBridge)==64,"partial bridge ABI");
static_assert(offsetof(PartialDownPhase,ticket)==16&&offsetof(PartialDownPhase,reader_error)==56,"partial phase offsets");
static_assert(std::is_standard_layout<PartialDownPhase>::value,"partial phase layout");
inline bool partial_down_bridge_tensor(const ggml_tensor* t){
    return t&&t->type==GGML_TYPE_I64&&ggml_nelements(t)==8&&ggml_is_contiguous(t);
}
inline bool partial_down_mid(const ggml_tensor* t){
    return t&&t->op==GGML_OP_GET_ROWS&&t->op_params[partial_down_tag_word]==partial_down_mid_tag;
}
inline bool partial_down_mid_supported(const ggml_tensor* t){
    return partial_down_mid(t)&&t->type==GGML_TYPE_F32&&t->ne[0]==2048&&t->ne[1]==1&&t->ne[2]==4&&t->ne[3]==1&&
        ggml_is_contiguous(t)&&t->src[0]&&t->src[0]->type==GGML_TYPE_F32&&ggml_nelements(t->src[0])==65536&&
        t->src[1]&&t->src[1]->type==GGML_TYPE_I32&&ggml_nelements(t->src[1])==16&&
        partial_down_bridge_tensor(t->src[2])&&t->src[3]&&t->src[3]->type==GGML_TYPE_I32&&ggml_nelements(t->src[3])==1&&
        t->src[4]&&t->src[4]->type==GGML_TYPE_I32&&ggml_nelements(t->src[4])==2;
}
}
