#pragma once
#include "strata/prefill/native_grouped_mmq.hpp"
#include "strata/prefill/moe_mmq.hpp"
namespace strata::prefill::mmq {
// Returns the measured provider for a grouped quant format and GU/Down role.
// Without an explicit backend selection, retain canonical register arithmetic.
void run_blackwell_dense_int8(int type,const void* weights,const void* raw_q8,void* packed,
    size_t packed_bytes,float* output,int width,int rows,int outputs,void* stream);
void run_blackwell_dense_group_f16(int type,const void* weights,const void* raw_q8,void* codes,
    size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,int width,int rows,int outputs,void* stream);
// Explicit tile entrypoint is for bounded paired operator gates; runtime reads the opt-in once.
void run_blackwell_dense_group_f16_tile(int type,const void* weights,const void* raw_q8,void* codes,
    size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,int width,int rows,int outputs,
    void* stream,int tile_rows);
struct BlackwellDenseResources {
    int registers=0; size_t shared_bytes=0,local_bytes=0; int max_threads=0,active_blocks=0;
};
BlackwellDenseResources blackwell_dense_resources(int type,int width,int tile_rows);
// Dense group INT8 is independent of the routed backend and defaults off.
// The bounded runtime domain uses1024..2048-row pieces and tile128 only.
bool blackwell_dense_group_int8_enabled(int type,int rows,int width,int outputs);
void run_blackwell_dense_group_int8_tile(int type,const void* weights,const void* raw_q8,void* codes,
    size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,int width,int rows,int outputs,
    void* stream,int tile_rows);
BlackwellDenseResources blackwell_dense_group_int8_resources(int type,int width,int tile_rows);
// Explicit I16 control/I32/I64 gate entry; runtime wide policy defaults to I16.
void run_blackwell_dense_wide_int8_tile(int type,const void* weights,const void* raw_q8,void* codes,
    size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,int width,int rows,int outputs,
    void* stream,int output_rows);
BlackwellDenseResources blackwell_dense_wide_int8_resources(int type,int width,int output_rows);
void run_blackwell_dense_swizzle_int8_tile(int type,const void* weights,const void* raw_q8,
    void* codes,size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,
    int width,int rows,int outputs,void* stream);
BlackwellDenseResources blackwell_dense_swizzle_int8_resources(int type,int width);
// I128 gate resource shared_bytes includes the opted-in dynamic region.
void run_blackwell_dense_i128_int8_tile(int type,const void* weights,const void* raw_q8,void* codes,
    size_t code_bytes,void* metadata,size_t metadata_bytes,float* output,int width,int rows,int outputs,void* stream);
BlackwellDenseResources blackwell_dense_i128_int8_resources(int type,int width);
int blackwell_dense_tile_rows(int type,int rows);
bool blackwell_q5_tc_enabled();
int blackwell_mmq_backend(int type, int parts);
struct NativeMmqEncodedInput {
    const void* codes = nullptr;
    const void* metadata = nullptr;
    const int32_t* source_ids = nullptr;
    int source_rows = 0;
    // Carry the host plan geometry into the product; never re-read an env flag.
    int tile_rows = 64;
};
struct NativeBlackwellGroupResources {
    int registers = 0, threads = 256, active_blocks_per_sm = 0;
    size_t local_bytes = 0, static_shared_bytes = 0, dynamic_shared_bytes = 0;
};
// Bounded operator-gate query, with no production logging or allocation.
NativeBlackwellGroupResources blackwell_group_resources(int type, int width, int tile_rows,
    bool active = true, int panel_tokens = 0);
// Independent paired Q4 GU group arithmetic; same backend6 compact plan/arena.
void validate_blackwell_gu_int8_views(const void* q8, void* controls, size_t control_bytes,
    void* packed, size_t packed_bytes, float* output, int entries, int source_rows, void* stream);
NativeMmqEncodedInput encode_blackwell_gu_int8_input(const void* q8, void* packed, size_t packed_bytes,
    int source_rows, const int32_t* source_ids, void* stream);
void run_blackwell_gu_int8(const NativeMmqMatrix* matrices, int groups, const int32_t* output_ids,
    float* output, int entries, int max_rows, void* stream, const NativeMmqTile* tiles,
    int tile_count, const NativeMmqEncodedInput& encoded);
NativeBlackwellGroupResources blackwell_gu_int8_resources(int tile_rows, bool active = true);
// I32/K256 product; reuses the I16 canonical input encoder and compact arena.
void run_blackwell_gu_int8_i32(const NativeMmqMatrix* matrices,int groups,const int32_t* output_ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded);
NativeBlackwellGroupResources blackwell_gu_int8_i32_resources(int tile_rows,bool active=true);
void run_blackwell_gu_int8_i64(const NativeMmqMatrix* matrices,int groups,const int32_t* output_ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded);
NativeBlackwellGroupResources blackwell_gu_int8_i64_resources(int tile_rows,bool active=true);
// Down-only exact K32 IMMA; explicit finite gate API, runtime opt-in is separate.
void validate_blackwell_down_int8_views(int type,const void* q8,void* controls,size_t control_bytes,
    void* packed,size_t packed_bytes,float* output,int entries,int source_rows,void* stream);
NativeMmqEncodedInput encode_blackwell_down_int8_input(int type,const void* q8,void* packed,
    size_t packed_bytes,int source_rows,const int32_t* source_ids,void* stream);
void run_blackwell_down_int8(int type,const NativeMmqMatrix* matrices,int groups,const int32_t* output_ids,
    float* output,int entries,int max_rows,void* stream,const NativeMmqTile* tiles,
    int tile_count,const NativeMmqEncodedInput& encoded);
NativeBlackwellGroupResources blackwell_down_int8_resources(int type,int tile_rows,bool active=true);
size_t blackwell_input_bytes(int type, int width, int rows);
NativeMmqEncodedInput repack_blackwell_input(const void* source, void* packed, int type,
    int width, int rows, const int32_t* source_ids, int backend, void* stream);
struct BlackwellFusedInputResources {
    int registers=0, threads=256, active_blocks_per_sm=0;
    size_t shared_bytes=0, local_bytes=0;
};
BlackwellFusedInputResources blackwell_swiglu_input_resources(bool diagnostic = false);
// Backend6 Q4/Q5 only. Diagnostic Q8 output is reserved for operator gates.
NativeMmqEncodedInput encode_blackwell_swiglu_input(const float* gate, const float* up,
    void* packed, size_t packed_bytes, int type, int width, int source_rows,
    const int32_t* source_ids, void* stream, void* diagnostic_q8 = nullptr);
void run_blackwell_grouped(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* output_ids, const void* packed, float* output,
    int rows, int max_rows, int width, int outputs, void* stream,
    const NativeMmqTile* tiles, int tile_count, int backend, const NativeMmqEncodedInput& encoded);
}
