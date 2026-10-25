#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
namespace strata::prefill::mmq {
struct NativeMmqMatrix {
    uint64_t gate = 0, up = 0;
    int32_t begin = 0, end = 0;
};
struct NativeMmqTile { int32_t matrix = 0, tile = 0; };
// Canonical Q8 codes/scales, gathered once by route entry; no requantization.
void repack_native_q8_grouped(const void* input, void* output, int type, int cols,
    int source_rows, int rows, const int32_t* source_ids, void* stream);
struct NativeMmqEncodedInput;
// One tiled launch for arbitrary resident quantized matrices and scatter rows.
void native_grouped_mmq(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* output_ids, const void* packed, float* output,
    int rows, int max_rows, int cols, int outputs, void* stream,
    const NativeMmqTile* tiles = nullptr, int tile_count = 0, const NativeMmqEncodedInput* encoded = nullptr);
struct NativeGroupedMmqWork {
    size_t control_bytes = 0, packed_bytes = 0;
    int groups = 0, launches = 0, scheduled_tiles = 0, active_tiles = 0;
    int max_rows = 0, rectangular_tiles = 0, entries = 0; // Pure host plan costs; no CUDA/lease state.
};
struct NativeMmqPlan;
class NativeGroupedMmqProvider {
public:
    explicit NativeGroupedMmqProvider(void* stream);
    ~NativeGroupedMmqProvider();
    NativeGroupedMmqWork run(int type, const unsigned long long* first, const unsigned long long* second,
        const int32_t* source_ids, int entries, int source_rows,
        const void* q8, void* controls, size_t controls_bytes,
        void* packed, size_t packed_bytes, float* output, int cols, int outputs, bool active_tiles = false);
    // False means no output/control/packed write; caller must retain native quant+run fallback.
    bool try_run_down_f32(int type,const unsigned long long* down,const int32_t* source_ids,
        int entries,int source_rows,const float* input,void* controls,size_t controls_bytes,
        void* packed,size_t packed_bytes,float* output,NativeGroupedMmqWork& work);
    // Same Down plan/lease/product; one fused canonical activation encoder.
    NativeGroupedMmqWork run_swiglu_down(int type, const unsigned long long* weights,
        const int32_t* source_ids, int entries, int source_rows, const float* gate, const float* up,
        void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
        float* output, int cols, int outputs, bool active_tiles = false);
    // Explicit finite operator-gate entry. Runtime run() reads a strict opt-in.
    NativeGroupedMmqWork run_gu_int8(const unsigned long long* gate, const unsigned long long* up,
        const int32_t* source_ids, int entries, int source_rows, const void* q8,
        void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
        float* output, bool active_tiles = false);
    NativeGroupedMmqWork run_gu_int8_i32(const unsigned long long* gate,const unsigned long long* up,
        const int32_t* source_ids,int entries,int source_rows,const void* q8,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
        float* output,bool active_tiles=false);
    NativeGroupedMmqWork run_gu_int8_i64(const unsigned long long* gate,const unsigned long long* up,
        const int32_t* source_ids,int entries,int source_rows,const void* q8,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
        float* output,bool active_tiles=false);
    // Once-encoded full GU plan; callback makes each whole matrix wave ready.
    // Submissions retain global matrix/row IDs and the complete output stride.
    NativeGroupedMmqWork run_gu_pipeline(const unsigned long long* gate,const unsigned long long* up,
        const int32_t* source_ids,int entries,int source_rows,const void* q8,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,float* output,
        int matrices_per_wave,const std::function<void(const NativeMmqMatrix*,int,bool)>& ready);
    // Explicit finite gate entry; production run() keeps a bounded startup flag.
    NativeGroupedMmqWork run_down_int8(int type,const unsigned long long* weights,
        const int32_t* source_ids,int entries,const void* q8,void* controls,size_t controls_bytes,
        void* packed,size_t packed_bytes,float* output,bool active_tiles=false);
private:
    NativeGroupedMmqWork submit(const NativeMmqPlan& plan,int type,int parts,int entries,int source_rows,
        const void* input,void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
        float* output,int cols,int outputs,bool active,bool ordinary,bool direct);
    NativeGroupedMmqWork run_input(int type, const unsigned long long* first, const unsigned long long* second,
        const int32_t* source_ids, int entries, int source_rows, const void* q8,
        void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
        float* output, int cols, int outputs, bool active_tiles, const float* gate, const float* up,
        bool force_gu_int8 = false, bool force_down_int8 = false, bool force_gu_i32 = false, bool force_gu_i64 = false);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::prefill::mmq
