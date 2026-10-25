#pragma once
#include "strata/prefill/native_grouped_mmq.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
namespace strata::prefill::mmq {
// GGUF type40 raw block64: four E4M3 block16 scales, then32 split-nibble bytes.
// Per-expert F32 global factors are explicit control metadata, never baked twice.
struct NativeNvfp4Matrix {
    uint64_t gate=0,up=0;
    float gate_scale=1,up_scale=1;
    int32_t begin=0,end=0,expert_id=-1,reserved=0;
};
struct NativeNvfp4Input {
    const uint8_t *values=nullptr,*scales=nullptr;
    const float* globals=nullptr;
    const int32_t* source_ids=nullptr;
    int source_rows=0,columns=0;
};
struct NativeNvfp4Resources {
    int registers=0,threads=128,active_blocks_per_sm=0;
    size_t shared_bytes=0,local_bytes=0;
};
// One compact F32-row encoding: adjacent E2M1 nibbles, linear row/block16 E4M3,
// and4B F32 global per row. Global=max(rowamax/(6*448),FLT_MIN), zero row global1.
// Invalid F32 rows carry a NaN global and NaN block scale; they cannot become finite FP4 zeros.
size_t native_nvfp4_input_bytes(int source_rows,int columns);
NativeNvfp4Input encode_native_nvfp4(const float* input,void* packed,size_t packed_bytes,
    int source_rows,int columns,const int32_t* sorted_source_ids,void* stream);
// Missing TG rows have no GU output: do not read uninitialized hidden rows.
NativeNvfp4Input encode_native_nvfp4_masked(const float* input,void* packed,size_t packed_bytes,
    int source_rows,int columns,const int32_t* sorted_source_ids,
    const unsigned long long* active_addresses,void* stream);
void native_grouped_nvfp4(const NativeNvfp4Matrix* matrices,int groups,int parts,
    const int32_t* output_ids,float* output,int entries,int max_rows,int outputs,
    const NativeMmqTile* tiles,int tile_count,int tile_rows,
    NativeNvfp4Input input,void* stream);
NativeNvfp4Resources native_nvfp4_resources(int tile_rows);
class NativeGroupedNvfp4Provider {
public:
    explicit NativeGroupedNvfp4Provider(void* stream);
    ~NativeGroupedNvfp4Provider();
    NativeGroupedNvfp4Provider(const NativeGroupedNvfp4Provider&)=delete;
    NativeGroupedNvfp4Provider& operator=(const NativeGroupedNvfp4Provider&)=delete;
    // Host arrays have entries elements; scales[e] belongs to expert_ids[e].
    // Up arrays are both null for Down, both present for paired GU.
    // Input is raw F32: GU norm[source_rows,K], Down hidden[entries,K].
    // Output retains existing [parts,entries,outputs] and original entry scatter.
    // Raw input/output may overlap: the complete encoder precedes every product store.
    // Controls/packed must be separate from both. Scratch is borrowed until stream completion.
    // Calls support1..65536 entries/rows; PP performance qualification is separate.
    NativeGroupedMmqWork run(const unsigned long long* gate,const unsigned long long* up,
        const float* gate_scales,const float* up_scales,const int32_t* expert_ids,
        const int32_t* source_ids,int entries,int source_rows,const float* input,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
        float* output,int columns,int outputs,int tile_rows=64);
    // Same full control/encode lease, matrix-wave product subsets for actual-copy overlap.
    NativeGroupedMmqWork run_gu_pipeline(const unsigned long long* gate,const unsigned long long* up,
        const float* gate_scales,const float* up_scales,const int32_t* expert_ids,
        const int32_t* source_ids,int entries,int source_rows,const float* input,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,float* output,
        int matrices_per_wave,const std::function<void(const NativeNvfp4Matrix*,int,bool)>& ready);
private:
    NativeGroupedMmqWork run_input(const unsigned long long* gate,const unsigned long long* up,
        const float* gate_scales,const float* up_scales,const int32_t* expert_ids,
        const int32_t* source_ids,int entries,int source_rows,const float* input,
        void* controls,size_t controls_bytes,void* packed,size_t packed_bytes,
        float* output,int columns,int outputs,int tile_rows,int wave,
        const std::function<void(const NativeNvfp4Matrix*,int,bool)>& ready);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
