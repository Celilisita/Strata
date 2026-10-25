#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/gemma4/native/assistant_spec.hpp"
#include "strata/gemma4/native/assistant_quant.hpp"
#include "strata/gemma4/native/decoder_ops.hpp"
#include "strata/gemma4/native/history.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::gemma4::native::assistant {
// Device-resident immutable per-round bounds. Origins are absolute positions,
// lengths describe supplied shared states, and physical slots are absolute%capacity.
// RoPE P is deliberately distinct from KV end P-1 and explicit SWA left.
struct Bounds {
    int rope_position=0,kv_end=-1,sliding_left=0;
    int sliding_origin=0,sliding_length=0,global_origin=0,global_length=0;
};
struct ResidentTensor {std::string name;const void* device=nullptr;size_t bytes=0;};
struct Output {
    const float *logits=nullptr,*post_state=nullptr;
    const int* proposal=nullptr;const unsigned* invalid=nullptr;
};
// One query, explicit interval, no KV writes. Bounds/CacheView are benchmarkable
// without constructing a model; invalid bits are ORed, never cleared here.
void cross_attention(const float* query,float* output,CacheView,const Bounds*,bool sliding,
    int context,unsigned* invalid,cudaStream_t);
enum class HeadBackend {DecodeSgemm=0,Direct399=1};
class Draft {
    struct Weights {
        Q40Matrix q,out,gate,up,down;
        const float *pre=nullptr,*post=nullptr,*q_norm=nullptr,*ffn_pre=nullptr,*ffn_post=nullptr,*scale=nullptr;
    };
    Spec spec_;strata::runtime::CapacityStorage weights_,memory_;
    std::vector<ResidentTensor> inventory_;std::array<Weights,4> layers_;
    Q40Matrix pre_projection_,post_projection_,head_;
    const float *output_norm_=nullptr,*frequencies_=nullptr;
    cudaStream_t stream_;cublasHandle_t handle_=nullptr;
    float *decoded_=nullptr,*condition_=nullptr,*hidden_[2]{},*normalized_=nullptr,*query_=nullptr,*attended_=nullptr;
    float *projected_=nullptr,*post_=nullptr,*after_=nullptr,*gate_=nullptr,*up_=nullptr,*activation_=nullptr;
    float *down_=nullptr,*features_=nullptr,*post_state_=nullptr,*logits_=nullptr;
    int* proposal_=nullptr;unsigned* invalid_=nullptr;void* workspace_=nullptr;
    size_t decoded_values_=0;bool borrowed_decode_=false,split_attention_=false;
    int runtime_context_=131072;
    // Independent per-instance policy; never encoded in another selector.
    HeadBackend head_backend_=HeadBackend::DecodeSgemm;
    std::uint64_t head_epoch_=0,head_parent_enqueues_=0,head_direct_enqueues_=0;
    const void* payload(const TensorInfo*)const;
    Q40Matrix matrix(const TensorInfo*)const;
    const float* scalar(const TensorInfo*)const;
    void project(Q40Matrix,const float*,float*);
    void validate_lease(const float*,const float*,CacheView,CacheView,const Bounds*)const;
public:
    // Cold setup copies49 raw tensors, allocates all storage and drains upload.
    // Optional borrowed decode arena must be >=128MiB, exclusive during run,
    // and outlive this Draft and all its captured graph replays.
    Draft(const GgufFile&,cudaStream_t,void* borrowed_decode=nullptr,size_t decoded_bytes=0);
    ~Draft();Draft(const Draft&)=delete;Draft& operator=(const Draft&)=delete;
    // Input arrays are5376 floats each. Already scaled main embedding first;
    // seed is final-RMS main hidden or preceding assistant post_state.
    // No allocation, synchronization, D2H, CPU operator or cache mutation.
    void run(const float* main_scaled_embedding,const float* normalized_seed,
        CacheView sliding_main58,CacheView global_main59,const Bounds* device_bounds,
        const HistoryView* global_history=nullptr);
    // Cold policy: caller destroys all controller/external graphs first.
    // Joining this stream cannot invalidate graphs held by another owner.
    void select_split_attention(bool enabled);
    // Explicit runtime position extension; GGUF context/frequencies stay intact.
    // Caller must release external Graphs before this cold policy transition.
    void select_runtime_context(int);
    int runtime_context()const{return runtime_context_;}
    int metadata_context()const{return spec_.context;}
    bool split_attention_enabled()const{return split_attention_;}
    // Cold request boundary: release Controller/external Graphs first. Always
    // joins, increments the policy epoch and clears enqueue counters, including
    // when selecting the same backend. Default epoch0/Parent is valid.
    void select_head_backend(HeadBackend);
    HeadBackend head_backend()const{return head_backend_;}
    std::uint64_t head_policy_epoch()const{return head_epoch_;}
    // Counts host enqueues during warm/capture/eager calls, never Graph replays.
    std::uint64_t head_parent_enqueues()const{return head_parent_enqueues_;}
    std::uint64_t head_direct399_enqueues()const{return head_direct_enqueues_;}
    Output output()const{return {logits_,post_state_,proposal_,invalid_};}
    const std::vector<ResidentTensor>& resident_tensors()const{return inventory_;}
    size_t weight_bytes()const{return weights_.bytes();}
    size_t scratch_bytes()const{return memory_.bytes();}
    size_t decoded_capacity_bytes()const{return decoded_values_*sizeof(float);}
    bool borrows_decode()const{return borrowed_decode_;}
};
}
