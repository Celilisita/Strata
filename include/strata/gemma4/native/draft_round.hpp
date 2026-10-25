#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/gemma4/native/model.hpp"
#include "strata/gemma4/native/assistant.hpp"
#include <array>
#include <vector>

namespace strata::gemma4::native {
namespace detail {
struct DraftRoundState {
    assistant::Bounds bounds{};
    int remaining=0,drafts[2]{};
    unsigned invalid=0;
    int eog_ids[16]{};
};
static_assert(sizeof(DraftRoundState)==108,"Gemma218 fixed GPU metadata ABI");
void prepare_draft_round(const int*,const unsigned*,DraftRoundState*,int,int,int,cudaStream_t);
void save_draft_round(const int*,const unsigned*,DraftRoundState*,int,cudaStream_t);
}
struct DraftRoundStats {
    // External benchmark owns and separately times the one Draft constructor.
    double assistant_construct_wall_s=0,bind_wall_s=0;
    size_t assistant_weight_bytes=0,assistant_scratch_bytes=0,borrowed_decode_bytes=0;
    size_t metadata_bytes=0,bind_H2D_bytes=0,assistant_tensors=0;
    std::array<size_t,3> captures{},replays{},graph_nodes{};
    std::array<double,3> capture_host_wall_s{};
    size_t warm_draft_runs=0,rounds=0,emitted=0,accepted=0,raw_matched=0;
    size_t base_T1_tail_rounds=0,tail_budget=0,tail_room=0,tail_assistant_context=0;
};

// Borrow one Main and one49-tensor Draft. Caller constructs that Draft once
// with Main.draft_scratch()'s stream/arena and separately records loadwall.
// Both must outlive this controller/graphs. Bind after reset+prefill; the same
// Draft can serve the preceding217 unit and subsequent MTP0/1/2 requests.
class DraftRoundController {
    Model& main_;
    strata::runtime::CapacityStorage memory_;
    assistant::Draft& draft_;
    detail::DraftRoundState* state_=nullptr;
    const void* owner_=nullptr;
    cudaStream_t stream_=nullptr;
    void* decoded_=nullptr;
    size_t decoded_bytes_=0;
    MainDraftLease main_lease_{};
    MainCacheLease local_{},global_{};
    std::array<cudaGraph_t,3> graphs_{};
    std::array<cudaGraphExec_t,3> executables_{};
    std::uint64_t epoch_=0,failed_epoch_=0,draft_head_epoch_=0;
    int position_=0,capacity_=0,remaining_=0,requested_=0,eog_count_=0,assistant_context_=0;
    bool bound_=false,finished_=false,poisoned_=false,draft_warmed_=false;
    DraftRoundStats stats_{};
    void validate_binding()const;
    void enqueue(int drafts);
    void capture(int drafts);
    void destroy_graphs()noexcept;
public:
    DraftRoundController(Model&,assistant::Draft&);
    ~DraftRoundController();
    DraftRoundController(const DraftRoundController&)=delete;
    DraftRoundController& operator=(const DraftRoundController&)=delete;
    // Budget excludes the anchor already emitted by prefill. This cold call
    // uploads only its fixed metadata/EOG table and drains that upload. Caller
    // stops on an EOG prefill anchor before binding a nonempty budget.
    void bind(int remaining_after_anchor,const std::vector<int>& eog_ids,
        int maximum_drafts,int reserved_capacity);
    // Existing graphs: GraphLaunch + exactly one44-byte finish_verify D2H/join.
    // First-use warm/capture belongs inside the caller's TG/request timer.
    VerifyResponse round();
    // REQUIRED before every Model reset, policy change or destruction. This
    // cold call joins the stream, destroys all graphs and invalidates bind.
    void release_graphs();
    int remaining()const{return remaining_;}
    bool finished()const{return finished_;}
    const DraftRoundStats& stats()const{return stats_;}
};
}
