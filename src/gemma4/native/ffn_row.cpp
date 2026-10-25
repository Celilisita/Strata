#include "half_row_pair.hpp"
#include "half_k_panels.hpp"
#include "half_k_three_panels.hpp"
#include "strata/gemma4/native/pp3_q4_panel.hpp"
#include "strata/gemma4/native/fast_full_k_q4.hpp"
namespace strata::gemma4::native {
void Ffn::row_half_prefill(const float* normalized,int tokens,cudaStream_t stream){
    if((!row_half_&&!compact_half_&&!three_panel_)||pp_algorithm_!=2||!residual_pp_||!normalized||tokens!=1280||tokens>maximum_||!stream)
        throw std::invalid_argument("Gemma row-half precision/extent lease");
    HalfRowPairLease lease;
    lease.input=normalized;
    lease.weight_arena=decoded_;lease.weight_bytes=size_t(hidden_)*intermediate_*sizeof(float);
    lease.input_arena=buffers_.down;lease.input_bytes=size_t(maximum_)*hidden_*sizeof(float);
    lease.workspace=workspace_;lease.workspace_bytes=32u*1024*1024;
    lease.output={buffers_.gate,buffers_.activation,size_t(maximum_)*intermediate_*2,
        size_t(maximum_)*intermediate_,intermediate_,tokens};
    lease.handle=handle_;lease.stream=stream;
    if(three_panel_){
        if(short_pp_backend_==ShortPpBackend::FastPanel3){
            const Q4PpPanelBinding live{this,++short_pp_epoch_,stream};
            q4_pp_panel_project(weights_.gate,weights_.up,{lease,live},live);
        }else if(short_pp_backend_==ShortPpBackend::FastFullK){
            const FastFullKBinding live{this,++short_pp_epoch_,stream};
            fast_full_k_q4_project(weights_.gate,weights_.up,{lease,live},live);
        }else half_row_pair_three_panel_project(weights_.gate,weights_.up,lease);
        ++short_pp_enqueues_[static_cast<unsigned>(short_pp_backend_)];
    }else if(compact_half_)half_row_pair_panel_project(weights_.gate,weights_.up,lease);
    else half_row_pair_project(weights_.gate,weights_.up,lease);
    half_row_pair_gelu(lease.output,stream);
    // Same GELU output and existing full Down; no new arena or global sidecar.
}
}
