#pragma once
#include <cuda_runtime.h>
namespace strata::gemma4::native {
enum class VerifyStop:int {Continue=0,Eog=1,Budget=2,Invalid=3};
// Both pointers are GPU-resident. remaining is decremented only on a successful
// commit. EOG IDs are a cold-owned table; zero count permits a null table.
struct VerifyPolicy {int* remaining=nullptr;const int* eog_ids=nullptr;int eog_count=0;};
// A round verifies [pending,draft1,(draft2)]. Pending was already emitted.
// matched is the raw matching prefix; accepted=count-1 is the committed prefix.
// The final emitted ID is always represented as the target bonus, including an
// EOG/matched draft or a budget-truncated draft. That ID remains unprocessed.
struct VerifyResponse {
    int count=0,accepted=0,matched=0,position=0,finished=0;
    VerifyStop reason=VerifyStop::Continue;int ids[3]{};
};
void form_verify_inputs(const int* pending,const int* draft1,const int* draft2,
    int* inputs,int rows,int vocabulary,unsigned* invalid,cudaStream_t);
// Independent capped-greedy rows; never advances the model position. Workspace
// is rows*ceil(vocabulary/256)*8 bytes. Logits are capped in place exactly once;
// capped-score ties choose the lower ID, including saturation ties.
void pick_verify_rows(float* logits,float* partials,int* picks,int rows,
    int vocabulary,float cap,unsigned* invalid,cudaStream_t);
// Position still denotes the round start. The KV arena must retain the previous
// window plus all rows: moving a cursor cannot restore overwritten ring entries.
// Position is a logical cursor, not physical ring capacity. The caller must
// retain at least window+max_verify_rows ring entries and allocate the complete
// verify span before running the model. This helper never changes KV payloads.
// EOG/remaining stop before writing seed/pending/position when count is zero.
// Otherwise seed=row_features[count-1], position=start+count; all computed
// features must be finite. All write spans must be disjoint from every other
// span. No allocation, synchronization, D2H or CPU inference occurs here.
void commit_verify_rows(const int* inputs,const int* picks,const float* row_features,
    float* seed,int* pending,int* position,VerifyResponse*,int rows,int hidden,
    int vocabulary,int capacity,VerifyPolicy,unsigned* invalid,cudaStream_t);
}
