#include "mtp_abort.hpp"
namespace strata::completed_v100::qwen35::native::mtp_abort {
namespace {thread_local const NativeMtp* owner=nullptr;thread_local cudaStream_t stream=nullptr;thread_local bool failed=false;thread_local bool fault=false;}
void select(NativeMtp& m,cudaStream_t s,bool enabled){
 if(m.graph_nodes())throw std::logic_error("mtp_abort policy requires released Graphs");
 if(enabled&&(!s||!m.prefix_states_enabled()||!m.gpu_draft_chain()))
  throw std::invalid_argument("mtp_abort requires retained-prefix GPU-chain service");
 if(owner&&owner!=&m)throw std::logic_error("mtp_abort foreign active service");
 owner=enabled?&m:nullptr;stream=enabled?s:nullptr;failed=false;fault=false;
}
bool selected(const NativeMtp& m,cudaStream_t s){return owner==&m&&stream==s;}
void guard(const NativeMtp& m,cudaStream_t s){
 if(owner==&m&&(stream!=s||failed))throw std::logic_error("mtp_abort poisoned/foreign request must reset");
}
void poison(const NativeMtp& m)noexcept{if(owner==&m)failed=true;}
void clear(const NativeMtp& m)noexcept{if(owner==&m){owner=nullptr;stream=nullptr;failed=false;fault=false;}}
void arm_fault(NativeMtp& m){if(owner!=&m||failed)throw std::logic_error("mtp_abort fault hook requires active opt-in");fault=true;}
bool take_fault(const NativeMtp& m)noexcept{if(owner!=&m||!fault)return false;fault=false;return true;}
}
