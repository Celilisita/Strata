#pragma once
namespace strata::qwen35::native {
// Derived from complete growth costs; explicit environment flags override it.
inline bool fresh_pool_default(int required){return required<=16384?true:true;} // POOL_POLICY
inline bool fresh_mtp_default(int required){return required<=16384?true:true;} // MTP_POLICY
}
