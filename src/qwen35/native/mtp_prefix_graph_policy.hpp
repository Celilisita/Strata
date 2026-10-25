#pragma once
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
struct MtpPrefixGraphStats {
    bool initial_omitted=false,ready=false;size_t lazy_builds=0;
    double record_ms=0,instantiate_ms=0,cold_ms=0;
};
inline MtpPrefixGraphStats &mtp_prefix_graph_stats(){static thread_local MtpPrefixGraphStats stats;return stats;}
// PREPARE only. The service's reuse mode already proves a complete current-request eager prefix.
inline bool skip_service_mtp_prefix_graph(bool reuse){
    const char *raw=std::getenv("STRATA_DENSE_MTP_SKIP_UNUSED_PREFIX_GRAPH");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("MTP unused prefix graph mode must be auto,0,1");
    return reuse&&mode!="0";
}
}
