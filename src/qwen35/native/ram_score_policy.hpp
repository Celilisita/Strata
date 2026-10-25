#pragma once
#include "ram_score.hpp"
#include "ram40_score.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
struct RamScoreRequest {bool enabled=false;std::uint64_t submissions=0;};
inline RamScoreRequest &ram_score_request(){static thread_local RamScoreRequest s;return s;}
// Separate host counter preserves the existing16-byte request/TLS layout in retained32K CUDA.
inline std::uint64_t &ram_score40_submissions(){static thread_local std::uint64_t n=0;return n;}
inline std::uint64_t ram_score40_host_submissions(){return ram_score40_submissions();}
// PREPARE only, after releasing old Graphs; replay never reads environment or host state.
inline void configure_ram_score_request(bool admitted){
    const char *mode=std::getenv("STRATA_DENSE_RAM_SCORE_REUSE");
    if(mode&&std::string(mode)!="auto"&&std::string(mode)!="0"&&std::string(mode)!="1")
        throw std::invalid_argument("RAM score reuse mode");
    int device=0;check_cuda(cudaGetDevice(&device),"RAM score cold device");const auto caps=capabilities(device);
    auto &s=ram_score_request();s={admitted&&(!mode||mode[0]!='0')&&caps.major==12&&caps.minor==0,0};ram_score40_submissions()=0;
    if(s.enabled){const auto r=ram_score_resources();if(r.local_bytes||r.blocks_per_sm<1)throw std::runtime_error("RAM score resource budget");}
    if(s.enabled){const auto r=ram40_score_resources();if(r.binary_version!=120||r.local_bytes||r.blocks_per_sm<1)throw std::runtime_error("RAM40 score resource budget");}
}
inline std::uint64_t ram_score_host_submissions(){return ram_score_request().submissions;}
inline bool use_ram_score_reuse(RamKvView r,int tokens,int heads,int parts,size_t bytes){
    return ram_score_request().enabled&&tokens==3&&heads==24&&r.heads==4&&parts==64&&
        r.selected_tokens==8192&&r.selected_slots==512&&r.resident==32768&&r.slots==2048&&
        r.pages>=2049&&r.pages<=4096&&r.host_capacity==r.pages*RamPage&&bytes>=2371584;
}
inline bool use_ram40_score_reuse(RamKvView r,int tokens,int heads,int parts,size_t bytes){
    return ram_score_request().enabled&&tokens==3&&heads==24&&r.heads==4&&parts==64&&
        r.selected_tokens==8192&&r.selected_slots==512&&r.keep_queries==3&&r.resident==40960&&r.slots==2560&&
        r.pages==2560&&r.host_capacity==40960&&r.host_capacity==r.pages*RamPage&&
        r.host_key==r.key&&r.host_value==r.value&&bytes>=2371584;
}

}
