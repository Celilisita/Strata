#pragma once
#include "page_row.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35::native {
struct PageRowRequestStats {bool requested=false,admitted=false;std::uint64_t host_calls=0,fallback_calls=0;int last_heads=0,last_pages=0,last_resident=0;};
inline PageRowRequestStats& page_row_request_state(){static thread_local PageRowRequestStats s;return s;}
inline PageRowRequestStats page_row_request_stats(){return page_row_request_state();}
inline bool page_row_requested(){
    static const bool value=[](){const char*p=std::getenv("STRATA_DENSE_RAM_PAGE_ROW");
        if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("RAM page-row flag must be0or1");return p&&p[0]=='1';}();return value;
}
// PREPARE after all old graph leases are released. Graph replay retains no host request object.
inline void configure_page_row_request(bool allowed){
    auto&s=page_row_request_state();s={};s.requested=page_row_requested();if(!s.requested||!allowed)return;
    int device=0;check_cuda(cudaGetDevice(&device),"page-row cold device");const auto c=capabilities(device);
    if(c.major!=12||c.minor!=0)return;
    const auto r=page_row_resources();if(r.binary_version!=120||r.local_bytes||r.baseline_thread_limit<1||r.blocks_per_sm<r.baseline_thread_limit)
        throw std::runtime_error("page-row current occupancy resource gate");s.admitted=true;
}
inline bool use_page_row_scores(RamKvView v,int shift,int heads){
    auto&s=page_row_request_state();const bool use=s.admitted&&shift==0&&heads==24&&v.heads==4&&v.resident==40960&&
        v.host_capacity==40960&&v.pages==2560&&v.slots==2560&&v.selected_tokens==8192&&v.selected_slots==512&&v.keep_queries==3&&
        v.key&&v.value&&v.host_key==v.key&&v.host_value==v.value;
    if(use){++s.host_calls;s.last_heads=heads;s.last_pages=v.pages;s.last_resident=v.resident;}
    else if(s.requested)++s.fallback_calls;return use;
}
}
