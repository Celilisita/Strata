#pragma once
#include <algorithm>
#include <stdexcept>
namespace strata::qwen35::native {
// No paging may occur during this whole fresh request, including tentative rows.
// A current-position-only test would be unsafe after a prior eviction/rollback.
inline int resident_ram_request_bound(int maximum,int lookahead,int resident,int host_capacity){
    if(maximum<1||maximum>262144||lookahead<0||lookahead>262144||resident<0||resident>262144||host_capacity<1||host_capacity>262144)
        throw std::invalid_argument("resident RAM request budget");
    const int available=std::min(resident,host_capacity);
    return maximum>available||lookahead>available-maximum?0:maximum+lookahead;
}
}
