#pragma once
#include <cstddef>
#include <cstdint>
namespace strata::glm5_next {
// Optional bounded benchmark trace; no host work or mailbox access per route.
inline constexpr uint32_t route_trace_capacity=131072;
struct RouteTraceHeader {
    uint32_t count=0,overflow=0,enabled=0,reserved[13]{};
};
struct RouteTraceRecord {
    uint32_t layer=0,count=0,missing=0,reserved=0;
    uint16_t ids[16]{};
};
static_assert(sizeof(RouteTraceHeader)==64&&sizeof(RouteTraceRecord)==48);
inline constexpr std::size_t route_trace_bytes=sizeof(RouteTraceHeader)+route_trace_capacity*sizeof(RouteTraceRecord);
}
