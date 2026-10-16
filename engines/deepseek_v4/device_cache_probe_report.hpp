#pragma once
#include "device_cache_backend.hpp"
#include <ostream>
namespace strata::deepseek_v4 {
void write_cache_api_costs(std::ostream&, const DeviceCacheDmaCosts&);
}
