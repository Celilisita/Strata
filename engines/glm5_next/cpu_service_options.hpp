#pragma once
#include <system_error>
#include "cpu_placement.hpp"
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline int cpu_service_threads() {
    const char* value=std::getenv("STRATA_GLM53_CPU_THREADS");
    if(!value){const auto& placement=cpu_placement();const int count=placement.enabled?int(placement.logical.size()):8;validate_cpu_threads(count);return count;}
    int count=0;const char* end=value+std::strlen(value);const auto parsed=std::from_chars(value,end,count);
    if(parsed.ec!=std::errc{}||parsed.ptr!=end||count<1||count>32)throw std::runtime_error("GLM CPU threads must be1..32");
    validate_cpu_threads(count);return count;
}
inline bool cpu_wide_requested(){const char* flag=std::getenv("STRATA_GLM53_CPU_IQ3_WIDE");return flag&&!std::strcmp(flag,"1");}
}
