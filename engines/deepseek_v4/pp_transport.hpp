#pragma once
#include "expert_source_pin.hpp"
#include "packed_expert_store.hpp"
#include <stdexcept>

namespace strata::deepseek_v4 {
using PpTransportQuery = ExpertSourcePin::Segment (*)(const void*, size_t, void*);
// The backend borrows the existing registration owner. It never registers
// another range or retains a source allocation. Unbinding drains both streams.
inline void bind_pp_transport(ggml_backend_t backend, PpTransportQuery query, void* owner) {
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Bind = bool (*)(ggml_backend_t, PpTransportQuery, void*);
    const auto bind = reinterpret_cast<Bind>(ggml_backend_reg_get_proc_address(reg, "strata_pp_stream_bind_transport"));
    if (!bind || !bind(backend, query, owner))
        throw std::runtime_error("PP source registration projection unavailable");
}
inline void bind_pp_packed(ggml_backend_t backend, PackedExpertQuery query, void* owner) {
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Bind=bool(*)(ggml_backend_t,PackedExpertQuery,void*);
    const auto bind=reinterpret_cast<Bind>(ggml_backend_reg_get_proc_address(reg,"strata_pp_bind_packed"));
    if(!bind||!bind(backend,query,owner))throw std::runtime_error("PP packed inventory binding unavailable");
}
}
