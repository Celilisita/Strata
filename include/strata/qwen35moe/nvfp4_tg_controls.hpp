#pragma once
#include "strata/prefill/native_grouped_nvfp4.hpp"
#include "strata/core/device_miss.hpp"
#include <cstddef>
#include <stdexcept>
namespace strata::qwen35moe {
// One token, eight unique experts. Borrow the independent expert_q8 owner.
// Top8 writes these controls; both products consume them on the same stream.
struct Nvfp4TgControls {
    prefill::mmq::NativeNvfp4Matrix gu[8],down[8];
    prefill::mmq::NativeMmqTile tiles[8];
    int32_t sources[16],outputs[8];
};
static_assert(sizeof(Nvfp4TgControls)==800,"bounded TG control scratch");
inline void validate_nvfp4_routes(const core::NativeRouteLedger* ledger,size_t layers,
                                 uint32_t epoch,uint32_t token) {
    if(!ledger||!layers||layers>64||!epoch||!token)throw std::invalid_argument("NVFP4 route ledger domain");
    for(size_t il=0;il<layers;++il) {
        const auto& r=ledger->layers[il];bool seen[256]{};
        if(r.epoch!=epoch||r.token!=token||r.layer!=il||(r.mask>>8))
            throw std::runtime_error("incomplete/stale NVFP4 token route ledger");
        for(int id:r.ids){if(id<0||id>=256||seen[id])throw std::runtime_error("invalid/duplicate NVFP4 token route");seen[id]=true;}
    }
}
inline void bind_nvfp4_tg(core::NativeMissArgs& a,void* scratch,size_t bytes,const float* scales) {
    if(!scratch||reinterpret_cast<uintptr_t>(scratch)%8||bytes<sizeof(Nvfp4TgControls)||!scales)
        throw std::invalid_argument("unreserved NVFP4 TG controls");
    auto* c=static_cast<Nvfp4TgControls*>(scratch);
    a.nv_gu=c->gu;a.nv_down=c->down;a.nv_tiles=c->tiles;
    a.nv_sources=c->sources;a.nv_outputs=c->outputs;a.nv_scales=scales;
}
}
