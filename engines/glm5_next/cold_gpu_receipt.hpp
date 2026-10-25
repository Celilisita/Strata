#pragma once
#include "cold_gpu_host_mask.hpp"
#include "../deepseek_v4/device_cache_protocol.hpp"
#include <stdexcept>

namespace strata::glm5_next {
struct ColdGpuTicketReceipt {
    ColdGpuSelection selected;
    uint32_t occurrences=0;
};

// Pure validation of one already-published ticket. This borrows metadata and
// directory snapshots; it does not establish the owner's phase/lease lifetime.
inline ColdGpuTicketReceipt validate_cold_gpu_ticket_receipt(
    const ColdGpuLayer& meta,uint64_t epoch,int layer,uint32_t missing,
    const int32_t* directory,const deepseek_v4::DeviceMissPacket& packet) {
    const uint32_t mask=packet.gpu_cold_mask;
    if(packet.owner_epoch!=epoch||packet.native_request!=5||packet.routed_count!=16||
        int(packet.layer)!=layer||!cold_gpu_layout_valid(meta,epoch,layer)||!directory||
        (missing&~0xffffu)||(mask&~missing))
        throw std::runtime_error("GLM cold GPU ticket identity differs");
    ColdGpuTicketReceipt receipt;
    receipt.selected=cold_gpu_host_selection(meta,packet.ids,missing);
    const auto& selected=receipt.selected;
    if(mask!=selected.mask)throw std::runtime_error("GLM cold GPU selected mask differs");
    if(!mask)return receipt;
    if((meta.topology_fallback?(selected.count<1||selected.count>2):(selected.count!=2&&selected.count!=4))||selected.count>meta.reserved_rows)
        throw std::runtime_error("GLM cold GPU selected count differs");
    if(meta.topology_fallback){
        for(uint32_t i=0;i<selected.count;++i)if(directory[selected.keys[i]]!=-1)
            throw std::runtime_error("GLM cold GPU selected key is resident");
        for(int rank=0;rank<16;++rank)if(mask&(1u<<rank)){
            uint32_t row=0;for(;row<selected.count;++row)if(packet.ids[rank]==selected.keys[row])break;
            if(row==selected.count||!(missing&(1u<<rank))||packet.reply_slots[rank]!=int32_t(meta.first_row+row))
                throw std::runtime_error("GLM cold GPU occurrence row receipt differs");
        }
    }else{
        for(uint32_t i=0;i<selected.count;++i)if(directory[selected.keys[i]]!=-1||packet.reply_slots[selected.ranks[i]]!=int32_t(meta.first_row+i))
            throw std::runtime_error("GLM cold GPU sorted row receipt differs");
    }
    receipt.occurrences=meta.topology_fallback?cold_gpu_occurrences(mask):selected.count;
    if(meta.topology_fallback&&(receipt.occurrences<selected.count||receipt.occurrences>2*selected.count))
        throw std::runtime_error("GLM cold GPU occurrence count differs");
    return receipt;
}
}
