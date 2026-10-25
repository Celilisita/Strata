#pragma once
#include "partial_down_protocol.hpp"
#include "../deepseek_v4/device_cache_protocol.hpp"
#include "../deepseek_v4/cpu_expert_service.hpp"
#include "../deepseek_v4/cpu_partial_down.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace strata::glm5_next {
inline uint32_t partial_down_acquire(const uint32_t& word){
#ifdef _MSC_VER
    return uint32_t(_InterlockedCompareExchange(reinterpret_cast<volatile long*>(const_cast<uint32_t*>(&word)),0,0));
#else
    return __atomic_load_n(&word,__ATOMIC_ACQUIRE);
#endif
}
inline void partial_down_release(uint32_t& word,uint32_t value){
#ifdef _MSC_VER
    _InterlockedExchange(reinterpret_cast<volatile long*>(&word),long(value));
#else
    __atomic_store_n(&word,value,__ATOMIC_RELEASE);
#endif
}
struct PartialDownHostReceipt {
    bool selected=false;
    uint32_t gu_routes=0,down_routes=0,published_mid_bytes=0,contribution_bytes=0;
};
class PartialDownHostTicket {
public:
    PartialDownHostTicket(PartialDownPhase& phase,deepseek_v4::DeviceMissPacket& packet,uint64_t epoch,uint64_t generation)
        :phase_(phase),packet_(packet),epoch_(epoch),generation_(generation),ticket_(packet.sequence){
        if(!ticket_||phase_.ticket!=ticket_||phase_.owner_epoch!=epoch_||phase_.bank_generation!=generation_||
            packet_.owner_epoch!=epoch_||packet_.native_request!=5||packet_.layer!=24||packet_.routed_count!=16)
            throw std::runtime_error("partial Down ticket identity differs");
    }
    bool candidate()const {
        return phase_.partial_mask==partial_down_role_mask&&phase_.cpu_down_mask==partial_down_cpu_mask&&
            phase_.down_keys[0]==57&&phase_.down_keys[1]==234&&packet_.gpu_cold_mask==partial_down_whole_mask;
    }
    void commit_stock(){commit(partial_down_stock);}
    void commit_partial(){if(!candidate())throw std::runtime_error("partial Down candidate differs");commit(partial_down_selected);}
    void fail()noexcept {
        packet_.error=1;packet_.cpu_error=1;partial_down_release(phase_.error_seq,ticket_);
    }
    static bool publish_mid_callback(void* context,const float* mid,size_t rows,size_t width){
        return static_cast<PartialDownHostTicket*>(context)->publish_mid(mid,rows,width);
    }
private:
    void commit(uint32_t value){
        if(partial_down_acquire(phase_.decision_seq)||partial_down_acquire(phase_.error_seq))
            throw std::runtime_error("partial Down decision already committed");
        phase_.decision=value;partial_down_release(phase_.decision_seq,ticket_);
    }
    bool publish_mid(const float* mid,size_t rows,size_t width){
        if(!mid||rows!=10||width!=2048||phase_.ticket!=ticket_||
            partial_down_acquire(phase_.decision_seq)!=ticket_||phase_.decision!=partial_down_selected||
            partial_down_acquire(phase_.mid_ready_seq)||partial_down_acquire(phase_.error_seq)||
            partial_down_acquire(phase_.reader_error))
            throw std::runtime_error("partial Down mid publication differs");
        for(unsigned row=0;row<4;++row){const float* source=mid+size_t(partial_down_mid_rows[row])*width;
            for(size_t i=0;i<width;++i)if(!std::isfinite(source[i]))throw std::runtime_error("partial Down mid is nonfinite");
            std::memcpy(packet_.contribution+size_t(partial_down_ranks[row])*4096,source,width*sizeof(float));}
        partial_down_release(phase_.mid_ready_seq,ticket_);return true;
    }
    PartialDownPhase& phase_;deepseek_v4::DeviceMissPacket& packet_;
    uint64_t epoch_,generation_;uint32_t ticket_;
};

// One serial-worker adapter shared with the real-lookup fixture. False means
// a stock decision was committed; the caller then executes its original GU/Down.
inline PartialDownHostReceipt partial_down_service_impl(deepseek_v4::CpuExpertService& service,
    deepseek_v4::DeviceMissPacket& packet,PartialDownPhase& phase,uint64_t epoch,uint64_t generation,
    const int32_t* ids,const uint8_t* positions,const int32_t* ranks,size_t count,bool diagnostic_fail_after_commit=false){
    PartialDownHostTicket ticket(phase,packet,epoch,generation);PartialDownHostReceipt result;
    bool exact=count==10&&ids&&positions&&ranks;
    for(size_t i=0;exact&&i<10;++i)exact=ids[i]==partial_down_cpu_ids[i]&&positions[i]==partial_down_cpu_positions[i]&&ranks[i]==partial_down_gu_ranks[i];
    std::array<float,6*4096> output{};
    deepseek_v4::PartialDownJob job;
    job.activations=packet.activation;job.ids=ids;job.positions=positions;job.original_ranks=partial_down_gu_ranks;
    job.down_rows=partial_down_cpu_rows;job.output=output.data();job.output_elements=output.size();
    job.publish_mid=PartialDownHostTicket::publish_mid_callback;job.publication_context=&ticket;
    if(!exact||!ticket.candidate()||!service.partial_down_preflight(job)){ticket.commit_stock();return result;}
    ticket.commit_partial();job.partial_decision_committed=true;
    try {
        if(diagnostic_fail_after_commit)throw std::runtime_error("diagnostic partial Down failure after decision commit");
        const auto receipt=service.compute_partial_down_diagnostic(job);
        if(receipt.gu_routes!=10||receipt.down_routes!=6||receipt.mid_width!=2048||
            receipt.publication_callbacks!=1||!receipt.mid_published)
            throw std::runtime_error("partial Down CPU path receipt differs");
        packet.cpu_mask=0;
        for(unsigned row=0;row<6;++row){for(unsigned i=0;i<4096;++i)if(!std::isfinite(output[size_t(row)*4096+i]))
                throw std::runtime_error("partial Down CPU contribution nonfinite");
            const unsigned rank=partial_down_cpu_ranks[row];
            std::memcpy(packet.contribution+size_t(rank)*4096,output.data()+size_t(row)*4096,4096*sizeof(float));
            packet.reply_slots[rank]=-1;packet.cpu_mask|=1u<<rank;}
        if(packet.cpu_mask!=partial_down_cpu_mask)throw std::runtime_error("partial Down CPU mask differs");
        result={true,10,6,32768,98304};return result;
    }catch(...){ticket.fail();throw;}
}
inline PartialDownHostReceipt partial_down_service(deepseek_v4::CpuExpertService& service,
    deepseek_v4::DeviceMissPacket& packet,PartialDownPhase& phase,uint64_t epoch,uint64_t generation,
    const int32_t* ids,const uint8_t* positions,const int32_t* ranks,size_t count,bool diagnostic_fail_after_commit=false){
    try {return partial_down_service_impl(service,packet,phase,epoch,generation,ids,positions,ranks,count,diagnostic_fail_after_commit);}
    catch(...){packet.error=1;packet.cpu_error=1;partial_down_release(phase.error_seq,packet.sequence);throw;}
}
}
