#include "cpu_weight_fixture.hpp"
#include "cpu_read_ahead_probe.hpp"
#include "checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace strata::deepseek_v4;
struct NativeReadSamples {
    std::vector<double> v;
    void add(double x) {v.push_back(x);}
    double median() const {auto a=v;std::sort(a.begin(),a.end());return a[a.size()/2];}
    void json(std::ostream& out) const {auto a=v;std::sort(a.begin(),a.end());out<<"{\"n\":"<<a.size()<<",\"median\":"<<median()
        <<",\"p10\":"<<a[a.size()/10]<<",\"p90\":"<<a[a.size()*9/10]<<",\"mean\":"<<std::accumulate(a.begin(),a.end(),0.0)/a.size()<<'}';}
};
int main(int argc,char** argv)try {
    if(argc!=4&&argc!=5)throw std::runtime_error("Expected GGUF shard, activation directory, output");
    CpuWeightFixture fixture(argv[1]);auto data=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(data.metadata.type!=GGML_TYPE_F32||data.data.size()<8192*sizeof(float))throw std::runtime_error("Authentic activation absent");
    std::vector<float> x(8192),out(12*4096),expected(12*4096);std::memcpy(x.data(),data.data.data(),8192*sizeof(float));
    Mxfp4CpuService cpu(8);std::ofstream report(argv[argc-1]);report<<std::setprecision(12)
        <<"{\"whole_model_loaded\":false,\"workers\":8,\"affinity\":\"unbound, unchanged\",\"spin\":true,\"vnni\":true,\"new_reader_threads\":0,\"window_ms\":0.34,\"cases\":[";
    if(argc==5){
        struct Replay {int32_t actual[12];uint32_t masks[2];int32_t predicted[12];uint32_t predicted_masks[2];int32_t selected[2];};static_assert(sizeof(Replay)==120);
        std::ifstream input(argv[3],std::ios::binary);Replay item;NativeReadSamples cold,prefetched,reads,bytes,savings;int cases=0;size_t predicted=0,correct=0;
        while(input.read(reinterpret_cast<char*>(&item),sizeof(item))){
            std::vector<int32_t> ids;for(int t=0;t<2;++t)for(int r=0;r<6;++r)if(item.masks[t]&(1u<<r))ids.push_back(item.actual[t*6+r]);
            for(int id:item.selected)if(id>=0){++predicted;correct+=std::find(ids.begin(),ids.end(),id)!=ids.end();ids.push_back(id);}
            std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());const auto spans=expert_spans(fixture.bank,ids);
            for(const auto& span:spans)for(size_t i=0;i<span.bytes;i+=4096)(void)*reinterpret_cast<const volatile uint8_t*>(span.data+i);
            auto compute=[&]{auto start=ReadClock::now();cpu.compute_pair(fixture.bank,x.data(),item.actual,item.masks,out.data());return read_ms(start);};
            for(int trial=0;trial<4;++trial){flush_spans(spans);const auto a=compute();cold.add(a);expected=out;
                flush_spans(spans);auto start=ReadClock::now();bytes.add(double(cpu.read_ahead_pair(fixture.bank,item.predicted,item.predicted_masks,.33,item.selected)));reads.add(read_ms(start));
                const auto b=compute();prefetched.add(b);savings.add(a-b);
                if(out!=expected)throw std::runtime_error("Route replay pre-reading changed real expert output");}
            ++cases;
        }
        if(!input.eof()||cases!=48)throw std::runtime_error("Replay route fixture incomplete");
        report<<"],\"replay_cases\":"<<cases<<",\"selected_ids\":"<<predicted<<",\"correct_selected_ids\":"<<correct<<",\"cold_ms\":";cold.json(report);
        report<<",\"prefetched_ms\":";prefetched.json(report);report<<",\"read_ms\":";reads.json(report);report<<",\"read_bytes\":";bytes.json(report);report<<",\"paired_saving_ms\":";savings.json(report);
        report<<",\"output_unchanged\":true,\"passed\":true}";return 0;
    }
    for(int mode=0;mode<3;++mode) {
        std::array<int32_t,12> routes{0,1,2,3,4,5,0,3,4,6,7,8};std::array<uint32_t,2> masks{7,7};
        const int count=mode==0?1:mode==1?5:9;
        if(mode==0)masks={1,0};
        if(mode==2){routes={0,1,2,3,4,5,6,7,8,9,10,11};masks={63,7};}
        std::vector<int32_t> ids;for(int i=0;i<count;++i)ids.push_back(i);
        const auto spans=expert_spans(fixture.bank,ids);uint64_t fault=0;
        for(const auto& span:spans)for(size_t i=0;i<span.bytes;i+=4096)fault+=span.data[i];
        auto compute=[&]{const auto start=ReadClock::now();if(cpu.compute_pair(fixture.bank,x.data(),routes.data(),masks.data(),out.data())!=size_t(count))throw std::runtime_error("Expert union layout changed");return read_ms(start);};
        auto check=[&]{for(size_t i=0;i<out.size();++i)if(!std::isfinite(out[i])||out[i]!=expected[i])throw std::runtime_error("Idle native read changed expert math");};
        compute();expected=out;
        NativeReadSamples cold,hot,full,partial,selected,full_read_ms,partial_read_ms,partial_bytes,selected_read_ms,selected_bytes;
        for(int trial=0;trial<32;++trial) {
            flush_spans(spans);cold.add(compute());check();hot.add(compute());check();
            flush_spans(spans);auto start=ReadClock::now();const auto bytes=cpu.read_ahead_pair(fixture.bank,routes.data(),masks.data());full_read_ms.add(read_ms(start));
            if(bytes!=uint64_t(count)*13369344)throw std::runtime_error("Native whole-expert read byte count incorrect");
            full.add(compute());check();
            flush_spans(spans);start=ReadClock::now();partial_bytes.add(double(cpu.read_ahead_pair(fixture.bank,routes.data(),masks.data(),0.34)));partial_read_ms.add(read_ms(start));
            partial.add(compute());check();
            const int32_t priority[]{0,count>1?1:-1};flush_spans(spans);start=ReadClock::now();
            selected_bytes.add(double(cpu.read_ahead_pair(fixture.bank,routes.data(),masks.data(),.34,priority)));
            selected_read_ms.add(read_ms(start));selected.add(compute());check();
        }
        if(mode)report<<',';report<<"{\"union_experts\":"<<count<<",\"shared_cpu_experts\":"<<(mode==1?1:0)<<",\"prefault_checksum\":"<<fault<<",\"cold_ms\":";cold.json(report);
        report<<",\"owner_hot_ms\":";hot.json(report);report<<",\"native_full_read_ms\":";full_read_ms.json(report);
        report<<",\"native_full_read_compute_ms\":";full.json(report);report<<",\"native_window_read_ms\":";partial_read_ms.json(report);
        report<<",\"native_window_read_bytes\":";partial_bytes.json(report);report<<",\"native_window_compute_ms\":";partial.json(report);
        report<<",\"selected2_read_ms\":";selected_read_ms.json(report);report<<",\"selected2_bytes\":";selected_bytes.json(report);report<<",\"selected2_compute_ms\":";selected.json(report);
        report<<",\"selected2_saved_ms\":"<<cold.median()-selected.median();
        report<<",\"full_read_compute_saved_ms\":"<<cold.median()-full.median()<<",\"window_compute_saved_ms\":"<<cold.median()-partial.median()<<",\"output_unchanged\":true}";report.flush();
    }
    const std::array<int32_t,12> routes{0,1,2,3,4,5,0,3,4,6,7,8};const uint32_t masks[]{7,7};
    auto deny=[](void*){return false;};auto start=ReadClock::now();
    const auto canceled=cpu.read_ahead_pair(fixture.bank,routes.data(),masks,.34,nullptr,deny);
    if(canceled)throw std::runtime_error("Urgent request didnotstop idle read");
    report<<"],\"canceled_bytes\":"<<canceled<<",\"immediate_cancel_ms\":"<<read_ms(start)<<",\"passed\":true}";if(!report)throw std::runtime_error("Native read evidence write failed");return 0;
}catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
