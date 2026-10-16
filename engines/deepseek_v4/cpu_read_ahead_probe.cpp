#include "cpu_read_ahead_probe.hpp"
#include "cpu_weight_fixture.hpp"
#include "checkpoint.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace strata::deepseek_v4;
struct Samples {
    std::vector<double> values;
    void add(double value) {values.push_back(value);}
    double median() const {auto sorted=values;std::sort(sorted.begin(),sorted.end());return sorted.at(sorted.size()/2);}
    void json(std::ostream& out) const {
        auto sorted=values;std::sort(sorted.begin(),sorted.end());
        out<<"{\"n\":"<<sorted.size()<<",\"median\":"<<median()<<",\"p10\":"<<sorted[sorted.size()/10]
           <<",\"p90\":"<<sorted[sorted.size()*9/10]<<",\"mean\":"
           <<std::accumulate(sorted.begin(),sorted.end(),0.0)/sorted.size()<<'}';
    }
};
static void same_result(const std::vector<float>& actual,const std::vector<float>& expected) {
    for(size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i])||actual[i]!=expected[i])throw std::runtime_error("Read-only prefetch changed same-pool expert output");
}
int main(int argc,char** argv)try {
    if(argc!=4&&argc!=5)throw std::runtime_error("Expected GGUF shard, activation checkpoint directory, report path, optional owner_hot_only");
    const bool owner_only=argc==5&&std::string(argv[4])=="owner_hot_only";
    const bool stagger_reuse=argc==5&&std::string(argv[4])=="stagger_reuse";
    const auto topology=cpu_ccd_topology();if(topology.size()!=2||topology[0].cores.size()!=8)throw std::runtime_error("Expected two CCDs");
    CpuWeightFixture fixture(argv[1]);const auto checkpoint=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(checkpoint.data.size()<8192*sizeof(float)||checkpoint.metadata.type!=GGML_TYPE_F32)
        throw std::runtime_error("Authentic two-position FFN activation missing");
    std::vector<float> x(8192);std::memcpy(x.data(),checkpoint.data.data(),8192*sizeof(float));
    if(stagger_reuse) {
        Mxfp4CpuService cpu(8);
        const std::vector<int32_t> target{0},other{1,2,3,4,5,6,7,8};
        const auto all=expert_spans(fixture.bank,{0,1,2,3,4,5,6,7,8});
        uint64_t prefault=0;
        for(const auto& span:all)for(size_t i=0;i<span.bytes;i+=4096)prefault+=span.data[i];
        std::vector<float> t(4096),d(4096),expected(4096),scratch(8*4096);
        cpu.compute(fixture.bank,x.data()+4096,target.data(),1,expected.data());
        std::ofstream report(argv[3]);
        report<<std::setprecision(12)
              <<"{\"whole_model_loaded\":false,\"production_worker_placement_unchanged\":true,"
              <<"\"worker_threads\":8,\"real_mxfp4_expert\":true,\"real_two_position_activation\":true,"
              <<"\"expert_bytes\":"<<fixture.bank.gate->nb[2]+fixture.bank.up->nb[2]+fixture.bank.down->nb[2]
              <<",\"prefault_checksum\":"<<prefault<<",\"cases\":[";
        bool first=true;
        for(int interfere:{0,4,6,8}) {
            Samples samples;
            for(int trial=0;trial<32;++trial) {
                flush_spans(all);
                cpu.compute(fixture.bank,x.data(),target.data(),1,t.data());
                if(interfere<=6&&interfere)cpu.compute(fixture.bank,x.data(),other.data(),interfere,scratch.data());
                if(interfere>6) {
                    cpu.compute(fixture.bank,x.data(),other.data(),6,scratch.data());
                    cpu.compute(fixture.bank,x.data(),other.data()+6,interfere-6,scratch.data()+6*4096);
                }
                const auto started=ReadClock::now();
                cpu.compute(fixture.bank,x.data()+4096,target.data(),1,d.data());
                samples.add(read_ms(started));same_result(d,expected);
            }
            if(!first)report<<',';first=false;
            report<<"{\"intervening_experts\":"<<interfere<<",\"second_compute_ms\":";
            samples.json(report);report<<'}';report.flush();
        }
        report<<"],\"same_payload_output_unchanged\":true}";
        if(!report)throw std::runtime_error("Stagger reuse report failed");
        return 0;
    }
    const std::vector<int32_t> one{0},five{0,1,2,3,4},future{16,17,18,19,20};
    const auto single=expert_spans(fixture.bank,one),multiple=expert_spans(fixture.bank,five),other=expert_spans(fixture.bank,future);
    uint64_t prefault=0;for(const auto& spans:{multiple,other})for(const auto& span:spans)
        for(size_t offset=0;offset<span.bytes;offset+=4096)prefault+=span.data[offset];
    DWORD_PTR original=0,system=0;if(!GetProcessAffinityMask(GetCurrentProcess(),&original,&system))throw std::runtime_error("Process affinity query failed");
    std::ofstream out(argv[3]);out<<std::setprecision(12)<<"{\"whole_model_loaded\":false,\"production_pool_modified\":false,\"activation\":\"real ff n norm layer0 two positions\",\"expert_bytes\":"
        <<fixture.bank.gate->nb[2]+fixture.bank.up->nb[2]+fixture.bank.down->nb[2]<<",\"prefault_checksum\":"<<prefault
        <<",\"gpu_window_ms\":0.34,\"cases\":[";
    for(int config=0;config<(owner_only?1:2);++config) {
        DWORD_PTR mask=original;if(config){mask=0;for(const auto& core:topology[0].cores)mask|=core.mask;}
        if(!SetProcessAffinityMask(GetCurrentProcess(),mask))throw std::runtime_error("Test-only process affinity failed");
        {
            Mxfp4CpuService cpu(config?4:8);ReadAheadProbePool readers(topology[0],4);
            std::vector<float> output(12*4096),expected(12*4096);std::array<int32_t,12> ids{0,1,2,3,4,5,0,1,2,3,4,5};
            const uint32_t masks[2]{31,31};
            auto compute=[&](int count) {std::fill(output.begin(),output.end(),0);const auto start=ReadClock::now();
                if(count==1)cpu.compute(fixture.bank,x.data(),one.data(),1,output.data());
                else if(cpu.compute_pair(fixture.bank,x.data(),ids.data(),masks,output.data())!=5)throw std::runtime_error("Five-expert union changed");
                return read_ms(start);};
            Samples cold1,hot1,cold5,hot5,partial5,pre1,pre5,window_ms,window_bytes,concurrent,concurrent_read,owner1,owner5;
            for(int count:{1,5}) {
                const auto& spans=count==1?single:multiple;compute(count);expected=output;
                for(int trial=0;trial<32;++trial) {
                    // Interleave treatments; cache flushes and page faults are outside compute timing.
                    flush_spans(spans);const double cold=compute(count);
                    const double owner_hot=compute(count);same_result(output,expected);
                    (count==1?cold1:cold5).add(cold);
                    (count==1?owner1:owner5).add(owner_hot);
                    if(owner_only)continue;
                    flush_spans(spans);readers.start(spans);const auto read=readers.wait();
                    const double hot=compute(count);same_result(output,expected);
                    (count==1?hot1:hot5).add(hot);(count==1?pre1:pre5).add(read.first);
                    if(count==5) {
                        flush_spans(spans);readers.start(spans,0.34);const auto limited=readers.wait();
                        window_ms.add(limited.first);window_bytes.add(double(limited.second));partial5.add(compute(count));same_result(output,expected);
                        flush_spans(spans);flush_spans(other);readers.start(other);
                        concurrent.add(compute(count));same_result(output,expected);concurrent_read.add(readers.wait().first);
                    }
                }
            }
            if(config)out<<',';out<<"{\"configuration\":\""<<(config?"ccd0_4_test_only_cache_ceiling":"best8_unbound_unchanged")
                <<"\",\"cpuid_ccd0_l3_bytes\":"<<readers.l3_bytes()<<",\"cold_single_ms\":";cold1.json(out);
            out<<",\"same_worker_hot_single_ms\":";owner1.json(out);
            out<<",\"same_worker_hot_five_union_ms\":";owner5.json(out);
            if(owner_only) {
                out<<",\"cold_five_union_ms\":";cold5.json(out);
                out<<",\"reader_jobs_executed\":0,\"same_input_payload_output_unchanged\":true}";out.flush();continue;
            }
            out<<",\"warm_single_ms\":";hot1.json(out);out<<",\"read_single_ms\":";pre1.json(out);
            out<<",\"cold_five_union_ms\":";cold5.json(out);out<<",\"warm_five_union_ms\":";hot5.json(out);
            out<<",\"read_five_ms\":";pre5.json(out);out<<",\"window_read_elapsed_ms\":";window_ms.json(out);
            out<<",\"window_read_bytes\":";window_bytes.json(out);out<<",\"partial_five_union_ms\":";partial5.json(out);
            out<<",\"concurrent_five_union_ms\":";concurrent.json(out);out<<",\"concurrent_read_ms\":";concurrent_read.json(out);
            out<<",\"concurrent_cpu_slowdown_fraction\":"<<concurrent.median()/cold5.median()-1
               <<",\"partial_compute_saved_ms\":"<<cold5.median()-partial5.median()
               <<",\"same_input_payload_output_unchanged\":true}";out.flush();
        }
        if(!SetProcessAffinityMask(GetCurrentProcess(),original))throw std::runtime_error("Probe affinity restore failed");
    }
    out<<"],\"passed\":true}";if(!out)throw std::runtime_error("Read-ahead receipt write failed");return 0;
}catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
