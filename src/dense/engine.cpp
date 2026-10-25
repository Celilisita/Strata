#include "strata/dense/model.hpp"
#include "strata/dense/sampling.hpp"
#include "commands.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}
struct Totals {
    strata::dense::AppendCost pp{}, tg{};
    void add(const strata::dense::AppendCost &c, bool prefill) {
        auto &v = prefill ? pp : tg;
        v.graph_ms+=c.graph_ms;v.input_ms+=c.input_ms;
        v.compute_ms+=c.compute_ms;v.readback_ms+=c.readback_ms;
        v.execute_readback_ms+=c.execute_readback_ms;
        v.compact_async_steps+=c.compact_async_steps;v.norm_fused_nodes+=c.norm_fused_nodes;
        v.input_bytes+=c.input_bytes;v.output_bytes+=c.output_bytes;v.nodes+=c.nodes;
        v.projection_fused_matmuls+=c.projection_fused_matmuls;
    }
};
void print_cost(const char *name, const strata::dense::AppendCost &c) {
    std::cout << ",\"" << name << "\":{\"graph_ms\":" << c.graph_ms
        << ",\"input_ms\":" << c.input_ms << ",\"compute_ms\":" << c.compute_ms
        << ",\"readback_ms\":" << c.readback_ms << ",\"h2d_bytes\":" << c.input_bytes
        << ",\"d2h_bytes\":" << c.output_bytes << ",\"nodes\":" << c.nodes
        << ",\"projection_fused_matmuls\":" << c.projection_fused_matmuls
        << ",\"execute_readback_ms\":" << c.execute_readback_ms
        << ",\"compact_async_steps\":" << c.compact_async_steps
        << ",\"norm_fused_nodes\":" << c.norm_fused_nodes << '}';
}
void generate(strata::dense::Model &model,const std::vector<int> &ids,const std::vector<int> &stops,
         int max_new,int chunk,
         strata::dense::Sampling params,bool device_greedy,const strata::dense::Commands &commands,
         const strata::dense::Command &command,const strata::dense::VisualInput *visual=nullptr) {
    const int n=int(ids.size());
    if(n<1||max_new<1||n>model.context()-max_new||stops.empty()||stops.size()>32)
        throw std::invalid_argument("dense request exceeds context or stop count");
    for(int stop:stops)if(stop<0||uint64_t(stop)>=model.spec().n_vocab)
        throw std::invalid_argument("stop token is outside vocabulary");
    strata::dense::Sampler sampler(params,ids);
    const bool compact_greedy = device_greedy && params.temperature == 0 &&
        (params.repeat_penalty == 1 || params.penalty_last_n == 0);
    model.set_device_greedy(compact_greedy);
    const auto begin = Clock::now();
    model.prepare_request(n+max_new);
    Totals totals;
    std::vector<float> logits;
    int processed=0;
    for (int start = 0; start < n && !commands.cancelled(command); start += chunk) {
        const int end = std::min(n,start+chunk);
        if(visual)logits=model.append_input(visual->slice(start,end),end==n);
        else logits = model.append(std::vector<int>(ids.begin()+start,ids.begin()+end),end==n);
        totals.add(model.last_cost(),true);
        processed=end;
        if(const char *progress=std::getenv("STRATA_DENSE_PP_PROGRESS");progress&&progress[0]=='1') {
            const double elapsed=ms(begin,Clock::now());
            std::cout<<"PP "<<processed<<' '<<n<<' '<<elapsed<<' '
                     <<(elapsed>0?processed*1000.0/elapsed:0)<<'\n'<<std::flush;
        }
    }
    const auto pp_end = Clock::now();
    auto first = pp_end;
    int output = 0;
    std::string finish = "length";
    for (; output < max_new;) {
        if(commands.cancelled(command)){finish="cancelled";break;}
        const int token = compact_greedy ? model.device_greedy_token() : sampler.next(logits);
        ++output;
        std::cout << "TOKEN " << token << '\n' << std::flush;
        if (output == 1) first = Clock::now();
        if (std::find(stops.begin(),stops.end(),token) != stops.end()) {
            finish="stop";break;
        }
        if (output == max_new) break;
        logits = model.append({token});
        totals.add(model.last_cost(),false);
    }
    if(finish=="cancelled")model.reset();
    const auto end = Clock::now();
    const auto &memory = model.memory();
    std::cout << "DONE {\"input_tokens\":" << processed << ",\"planned_input_tokens\":" << n << ",\"output_tokens\":" << output
        << ",\"finish_reason\":\"" << finish << "\",\"pp_ms\":" << ms(begin,pp_end)
        << ",\"tg_ms\":" << ms(first,end) << ",\"ttft_ms\":" << ms(begin,first)
        << ",\"request_ms\":" << ms(begin,end) << ",\"weights_bytes\":" << memory.weights
        << ",\"state_bytes\":" << memory.state << ",\"workspace_bytes\":" << memory.workspace;
    std::cout << ",\"state_capacity\":" << model.state_capacity()
        << ",\"request_kv\":" << (model.request_kv()?"true":"false");
    std::cout << ",\"pp_half_bytes\":" << memory.pp_half << ",\"backend_reserve_bytes\":" << memory.backend_reserve;
    std::cout << ",\"visual\":" << (visual?"true":"false")
        << ",\"norm_fusion_bytes\":" << memory.norm_fusion
        << ",\"cost_timing_scope\":\"sync_compute_or_async_submit_then_join;compare_execute_readback_sum\"";
    std::cout << ",\"device_greedy\":" << (compact_greedy?"true":"false");
    print_cost("pp_cost",totals.pp);print_cost("tg_cost",totals.tg);
    std::cout << "}\n" << std::flush;
}
void run(strata::dense::Model &model,std::istringstream &in,int chunk,strata::dense::Sampling params,
         bool device_greedy,const strata::dense::Commands &commands,const strata::dense::Command &command){
    int maximum=0,n=0,ns=0;
    if(!(in>>maximum>>n>>ns)||n<1||n>model.context()||ns<1||ns>32)
        throw std::invalid_argument("RUN requires max_new input_count stop_count");
    std::vector<int> ids(size_t(n),0),stops(size_t(ns),0);
    for(int &id:ids)if(!(in>>id))throw std::invalid_argument("missing RUN token");
    for(int &id:stops)if(!(in>>id))throw std::invalid_argument("missing RUN stop token");
    std::string extra;if(in>>extra)throw std::invalid_argument("extra RUN input");
    generate(model,ids,stops,maximum,chunk,params,device_greedy,commands,command);
}
void run_visual(strata::dense::Model &model,std::istringstream &in,int chunk,strata::dense::Sampling params,
         bool device_greedy,const strata::dense::Commands &commands,const strata::dense::Command &command,
         const std::filesystem::path &task){
    int maximum=0,ns=0;
    if(!(in>>maximum>>ns)||maximum<1||ns<1||ns>32)
        throw std::invalid_argument("RUNV requires max_new stop_count stops and prepared file");
    std::vector<int> stops(size_t(ns),0);
    for(int &id:stops)if(!(in>>id))throw std::invalid_argument("missing RUNV stop token");
    std::string path;std::getline(in>>std::ws,path);
    const auto absolute=std::filesystem::weakly_canonical(std::filesystem::u8path(path));
    const auto relative=absolute.lexically_relative(task);
    if(path.empty()||relative.empty()||relative.is_absolute()||*relative.begin()=="..")
        throw std::invalid_argument("RUNV input must be inside the configured task directory");
    // C++17 returns char and C++20 char8_t; copy the same UTF-8 code-unit bytes.
    const auto utf8=absolute.u8string();
    const std::string input_path(reinterpret_cast<const char *>(utf8.data()),utf8.size());
    const auto visual=strata::dense::read_visual_input(input_path,model.spec(),model.context(),maximum);
    generate(model,visual.ids,stops,maximum,chunk,params,device_greedy,commands,command,&visual);
}
}

int main(int argc, char **argv) {
    try {
        std::string path;
        auto task=std::filesystem::current_path();
        int context = 131072, chunk = 512, budget_mib = 12000, pp_half = 0, device_greedy = 0, projection_fusion = 0;
        int compact_async = 0, norm_fusion = 0, batch = 0, ubatch = 0;
        bool chunk_given = false, batch_given = false, ubatch_given = false;
        const auto prefill_count = [](const std::string &value) {
            size_t used = 0;
            const int count = std::stoi(value, &used);
            if (used != value.size()) throw std::invalid_argument("invalid dense prefill integer");
            return count;
        };
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (i+1 == argc) throw std::invalid_argument("missing option value");
            const std::string value = argv[++i];
            if (key=="--model") path=value;
            else if (key=="--context") context=std::stoi(value);
            else if (key=="--chunk") {chunk=prefill_count(value);chunk_given=true;}
            else if (key=="--batch") {batch=prefill_count(value);batch_given=true;}
            else if (key=="--ubatch") {ubatch=prefill_count(value);ubatch_given=true;}
            else if (key=="--budget-mib") budget_mib=std::stoi(value);
            else if (key=="--pp-half-cache") pp_half=std::stoi(value);
            else if (key=="--device-greedy") device_greedy=std::stoi(value);
            else if (key=="--projection-fusion") projection_fusion=std::stoi(value);
            else if (key=="--compact-async") compact_async=std::stoi(value);
            else if (key=="--norm-fusion") norm_fusion=std::stoi(value);
            else if (key=="--task-dir") task=std::filesystem::weakly_canonical(std::filesystem::u8path(value));
            else throw std::invalid_argument("unknown dense engine option: "+key);
        }
        if (batch_given || ubatch_given) {
            if (!batch_given || !ubatch_given || batch<1 || batch>8192 || batch!=ubatch ||
                (chunk_given && chunk!=ubatch))
                throw std::invalid_argument("dense requires equal batch/ubatch in1..8192 and matching chunk");
            chunk=ubatch;
        }
        if (path.empty() || chunk<1 || chunk>8192 || budget_mib<1 || budget_mib>12000 || pp_half<0 || pp_half>1 ||
            device_greedy<0 || device_greedy>1 || projection_fusion<0 || projection_fusion>1 ||
            compact_async<0 || compact_async>1 || norm_fusion<0 || norm_fusion>1)
            throw std::invalid_argument("invalid dense engine configuration");
        const auto begin = Clock::now();
        strata::dense::Model model(path,context,size_t(budget_mib)*1024*1024,pp_half!=0,projection_fusion!=0,compact_async!=0,norm_fusion!=0);
        const auto &s = model.spec();
        std::cout << "READY {\"architecture\":\"" << s.architecture << "\",\"layers\":" << s.n_layers
            << ",\"hidden\":" << s.n_embd << ",\"context\":" << context
            << ",\"state_capacity\":" << model.state_capacity()
            << ",\"request_kv\":" << (model.request_kv()?"true":"false")
            << ",\"prefill_batch\":" << chunk << ",\"prefill_ubatch\":" << chunk << ",\"prefill_token_capacity\":8192"
            << ",\"input_uploads\":\"same_stream_async\""
            << ",\"pp_half\":" << pp_half << ",\"pp_half_cold_ms\":" << model.pp_half_cold_ms()
            << ",\"pp_half_bytes\":" << model.memory().pp_half
            << ",\"pp_half_tensors\":" << model.pp_half_tensor_count()
            << ",\"pp_half_status\":\"" << model.pp_half_status() << "\""
            << ",\"device_greedy_requested\":" << device_greedy
            << ",\"projection_fusion\":" << projection_fusion
            << ",\"projection_pack_groups\":" << model.projection_pack_group_count()
            << ",\"projection_pack_members\":" << model.projection_pack_member_count()
            << ",\"compact_async_requested\":" << compact_async
            << ",\"compact_async_available\":" << (model.compact_async_available()?"true":"false")
            << ",\"compact_async_status\":\"" << model.compact_async_status() << "\""
            << ",\"norm_fusion\":" << norm_fusion
            << ",\"norm_fusion_bytes\":" << model.memory().norm_fusion
            << ",\"norm_fusion_scope\":\"prefill_only_backend_eligible_rms_mul\""
            << ",\"device_greedy_tie_break\":\"backend_argmax_not_lowest_id_guaranteed\""
            << ",\"capabilities\":[\"sampling\",\"cancel\"" << (s.architecture=="qwen35"?",\"visual\"":"")
            << "],\"load_ms\":" << ms(begin,Clock::now()) << "}\n" << std::flush;
        strata::dense::Sampling sampling;
        strata::dense::Commands commands;
        for (;;) {
            const auto item=commands.take();
            try {
                std::istringstream in(item.line);std::string command;in>>command;
                if (command=="QUIT") break;
                if (command=="RESET") {model.reset();std::cout<<"RESET\n"<<std::flush;}
                else if(command=="FUSION") {
                    int enabled = -1;std::string extra;
                    if(!(in>>enabled) || enabled<0 || enabled>1 || (in>>extra))
                        throw std::invalid_argument("FUSION requires one 0/1 value");
                    model.set_projection_fusion(enabled!=0);
                    std::cout<<"FUSION "<<enabled<<'\n'<<std::flush;
                }
                else if(command=="COST") {
                    int async = -1, norm = -1;std::string extra;
                    if(!(in>>async>>norm) || async<0 || async>1 || norm<0 || norm>1 || (in>>extra))
                        throw std::invalid_argument("COST requires two 0/1 values");
                    model.set_cost_modes(async!=0,norm!=0);
                    std::cout<<"COST "<<async<<' '<<norm<<'\n'<<std::flush;
                }
                else if(command=="PARAM"){
                    strata::dense::Sampling next;
                    if(!(in>>next.temperature>>next.top_k>>next.top_p>>next.repeat_penalty>>next.penalty_last_n>>next.seed))
                        throw std::invalid_argument("PARAM requires temperature top_k top_p repeat_penalty last_n seed");
                    std::string extra;if(in>>extra)throw std::invalid_argument("extra PARAM input");
                    next.validate();sampling=next;std::cout<<"PARAM\n"<<std::flush;
                }
                else if (command=="RUN") run(model,in,chunk,sampling,device_greedy!=0,commands,item);
                else if (command=="RUNV") run_visual(model,in,chunk,sampling,device_greedy!=0,commands,item,task);
                else throw std::invalid_argument("unknown dense engine command");
            } catch (const std::exception &e) {
                model.reset();std::cout << "ERROR " << e.what() << '\n' << std::flush;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "dense engine: " << e.what() << '\n';return 1;
    }
}
