#pragma once
#include "serve_sampling.hpp"
#include <cerrno>
#include <charconv>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_set>

namespace strata::deepseek_v4 {
struct ServeRequest : ServeSampling {
    int limit=0;
    std::string embeddings_path;
    std::vector<int32_t> ids;
};
inline bool serve_unsigned(const std::string& text,uint64_t& number) {
    if(text.empty())return false;
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),number);
    return parsed.ec==std::errc()&&parsed.ptr==text.data()+text.size();
}
inline bool serve_number(const std::string& text,double& number) {
    if(text.empty())return false;
    char* end=nullptr;errno=0;number=std::strtod(text.c_str(),&end);
    return end!=text.c_str()&&*end=='\0'&&errno!=ERANGE&&std::isfinite(number);
}
inline bool serve_ids(const std::string& text,std::vector<int32_t>& ids,bool image) {
    ids.clear();if(text.empty()||text.back()==',')return false;
    size_t first=0;
    while(first<text.size()) {
        const size_t end=text.find(',',first);
        const auto word=text.substr(first,end==std::string::npos?end:end-first);
        if(image&&word=="-1")ids.push_back(-1);
        else {
            uint64_t value=0;
            if(!serve_unsigned(word,value)||value>uint64_t(std::numeric_limits<int32_t>::max()))return false;
            ids.push_back(int32_t(value));
        }
        if(end==std::string::npos)break;
        first=end+1;
    }
    return !ids.empty();
}
inline bool parse_serve_request(const std::string& line,ServeRequest& out,std::string& error) {
    out=ServeRequest{};error.clear();
    const bool image=line.rfind("GENI ",0)==0;
    if(!image&&line.rfind("GEN ",0)!=0) {
        error="expected GEN <max_new> [sampling options] <ids> or GENI <max_new> [options] <file> <ids>";return false;
    }
    std::istringstream stream(line.substr(image?5:4));std::string word,id_text;
    uint64_t limit=0;
    if(!(stream>>word)||!serve_unsigned(word,limit)||limit<1||limit>262144) {
        error="max_new must be in 1..262144";return false;
    }
    out.limit=int(limit);std::unordered_set<std::string> seen;
    while(stream>>word) {
        const auto equals=word.find('=');
        if(equals==std::string::npos) {
            if(image){out.embeddings_path=word;stream>>id_text;}else id_text=word;
            break;
        }
        const auto key=word.substr(0,equals),value=word.substr(equals+1);
        if(!seen.insert(key).second){error="duplicate request parameter: "+key;return false;}
        double number=0;uint64_t integer=0;
        if(key=="ignore_eog"||key=="mtp") {
            if(value!="0"&&value!="1"){error=key+" must be 0 or 1";return false;}
            if(key=="ignore_eog")out.ignore_eog=value=="1";else out.mtp=value=="1"?1:0;
        } else if(key=="temperature"||key=="top_p"||key=="min_p") {
            if(!serve_number(value,number)||(key=="temperature"&&number<0)||
               (key=="top_p"&&(number<=0||number>1))||(key=="min_p"&&(number<0||number>1))) {
                error="invalid sampling value for "+key;return false;
            }
            if(key=="temperature")out.temperature=number;
            else if(key=="top_p")out.top_p=number;else out.min_p=number;
        } else if(key=="top_k"||key=="penalty_last_n") {
            if(!serve_unsigned(value,integer)||integer>262144){error="invalid request integer for "+key;return false;}
            if(key=="top_k")out.top_k=int(integer);else out.penalty_last_n=int(integer);
        } else if(key=="penalty_repeat"||key=="penalty_freq"||key=="penalty_present") {
            if(!serve_number(value,number)||(key=="penalty_repeat"&&number<=0)) {
                error="invalid penalty value for "+key;return false;
            }
            if(key=="penalty_repeat")out.penalty_repeat=number;
            else if(key=="penalty_freq")out.penalty_freq=number;else out.penalty_present=number;
        } else if(key=="seed") {
            if(!serve_unsigned(value,integer)){error="invalid sampling seed";return false;}
            out.seed=integer;out.seed_given=true;
        } else {error="unknown request parameter: "+key;return false;}
    }
    if((image&&out.embeddings_path.empty())||!serve_ids(id_text,out.ids,image)) {
        error="bad prompt IDs";return false;
    }
    if(stream>>word){error="unexpected trailing request data";return false;}
    return true;
}
// Resolve before prefill. Explicit MTP never silently changes a target policy.
inline bool resolve_serve_mtp(const ServeRequest& request,bool available,bool& enabled,std::string& error,
                              bool sampled_target=false) {
    enabled=false;
    if(request.mtp==0)return true;
    // Only GLM's sample-and-match target supports stochastic/penalized MTP.
    // Other engines keep the ordinary sampler when their MTP is greedy-only.
    const bool text=!request.ignore_eog&&request.embeddings_path.empty();
    if(request.temperature>0) {
        enabled=sampled_target&&available&&text&&request.limit>1;return true;
    }
    const bool eligible=(sampled_target||!request.penalties())&&text;
    if(request.mtp==1&&(!available||!eligible)) {
        error=!available?"MTP payload is not loaded":"MTP requires greedy text sampling, neutral penalties and normal EOG";
        return false;
    }
    enabled=available&&eligible&&request.limit>1;return true;
}
} // namespace strata::deepseek_v4
