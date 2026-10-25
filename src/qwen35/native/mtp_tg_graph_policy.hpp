#pragma once
#include <algorithm>
#include <cstddef>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
extern char **environ;
#endif
namespace strata::qwen35::native {
struct MtpTgGraphStats {
    bool eligible=false,hit=false,parked=false;size_t misses=0,hits=0,builds=0;
    double validation_ms=0,rearm_ms=0;
};
inline MtpTgGraphStats &mtp_tg_graph_stats(){static thread_local MtpTgGraphStats value;return value;}
inline bool mtp_tg_graph_enabled(){
    const char *raw=std::getenv("STRATA_DENSE_MTP_REUSE_TG_GRAPHS");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("MTP TG graph reuse mode must be auto,0,1");
    return mode!="0";
}
inline std::string mtp_tg_graph_key(const std::string &config){
#ifdef _WIN32
    char **env=_environ;
#else
    char **env=::environ;
#endif
    std::vector<std::string> policy;
    for(;env&&*env;++env){const std::string item=*env;std::string name=item.substr(0,item.find('='));
        for(auto &c:name)c=char(std::toupper(static_cast<unsigned char>(c)));
        if(name.rfind("STRATA_",0)==0&&name!="STRATA_DENSE_MTP_REUSE_TG_GRAPHS")policy.push_back(item);}
    std::sort(policy.begin(),policy.end());std::string key=std::to_string(config.size())+":"+config;
    for(const auto &item:policy)key+=std::to_string(item.size())+":"+item;return key;
}
}
