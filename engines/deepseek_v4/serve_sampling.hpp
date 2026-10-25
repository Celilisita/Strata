#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::deepseek_v4 {
struct ServeSampling {
    bool ignore_eog=false;
    double temperature=0.0,top_p=1.0,min_p=0.0;
    int top_k=64;
    uint64_t seed=0;
    bool seed_given=false;
    int mtp=-1; // Omitted: eligible default. Explicit0/1: per-request selection.
    double penalty_repeat=1.0,penalty_freq=0.0,penalty_present=0.0;
    int penalty_last_n=64;
    bool penalties()const{return penalty_last_n>0&&(penalty_repeat!=1.0||penalty_freq!=0.0||penalty_present!=0.0);}
};
inline std::mt19937_64 serve_rng(const ServeSampling& s) {
    return std::mt19937_64(s.seed_given?s.seed:std::random_device{}());
}
class ServeHistory {
    int window_=0;
    std::deque<int32_t> recent_;
    std::unordered_map<int32_t,uint32_t> counts_;
public:
    ServeHistory(const ServeSampling& s,const std::vector<int32_t>& prompt):window_(s.penalties()?s.penalty_last_n:0){
        if(window_)for(auto token:prompt)remember(token);
    }
    void remember(int32_t token){
        if(!window_||token<0)return;
        recent_.push_back(token);++counts_[token];
        if(recent_.size()>size_t(window_)){
            const auto old=recent_.front();recent_.pop_front();
            auto it=counts_.find(old);if(!--it->second)counts_.erase(it);
        }
    }
    unsigned count(int32_t token)const{const auto it=counts_.find(token);return it==counts_.end()?0:it->second;}
    float adjust(float logit,int32_t token,const ServeSampling& s)const{
        if(!window_)return logit;
        const unsigned n=count(token);if(!n)return logit;
        double value=logit;
        if(s.penalty_repeat!=1.0)value=value<=0?value*s.penalty_repeat:value/s.penalty_repeat;
        value-=s.penalty_freq*n+s.penalty_present;
        if(!std::isfinite(value)||std::abs(value)>std::numeric_limits<float>::max())
            throw std::invalid_argument("penalties exceed the finite logit range");
        return float(value);
    }
};
// Same ordinary top-k/min-p/top-p sampler, with requested penalties before ranking.
// The neutral path returns the original float logits without history work.
template<class IsEog>
int32_t serve_sample(const float* logits,int32_t count,const ServeSampling& s,
                     const ServeHistory& history,std::mt19937_64& rng,IsEog is_eog){
    if(!logits||count<=0)throw std::runtime_error("no model output logits");
    if(!std::isfinite(s.temperature)||s.temperature<0||s.top_k<0||
       !std::isfinite(s.top_p)||s.top_p<=0||s.top_p>1||
       !std::isfinite(s.min_p)||s.min_p<0||s.min_p>1)
        throw std::invalid_argument("invalid sampling policy");
    struct Candidate{float logit;int32_t id;};
    const auto higher=[](const Candidate&a,const Candidate&b){return a.logit>b.logit||(a.logit==b.logit&&a.id<b.id);};
    const size_t keep=s.top_k?std::min(size_t(s.top_k),size_t(count)):size_t(count);
    std::priority_queue<Candidate,std::vector<Candidate>,decltype(higher)> best(higher);
    for(int32_t id=0;id<count;++id){
        const float raw=logits[id];
        if(std::isnan(raw)||raw==std::numeric_limits<float>::infinity())throw std::runtime_error("non-finite output logits");
        if(!std::isfinite(raw)||(s.ignore_eog&&is_eog(id)))continue;
        const float logit=history.adjust(raw,id,s);
        if(best.size()<keep)best.push({logit,id});
        else if(higher({logit,id},best.top())){best.pop();best.push({logit,id});}
    }
    if(best.empty())throw std::runtime_error("no finite output logits");
    std::vector<Candidate> candidates;candidates.reserve(best.size());
    while(!best.empty()){candidates.push_back(best.top());best.pop();}
    std::sort(candidates.begin(),candidates.end(),higher);
    if(s.temperature==0)return candidates.front().id;
    std::vector<double> weights;weights.reserve(candidates.size());double total=0;
    for(const auto& item:candidates){
        const double weight=std::exp((double(item.logit)-candidates.front().logit)/s.temperature);
        if(weight<s.min_p)break;
        weights.push_back(weight);total+=weight;
    }
    if(weights.empty())throw std::runtime_error("sampling removed every candidate");
    double kept=0;size_t limit=0;
    do{kept+=weights[limit++];}while(limit<weights.size()&&kept/total<s.top_p);
    std::uniform_real_distribution<double> draw(0.0,kept);const double target=draw(rng);double cumulative=0;
    auto chosen=candidates[limit-1].id;
    for(size_t i=0;i<limit;++i){cumulative+=weights[i];if(target<cumulative){chosen=candidates[i].id;break;}}
    return chosen;
}
} // namespace strata::deepseek_v4
