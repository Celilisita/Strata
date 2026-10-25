#include "strata/dense/sampling.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace strata::dense {
void Sampling::validate() const {
    if (!std::isfinite(temperature)||temperature<0||temperature>2||!std::isfinite(top_p)||top_p<=0||top_p>1||
        !std::isfinite(repeat_penalty)||repeat_penalty<=0||top_k<0||penalty_last_n<0)
        throw std::invalid_argument("invalid dense sampling parameters");
}
int Sampler::next(const std::vector<float> &source) {
    if (source.empty()) throw std::invalid_argument("empty dense logits");
    int best=0;
    for(size_t i=0;i<source.size();++i){
        if(!std::isfinite(source[i]))throw std::runtime_error("nonfinite dense logits");
        if(source[i]>source[size_t(best)])best=int(i);
    }
    if(params_.temperature==0&&(params_.repeat_penalty==1||!params_.penalty_last_n)){
        history_.push_back(best);return best;
    }
    std::vector<float> adjusted;
    const std::vector<float> *values=&source;
    if(params_.repeat_penalty!=1&&params_.penalty_last_n){
        adjusted=source;values=&adjusted;
        const auto n=std::min(history_.size(),size_t(params_.penalty_last_n));
        std::unordered_set<int> seen(history_.end()-n,history_.end());
        for(int id:seen)if(id>=0&&size_t(id)<adjusted.size())
            adjusted[size_t(id)]=float(adjusted[size_t(id)]<0?adjusted[size_t(id)]*params_.repeat_penalty:
                                      adjusted[size_t(id)]/params_.repeat_penalty);
    }
    int picked=-1;
    if(params_.temperature==0){
        picked=int(std::max_element(values->begin(),values->end())-values->begin());
    }else{
        std::vector<int> order(values->size());std::iota(order.begin(),order.end(),0);
        auto greater=[&](int a,int b){return (*values)[size_t(a)]==(*values)[size_t(b)]?a<b:
                                            (*values)[size_t(a)]>(*values)[size_t(b)];};
        if(params_.top_k>0&&size_t(params_.top_k)<order.size()){
            std::nth_element(order.begin(),order.begin()+params_.top_k,order.end(),greater);
            order.resize(size_t(params_.top_k));
        }
        if(params_.top_p<1)std::sort(order.begin(),order.end(),greater);
        const float maximum=(*values)[size_t(*std::max_element(order.begin(),order.end(),
            [&](int a,int b){return (*values)[size_t(a)]<(*values)[size_t(b)];}))];
        std::vector<double> probability(order.size());double sum=0;
        for(size_t i=0;i<order.size();++i){probability[i]=std::exp(((*values)[size_t(order[i])]-maximum)/params_.temperature);sum+=probability[i];}
        if(!(sum>0)||!std::isfinite(sum))throw std::runtime_error("invalid dense sampling distribution");
        if(params_.top_p<1){
            double cumulative=0;size_t keep=0;
            do{cumulative+=probability[keep++];}while(keep<probability.size()&&cumulative<params_.top_p*sum);
            order.resize(keep);probability.resize(keep);sum=cumulative;
        }
        double draw=std::generate_canonical<double,53>(random_)*sum;
        picked=order.back();
        for(size_t i=0;i<order.size();++i){draw-=probability[i];if(draw<0){picked=order[i];break;}}
    }
    history_.push_back(picked);
    return picked;
}
}
