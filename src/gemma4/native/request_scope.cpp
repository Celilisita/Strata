#include "attention_swa.hpp"
#include <stdexcept>

namespace strata::gemma4::native::swa403 {
namespace {thread_local RequestScope* active=nullptr;}
RequestScope::RequestScope(Binding live,int position,bool enabled):binding_(live),enabled_(enabled){
    if(!live.owner||!live.epoch||!live.stream||position!=0)
        throw std::invalid_argument("Gemma403 request scope requires live cold P0");
    previous_=active;active=this;
}
RequestScope::~RequestScope()noexcept {active=previous_;}
bool request_enabled(Binding live){
    if(!active||active->binding_.owner!=live.owner)return false;
    if(active->binding_.epoch!=live.epoch||active->binding_.stream!=live.stream)
        throw std::invalid_argument("Gemma403 stale request scope epoch/stream");
    return active->enabled_;
}
void record_enqueue(Binding live){
    if(request_enabled(live))++active->enqueues_;
}
}
