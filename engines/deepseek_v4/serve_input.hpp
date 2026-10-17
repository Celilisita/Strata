#pragma once
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::deepseek_v4 {
// Only the reader owns descriptor0. Its detached worker owns heap state, so no
// blocked stdio lock or borrowed stack object survives a process-level error.
class ServeInput {
    using Flag=std::shared_ptr<std::atomic<bool>>;
    struct State {
        std::mutex mutex;std::condition_variable ready;
        std::deque<std::pair<std::string,Flag>> lines;
        Flag active;bool eof=false;
    };
    std::shared_ptr<State> state_=std::make_shared<State>();
public:
    ServeInput(){
        const auto state=state_;
        std::thread([state]{
            std::string buffer;char chunk[4096];
            auto line=[&](std::string& value){
                for(;;){
                    const auto end=buffer.find('\n');
                    if(end!=std::string::npos){value.assign(buffer,0,end);buffer.erase(0,end+1);return true;}
#ifdef _WIN32
                    const int n=_read(0,chunk,unsigned(sizeof(chunk)));
#else
                    const auto n=::read(0,chunk,sizeof(chunk));
                    if(n<0&&errno==EINTR)continue;
#endif
                    if(n<=0){if(buffer.empty())return false;value.swap(buffer);buffer.clear();return true;}
                    buffer.append(chunk,size_t(n));
                }
            };
            std::string value;
            while(line(value)){
                if(!value.empty()&&value.back()=='\r')value.pop_back();
                std::lock_guard<std::mutex> lock(state->mutex);
                if(value=="STOP"){
                    auto flag=state->active;
                    if(!flag&&!state->lines.empty())flag=state->lines.front().second;
                    if(flag)flag->store(true);
                    continue;
                }
                const bool quit=value=="QUIT"||value=="EXIT";
                state->lines.emplace_back(std::move(value),std::make_shared<std::atomic<bool>>(false));
                state->ready.notify_one();
                if(quit)break;
            }
            std::lock_guard<std::mutex> lock(state->mutex);state->eof=true;state->ready.notify_one();
        }).detach();
    }
    bool next(std::string& line,Flag& cancel){
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->ready.wait(lock,[&]{return state_->eof||!state_->lines.empty();});
        if(state_->lines.empty())return false;
        auto command=std::move(state_->lines.front());state_->lines.pop_front();
        line=std::move(command.first);cancel=std::move(command.second);state_->active=cancel;return true;
    }
    void complete(const Flag& flag){
        std::lock_guard<std::mutex> lock(state_->mutex);if(state_->active==flag)state_->active.reset();
    }
};
struct ServeInputLease {
    ServeInput& input;std::shared_ptr<std::atomic<bool>> flag;
    ~ServeInputLease(){input.complete(flag);}
};
} // namespace strata::deepseek_v4
