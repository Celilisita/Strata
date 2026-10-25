#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace strata::dense {
struct Command {std::string line;uint64_t epoch=0;};
class Commands {
public:
    Commands():reader_([this]{read();}){}
    ~Commands(){if(reader_.joinable())reader_.join();}
    Command take(){
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock,[this]{return !pending_.empty();});
        auto result=std::move(pending_.front());pending_.pop_front();return result;
    }
    bool cancelled(const Command &command)const{return epoch_.load()!=command.epoch;}
private:
    std::atomic<uint64_t> epoch_{0};
    std::mutex mutex_;std::condition_variable ready_;std::deque<Command> pending_;std::thread reader_;
    void read(){
        std::string line;
        while(std::getline(std::cin,line)){
            if(line=="CANCEL"){epoch_.fetch_add(1);continue;}
            const bool quit=line=="QUIT";
            if(quit||line=="RESET")epoch_.fetch_add(1);
            if(line.size()>16*1024*1024)line="INVALID_LINE_TOO_LARGE";
            {std::lock_guard<std::mutex> lock(mutex_);pending_.push_back({line,epoch_.load()});}
            ready_.notify_one();if(quit)return;
        }
        epoch_.fetch_add(1);
        {std::lock_guard<std::mutex> lock(mutex_);pending_.push_back({"QUIT",epoch_.load()});}
        ready_.notify_one();
    }
};
}
