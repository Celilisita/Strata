#pragma once
#include <string>
#include <vector>
#include "route_trace_protocol.hpp"
#include <nlohmann/json.hpp>
#include "ggml-backend.h"
struct llama_model;
class llama_moe_cache;
namespace strata::glm5_next {
class RouteCapture {
public:
    explicit RouteCapture(ggml_context*);
    ggml_tensor* tensor()const{return tensor_;}
    void initialize();
    void begin();
    void seed(const llama_model&,llama_moe_cache&,const std::vector<uint32_t>& prior);
    void write(const std::string&);
    static bool requested();
private:
    ggml_tensor* tensor_=nullptr;
    bool active_=false;
    nlohmann::ordered_json seed_;
};
}
