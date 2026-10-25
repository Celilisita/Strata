#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35moe::nvfp4_prefix_detail {
inline int parse(const char* value){
    if(!value||!std::strcmp(value,"256"))return 256;
    if(!std::strcmp(value,"64"))return 64;
    if(!std::strcmp(value,"128"))return 128;
    throw std::invalid_argument("NVFP4_PP_PREFIX_ROWS requires64|128|256|64,128");
}
struct LayerRows {int recurrent,attention;};
inline LayerRows parse_rows(const char* value){
    if(value&&!std::strcmp(value,"64,128"))return {64,128};
    const int count=parse(value);return {count,count};
}
inline int rows(bool recurrent){
    static const LayerRows value=parse_rows(std::getenv("STRATA_QWEN35_NVFP4_PP_PREFIX_ROWS"));
    return recurrent?value.recurrent:value.attention;
}
// A CPU submission scope, not an asynchronous pointer/ownership token. No state
// survives unwinding and no other request thread can inherit the small dispatcher.
inline thread_local int dense_rows=0;
class DenseScope {
public:
    explicit DenseScope(int rows):previous_(dense_rows){
        if(previous_||(rows!=64&&rows!=128&&rows!=256))throw std::invalid_argument("nested/invalid NV prefix dense scope");
        dense_rows=rows;
    }
    ~DenseScope(){dense_rows=previous_;}
    DenseScope(const DenseScope&)=delete;DenseScope& operator=(const DenseScope&)=delete;
private:int previous_;
};
inline bool small(int count){return count==dense_rows&&(count==64||count==128);}
}
