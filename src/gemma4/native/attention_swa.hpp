#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include "strata/gemma4/native/ffn.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::gemma4::native::swa403 {
constexpr int Heads=32,KvHeads=16,Dim=256,Window=1024,QueryTile=128,KeyTile=512;
constexpr int HeadsPerKv=Heads/KvHeads;
struct Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct Lease {AttentionWorkspace workspace{};Binding binding{};};
// Scale is one, as in the native Gemma attention. K/V remain canonical Q8.
// Supported PP chunks are 8192 or 6144, absolute past+tokens <=262144.
// The caller owns every buffer/handle and keeps them alive until stream join.
std::size_t workspace_bytes();
void attention(const float* query,float* output,CacheView,const int* position,
    int past,int heads,int tokens,Lease,Binding live);

// Host-only optional dispatch, with no Model/Ffn layout change. Construct after
// ALL cold selectors/reset, at P0. Leave scope before the TG controller Graph.
// Kernels receive no pointer to this host object. No active scope means OFF.
class RequestScope {
public:
    RequestScope(Binding live,int model_position,bool enabled);
    ~RequestScope() noexcept;
    RequestScope(const RequestScope&)=delete;
    RequestScope& operator=(const RequestScope&)=delete;
    // Successfully enqueued hot calls, not GPU-executed/replayed kernel count.
    std::size_t enqueues()const noexcept{return enqueues_;}
private:
    Binding binding_{};
    bool enabled_=false;
    std::size_t enqueues_=0;
    RequestScope* previous_=nullptr;
    friend bool request_enabled(Binding);
    friend void record_enqueue(Binding);
};
bool request_enabled(Binding live);
void record_enqueue(Binding live);
}
