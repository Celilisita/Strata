#pragma once
#include "strata/gemma4/native/assistant_quant.hpp"
#include <cstddef>
#include <cstdint>

namespace strata::gemma4::native::assistant::head399 {
constexpr int InputValues=1024,OutputValues=262144;
constexpr std::size_t WeightBytes=std::size_t(OutputValues)*(InputValues/32)*18;
struct Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct Lease {Q40Matrix matrix{};std::size_t weight_bytes=0;Binding binding{};};
// Optional operator, never selected by this module. Only the real Assistant
// token_embd.weight Q4_0 head: [262144 rows][1024 columns], 18-byte blocks.
// W=float(raw_half_scale)*float(nibble-8), X=actual F32 features. No input
// quantization, persistent decoded weights, scratch, allocation, or host join.
// Owner must retain weights/input/output and this binding through all work and
// Graph replays. Release external Graphs and join before changing the epoch or
// destroying storage. Host bindings are checked at enqueue/capture, not replay.
void project(const float* input,float* output,Lease,Binding live);
}
