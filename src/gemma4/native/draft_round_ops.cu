#include "strata/gemma4/native/draft_round.hpp"
#include <cstdint>
#include <stdexcept>

namespace strata::gemma4::native::detail {
namespace {
bool aligned(const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%4==0;}
bool overlaps(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(na>UINTPTR_MAX-x||nb>UINTPTR_MAX-y)throw std::invalid_argument("Gemma218 metadata overflow");
    return x<y+nb&&y<x+na;
}
__global__ void prepare(const int* position,const unsigned* main_invalid,
    DraftRoundState* state,int local_capacity,int global_capacity,int context){
    const int p=*position;
    // Sticky for this request; bind alone clears it. A warm-run failure must
    // survive the following capture/replay just like either real Draft flag.
    state->invalid|=*main_invalid;state->drafts[0]=state->drafts[1]=0;
    assistant::Bounds b{};
    if(p<1||p>=context||p>global_capacity){
        state->invalid|=32u;state->bounds=b;return;
    }
    const int origin=p>local_capacity?p-local_capacity:0;
    b.rope_position=p;b.kv_end=p-1;b.sliding_origin=origin;b.sliding_length=p-origin;
    b.sliding_left=p-1025>origin?p-1025:origin;
    b.global_origin=0;b.global_length=p;state->bounds=b;
}
__global__ void save(const int* proposal,const unsigned* invalid,DraftRoundState* state,int slot){
    const int id=*proposal;state->drafts[slot]=id>=0&&id<262144?id:0;
    atomicOr(&state->invalid,*invalid|(id>=0&&id<262144?0u:32u));
}
}
void prepare_draft_round(const int* position,const unsigned* invalid,DraftRoundState* state,
    int local_capacity,int global_capacity,int context,cudaStream_t stream){
    if(!aligned(position)||!aligned(invalid)||!aligned(state)||!stream||local_capacity<1||
       global_capacity<1||context<2||context>262144||
       overlaps(position,4,state,sizeof(*state))||overlaps(invalid,4,state,sizeof(*state)))
        throw std::invalid_argument("Gemma218 bounds metadata lease");
    prepare<<<1,1,0,stream>>>(position,invalid,state,local_capacity,global_capacity,context);
    primitive::check_cuda(cudaGetLastError(),"Gemma218 GPU explicit bounds");
}
void save_draft_round(const int* proposal,const unsigned* invalid,DraftRoundState* state,
    int slot,cudaStream_t stream){
    if(!aligned(proposal)||!aligned(invalid)||!aligned(state)||!stream||slot<0||slot>1||
       overlaps(proposal,4,state,sizeof(*state))||overlaps(invalid,4,state,sizeof(*state)))
        throw std::invalid_argument("Gemma218 proposal/invalid lease");
    save<<<1,1,0,stream>>>(proposal,invalid,state,slot);
    primitive::check_cuda(cudaGetLastError(),"Gemma218 GPU proposal and invalid aggregation");
}
}
