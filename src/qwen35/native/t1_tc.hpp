#pragma once
#include <strata/qwen35/native/attention_decode.hpp>
#include <strata/qwen35/native/kv_ram.hpp>
#include <array>
namespace strata::qwen35::native {
constexpr int T1TcParts=128,T1TcKeys=64;
constexpr size_t T1TcWorkspace=size_t(24)*T1TcParts*258*4;
struct T1TcResource {int registers,blocks_per_sm,binary_version;size_t shared_bytes,local_bytes;};
struct T1TcShape {int valid=0,capacity=0,shift=0;bool selected=false;const DecodeCursor *cursor=nullptr;const int *positions=nullptr;};
struct T1TcDiagnostic {int *codes=nullptr;float *delta=nullptr;};
std::array<T1TcResource,2> t1_tc_resources();
void t1_tc_configure();
bool t1_tc_enabled();
void t1_tc_validate(const float*,const void*,const void*,RamKvView,float*,void*,size_t,T1TcShape,T1TcDiagnostic={});
// One shared body: direct linear Q8 rows or exact selected directory/physical slots. No allocation or selection elision.
void t1_tc_attention(const float*,const void*,const void*,RamKvView,float*,void*,size_t,T1TcShape,cudaStream_t,T1TcDiagnostic={});
}
