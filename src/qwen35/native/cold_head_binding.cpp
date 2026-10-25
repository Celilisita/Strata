#include "strata/qwen35/native/cold_head_binding.hpp"
#include "strata/qwen35/native/weights.hpp"
#include "strata/qwen35/native/nvfp4_runtime.hpp"
#include "fp4_packed.hpp"
#include <cmath>
#include <cstring>
#include <limits>
namespace strata::qwen35::native {
namespace {
void metadata(const strata::GgufFile &f,const char *name,const std::string &value){
    const auto *m=f.get(name);
    if(!m||m->type!=strata::MetaType::STRING||m->s!=value)
        throw std::invalid_argument(std::string("cold head metadata: ")+name);
}
bool sha256(const std::string &value) {
    if (value.size() != 64) return false;
    bool nonzero = false;
    for (const unsigned char c : value) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        nonzero |= c != '0';
    }
    return nonzero;
}
ColdHeadBinding historical_source(const strata::GgufFile *fixture) {
    ColdHeadBinding source;
    source.fixture = fixture; source.source_bytes = 15740217984ull;
    source.source_model = "Qwen3.8-27B-original-derived-NVFP4.gguf";
    source.source_model_sha256 = "0611180fd2368fa914739b4037435dad27cff76118b8036b0d770814a2079d46";
    source.source_head_sha256 = "2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0";
    source.source_header_sha256 = "289085adcaca7960964ce6a11da0c5587acf75a0b4ac39e1b3123c29e6745f3b";
    source.source_norm_sha256 = "b3ba4f3231a21b1559f52115d3bffc7c28dcd03e4be22df269ef97a3f5d4bfbf";
    return source;
}
uint64_t aligned(uint64_t bytes, uint64_t alignment) {
    if (bytes > (std::numeric_limits<uint64_t>::max)() - (alignment - 1))
        throw std::invalid_argument("cold head directory extent overflow");
    return (bytes + alignment - 1) & ~(alignment - 1);
}
void directory(const strata::GgufFile &base, const strata::GgufFile &head,
               const strata::TensorInfo &weight, const strata::TensorInfo &scale,
               const strata::TensorInfo &norm) {
    const auto alignment = head.alignment();
    if (!alignment || alignment > 4096 || (alignment & (alignment - 1)) ||
        alignment != base.alignment())
        throw std::invalid_argument("cold head directory alignment");
    const auto weight_bytes = strata::qwen35::tensor_bytes(head, weight);
    const auto norm_bytes = strata::qwen35::tensor_bytes(head, norm);
    strata::qwen35::tensor_bytes(head, scale);
    if (weight.offset || scale.offset != aligned(weight_bytes, alignment) ||
        norm.offset != aligned(scale.offset + 4, alignment) || norm_bytes != 20480 ||
        head.data_start() > head.file_size() ||
        norm.offset > head.file_size() - head.data_start() ||
        norm_bytes != head.file_size() - head.data_start() - norm.offset)
        throw std::invalid_argument("cold head directory payload extent");
}
}
void validate_cold_head_source_identity(const ColdHeadBinding &binding) {
    if (!binding.has_source_identity()) return;
    if (!binding.source_bytes || binding.source_bytes > (uint64_t(1) << 50) ||
        binding.source_model.empty() || binding.source_model.size() > 512 ||
        !sha256(binding.source_model_sha256) || !sha256(binding.source_head_sha256) ||
        !sha256(binding.source_header_sha256) || !sha256(binding.source_norm_sha256))
        throw std::invalid_argument("cold head explicit source identity is incomplete or malformed");
    for (const unsigned char c : binding.source_model)
        if (c < 32 || c == 127 || c == '/' || c == '\\')
            throw std::invalid_argument("cold head source model must be its logical file name");
}
void validate_shared_head_identity(const strata::GgufFile &base,const strata::qwen35::Spec &s){
    if(s.n_embd!=5120||s.n_vocab!=248320||s.mtp.size()!=1||
       s.output!=base.find("output.weight")||s.embedding!=base.find("token_embd.weight")||
       s.output_norm!=base.find("output_norm.weight")||!s.output||!s.embedding||!s.output_norm||
       s.mtp[0].output!=s.output||s.mtp[0].embedding!=s.embedding||!s.mtp[0].output_norm||
       s.mtp[0].output_norm!=base.find(s.mtp[0].output_norm->name)||s.mtp[0].output_norm->type!=0||
       s.mtp[0].output_norm->shape!=std::vector<uint64_t>({5120}))
        throw std::invalid_argument("cold head requires actual shared main/draft output and embedding identities");
}
void validate_cold_head_binding(const strata::GgufFile &base,const strata::qwen35::Spec &s,ColdHeadBinding binding){
    validate_shared_head_identity(base,s);
    if(!binding.fixture)throw std::invalid_argument("cold head requires its pinned fixture");
    validate_cold_head_source_identity(binding);
    const bool historical = !binding.has_source_identity();
    const auto source = historical ? historical_source(binding.fixture) : binding;
    const auto &f=*binding.fixture;const auto *w=f.find("output.weight"),*g=f.find("output.scale"),*n=f.find("output_norm.weight");
    if(base.file_size()!=source.source_bytes||(historical&&f.file_size()!=715183104ull)||f.tensors().size()!=3||
       s.output->type!=14||s.output->shape!=std::vector<uint64_t>({5120,248320})||
       !w||w->type!=40||w->shape!=s.output->shape||!g||g->type!=0||g->shape!=std::vector<uint64_t>({1})||
       !n||n->type!=0||n->shape!=std::vector<uint64_t>({5120})||s.output_norm->type!=0||
       s.output_norm->shape!=n->shape)
        throw std::invalid_argument("cold head fixture directory/source/norm mismatch");
    directory(base, f, *w, *g, *n);
    if (std::memcmp(base.tensor_data(*s.output_norm), f.tensor_data(*n), 20480))
        throw std::invalid_argument("cold head copied norm differs from the actual source bytes");
    metadata(f,"strata.head_fp4.source_model",source.source_model);
    metadata(f,"strata.head_fp4.source_model_sha256",source.source_model_sha256);
    metadata(f,"strata.head_fp4.source_head_sha256",source.source_head_sha256);
    metadata(f,"strata.head_fp4.source_header_sha256",source.source_header_sha256);
    metadata(f,"strata.head_fp4.source_norm_sha256",source.source_norm_sha256);
    float global=0;std::memcpy(&global,f.tensor_data(*g),4);
    if(!(global>0)||!std::isfinite(global))throw std::invalid_argument("cold head global scale");
}
namespace {
Matrix load_packed_head(const strata::GgufFile &file,const strata::TensorInfo &tensor,
        DeviceStorage &owner,Nvfp4View &view,cudaStream_t stream){
#if CUDART_VERSION >= 12080
    Nvfp4View canonical;load_nvfp4_matrix(file,tensor,owner,canonical,stream);
    const auto values=nvfp4_value_bytes(canonical.rows,canonical.columns),sf=nvfp4_scale_bytes(canonical.rows,canonical.columns);
    void *destination=nullptr;
    try{
        destination=owner.allocate(values+sf);auto packed=fp4_pack_values(canonical,destination,stream);
        auto *scales=static_cast<unsigned char*>(destination)+values;
        check_cuda(cudaMemcpyAsync(scales,canonical.scales,sf,cudaMemcpyDeviceToDevice,stream),"cold head immutable SF publication");
        check_cuda(cudaStreamSynchronize(stream),"cold head publication lease");
        view=packed.view;view.scales=scales;
    }catch(...){cudaStreamSynchronize(stream);if(destination)owner.release(destination);owner.release(const_cast<void*>(canonical.values));throw;}
    owner.release(const_cast<void*>(canonical.values));
    return {destination,Quant::NVFP4,view.rows,view.columns,BlockLayout::Nvfp4WarpFragments};
#else
    throw std::runtime_error("NVFP4 shared head requires CUDA >=12.8 and admitted SM120");
#endif
}
}
Matrix load_cold_head_binding(const strata::GgufFile &base,const strata::qwen35::Spec &s,ColdHeadBinding binding,
        DeviceStorage &owner,Nvfp4View &view,cudaStream_t stream){
    validate_cold_head_binding(base,s,binding);
    return load_packed_head(*binding.fixture,*binding.fixture->find("output.weight"),owner,view,stream);
}
Matrix load_nvfp4_shared_head(const strata::GgufFile &file,const strata::qwen35::Spec &s,
        DeviceStorage &owner,Nvfp4View &view,cudaStream_t stream){
    validate_shared_head_identity(file,s);
    if(s.output->type!=40||s.output->shape!=std::vector<uint64_t>({5120,248320})||
       s.output_norm->type!=0||s.output_norm->shape!=std::vector<uint64_t>({5120}))
        throw std::invalid_argument("NVFP4 shared output tensor geometry/norm");
    return load_packed_head(file,*s.output,owner,view,stream);
}
}
