#include "strata/gemma4/native/decoder_ops.hpp"
#include "strata/gemma4/native/vector3.hpp"
#include "strata/gemma4/native/vector_grouped.hpp"
#include "strata/gemma4/native/vector_single.hpp"
#include "strata/gemma4/native/vector_single_fast.hpp"
#include "strata/gemma4/native/vector_striped_q4.hpp"
#include "strata/gemma4/native/vector_striped_q6_down.hpp"
#include "strata/gemma4/native/vector_paired_word_q4.hpp"
#include "strata/gemma4/native/vector_paired_direct_q4.hpp"
#include "strata/gemma4/native/ffn.hpp"
#include "strata/qwen35/native/ops.hpp"
#include <algorithm>
#include <cmath>
namespace strata::gemma4::native {
Ffn::Ffn(FfnWeights weights,int maximum,float epsilon):weights_(weights),hidden_(weights.gate.columns),
    intermediate_(weights.gate.rows),maximum_(maximum),epsilon_(epsilon){
    const auto valid=[](primitive::Matrix m){return m.data&&m.layout==primitive::BlockLayout::Gguf&&
        (m.quant==primitive::Quant::Q4K||m.quant==primitive::Quant::Q6K);};
    q6_down_weight_bytes(weights.down);
    if(!valid(weights.gate)||!valid(weights.up)||hidden_<256||intermediate_<256||
        hidden_%256||intermediate_%256||weights.up.rows!=intermediate_||weights.up.columns!=hidden_||
        weights.down.rows!=hidden_||weights.down.columns!=intermediate_||maximum<1||maximum>8192||
        !weights.pre_norm||!weights.post_norm||!weights.layer_scale||!std::isfinite(epsilon)||epsilon<=0)
        throw std::invalid_argument("Gemma4 Dense FFN geometry/precision lease");
    // This inventory has no raw Q6 copy to serve another activation policy.
    // Model enforces this for its whole inventory, even after a Q4 rebind.
    if(is_aligned_q6_down(weights_.down))single_ffn_=true;
    auto allocate=[&](size_t n){return static_cast<float*>(storage_.allocate(n*4));};
    auto* normalized=allocate(size_t(maximum)*hidden_);
    auto* gate_up=allocate(size_t(maximum)*intermediate_*2);
    buffers_={normalized,gate_up,gate_up+size_t(maximum)*intermediate_,
        allocate(size_t(maximum)*intermediate_),allocate(size_t(maximum)*hidden_),allocate(size_t(maximum)*hidden_)};
    decoded_=allocate(size_t(hidden_)*intermediate_);workspace_=storage_.allocate(32u*1024*1024);
    if(cublasCreate(&handle_)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma4 cuBLAS create");
    if(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH)!=CUBLAS_STATUS_SUCCESS){cublasDestroy(handle_);handle_=nullptr;throw std::runtime_error("Gemma4 FP32 math policy");}
}
Ffn::~Ffn(){if(handle_)cublasDestroy(handle_);}
void Ffn::inspect_residual_input(float* host,cudaStream_t stream)const{
    if(!host)throw std::invalid_argument("Gemma residual range output");
    primitive::check_cuda(cudaMemcpyAsync(host,workspace_,2*sizeof(float),cudaMemcpyDeviceToHost,stream),"Gemma cold operand range");
    primitive::check_cuda(cudaStreamSynchronize(stream),"Gemma cold operand range join");
}
bool Ffn::prepare_vector_input(const float* input,int columns,int tokens,cudaStream_t stream){
    if(!vector3_||tokens>3)return false;
    prepare_vector3(input,decoded_,size_t(hidden_)*intermediate_*4,columns,tokens,stream);return true;
}
SingleQ8View Ffn::prepare_linear_single(const float* input,int columns,int tokens,cudaStream_t stream){
    return prepare_single_q8(input,decoded_,size_t(hidden_)*intermediate_*sizeof(float),columns,tokens,stream);
}
void Ffn::linear_single(primitive::Matrix matrix,SingleQ8View view,float* output,cudaStream_t stream){
    if(static_cast<const void*>(view.data)!=decoded_||view.tokens<1||view.tokens>3||view.columns!=matrix.columns)
        throw std::invalid_argument("Gemma typed single projection owner/shape");
    linear_single_q8_fast(matrix,view,output,stream);
}
void Ffn::project(primitive::Matrix m,const float* input,float* output,int tokens,cudaStream_t stream,bool eligible,bool prepared){
    if(is_aligned_q6_down(m)&&tokens<=3)
        throw std::invalid_argument("Gemma native Q6 Down is only supported by explicit Single40 TG");
    if(vector3_&&tokens<=3){
        if(!prepared)prepare_vector_input(input,m.columns,tokens,stream);
        if(grouped_)linear_vector_grouped(m,decoded_,output,tokens,stream);
        else linear_vector3(m,decoded_,output,tokens,stream);return;
    }
    if(tokens==1){project_vector(m,input,output,stream);return;}
    if(size_t(m.rows)*m.columns>size_t(hidden_)*intermediate_)throw std::invalid_argument("Gemma shared decode arena exceeded");
    if(pp_algorithm_==2&&tokens>=128&&((residual_pp_&&eligible)||(all_pp_&&!eligible))){
        void* packed=input==buffers_.activation?buffers_.gate:buffers_.activation;
        const int slice=((all_pp_&&!eligible)||(full_down_&&m.data==weights_.down.data))?0:residual_slice_;
        half_library_project(m,input,output,tokens,decoded_,packed,handle_,workspace_,stream,slice);return;
    }
    if(((residual_pp_&&eligible)||(all_pp_&&!eligible))&&tokens>=128){
        // Gate/Up's activation output arena is dead until GELU; Gate is dead
        // once GELU has produced Down's input. No new allocation or aliasing.
        void* packed=input==buffers_.activation?buffers_.gate:buffers_.activation;
        residual_library_project(m,input,output,tokens,decoded_,packed,handle_,workspace_,stream,((all_pp_&&!eligible)||(full_down_&&m.data==weights_.down.data))?0:((residual_slice_==0&&tokens<1280)?1024:residual_slice_),eligible&&tokens>=1280&&(m.data==weights_.gate.data||m.data==weights_.up.data)?pp_algorithm_:-1);return;
    }
    decode_float(m,decoded_,stream);const float one=1.f,zero=0.f;
    if(cublasSetStream(handle_,stream)!=CUBLAS_STATUS_SUCCESS||cublasSetWorkspace(handle_,workspace_,32u*1024*1024)!=CUBLAS_STATUS_SUCCESS||
        cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST)!=CUBLAS_STATUS_SUCCESS||
        cublasSgemm(handle_,CUBLAS_OP_T,CUBLAS_OP_N,m.rows,tokens,m.columns,&one,decoded_,m.columns,
            input,m.columns,&zero,output,m.rows)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma4 FP32 cuBLAS projection");
}
void Ffn::rebind(FfnWeights w){
    const primitive::Matrix matrices[]{w.gate,w.up};
    for(auto m:matrices)if(!m.data||m.layout!=primitive::BlockLayout::Gguf||(m.quant!=primitive::Quant::Q4K&&m.quant!=primitive::Quant::Q6K))
        throw std::invalid_argument("Gemma rebind raw inventory");
    q6_down_weight_bytes(w.down);
    if(is_aligned_q6_down(w.down)&&!single_ffn_)
        throw std::invalid_argument("Gemma native Q6 Down rebind requires Single40 FFN");
    if(w.gate.rows!=intermediate_||w.gate.columns!=hidden_||w.up.rows!=intermediate_||w.up.columns!=hidden_||
       w.down.rows!=hidden_||w.down.columns!=intermediate_||!w.pre_norm||!w.post_norm||!w.layer_scale)throw std::invalid_argument("Gemma rebind shape");
    weights_=w;
}
void Ffn::select_native_pp(bool enabled){if(enabled)configure_native_pair();native_pp_=enabled;}
void Ffn::pair_prefill(const float* normalized,int tokens,cudaStream_t stream){
    if((!paired_pp_&&!native_pp_)||!residual_pp_||!normalized||tokens<1280||tokens>maximum_||!stream)
        throw std::invalid_argument("Gemma paired PP source/extent policy");
    PairPrefillLease lease;
    lease.source=normalized;lease.gate=buffers_.gate;lease.up=buffers_.up;
    lease.weight_arena=decoded_;lease.weight_bytes=size_t(hidden_)*intermediate_*4;
    // Down is dead until all Gate/Up products and GELU are complete.
    lease.input_arena=buffers_.down;lease.input_bytes=size_t(maximum_)*hidden_*4;
    lease.output_stride=size_t(maximum_)*intermediate_;lease.tokens=tokens;
    lease.workspace=workspace_;lease.handle=handle_;lease.stream=stream;
    lease.slice=residual_slice_;lease.algorithm=pp_algorithm_;
    lease.owner=this;lease.epoch=++prepared_epoch_;
    if(native_pp_)native_pair_project(weights_.gate,weights_.up,lease);
    else paired_library_project(weights_.gate,weights_.up,lease);
    gelu_product(buffers_.gate,buffers_.up,buffers_.activation,size_t(tokens)*intermediate_,stream);
}
void Ffn::run(const float* input,float* output,int tokens,cudaStream_t stream){
    if(!input||!output||input==output||!stream||tokens<1||tokens>maximum_)throw std::invalid_argument("Gemma4 FFN request extent");
    primitive::rms_norm(input,weights_.pre_norm,buffers_.normalized,hidden_,tokens,epsilon_,stream);
    if(single_ffn_&&tokens<=3){
        // 202/203 real-weight units retained positive T2/T3 fast paths. Admit
        // them only in this derived verify candidate; ordinary T1 is unchanged.
        // This typed lease is consumed before the arena is overwritten by
        // Down. Public prepared-vector projections keep their old layout.
        const size_t bytes=size_t(hidden_)*intermediate_*sizeof(float);
        const auto gate_input=prepare_single_q8(buffers_.normalized,decoded_,bytes,hidden_,tokens,stream);
        const auto raw_q4=[](primitive::Matrix w){return w.quant==primitive::Quant::Q4K&&w.layout==primitive::BlockLayout::Gguf;};
        const bool striped=tokens==1&&hidden_==5376&&intermediate_==21504;
        if(striped&&t1_q4_policy_!=T1Q4FfnPolicy::Off&&raw_q4(weights_.gate)&&raw_q4(weights_.up)){
            if(t1_gate_up_direct_){
                const PairedQ4WordBinding live{this,++t1_gu291_epoch_,stream};
                pair_single_q4_direct_t1({weights_.gate,weights_.up,buffers_.normalized,gate_input,live},
                    buffers_.gate,buffers_.up,buffers_.activation,live);++t1_gu291_enqueues_;
            }else if(t1_gate_up_word_){
                const PairedQ4WordBinding live{this,++t1_gu269_epoch_,stream};
                pair_single_q4_word_t1({weights_.gate,weights_.up,buffers_.normalized,gate_input,live},
                    buffers_.gate,buffers_.up,buffers_.activation,live);++t1_gu269_enqueues_;
            }else pair_single_q4_striped_t1(weights_.gate,weights_.up,gate_input,buffers_.gate,buffers_.up,buffers_.activation,stream);
            ++t1_q4_gate_up_enqueues_; // Logical244eligible GU, including269/291 substitutions.
        }else if(fast_ffn_)pair_single_q8_fast(weights_.gate,weights_.up,gate_input,buffers_.gate,buffers_.up,buffers_.activation,stream);
        else pair_single_q8(weights_.gate,weights_.up,gate_input,buffers_.gate,buffers_.up,buffers_.activation,stream);
        const auto down_input=prepare_single_q8(buffers_.activation,decoded_,bytes,intermediate_,tokens,stream);
        if(striped&&t1_q6_down_striped_&&is_aligned_q6_down(weights_.down)){
            const Q6DownStripeBinding live{this,++t1_q6_down_epoch_,stream};
            const AlignedQ6View weight{weights_.down.data,q6_down_weight_bytes(weights_.down),hidden_,intermediate_};
            linear_q6_down_striped_t1({weight,down_input,live},buffers_.down,live);
            ++t1_q6_down_enqueues_;
        }else if(striped&&t1_q4_policy_==T1Q4FfnPolicy::Both&&raw_q4(weights_.down)){
            linear_single_q4_striped_t1(weights_.down,down_input,buffers_.down,stream);
            ++t1_q4_down_enqueues_;
        }else if(fast_ffn_)linear_single_q8_fast(weights_.down,down_input,buffers_.down,stream);
        else linear_single_q8(weights_.down,down_input,buffers_.down,stream);
    }else{
        if((row_half_||compact_half_||three_panel_)&&pp_algorithm_==2&&residual_pp_&&tokens==1280){
            row_half_prefill(buffers_.normalized,tokens,stream);
        }else if((paired_pp_||native_pp_)&&residual_pp_&&tokens>=1280){
            pair_prefill(buffers_.normalized,tokens,stream);
        }else if(vector3_&&tokens<=3){
            prepare_vector_input(buffers_.normalized,hidden_,tokens,stream);
            if(grouped_)pair_vector_grouped(weights_.gate,weights_.up,decoded_,buffers_.gate,buffers_.up,buffers_.activation,tokens,stream);
            else pair_vector3(weights_.gate,weights_.up,decoded_,buffers_.gate,buffers_.up,buffers_.activation,tokens,stream);
        }else{
            project(weights_.gate,buffers_.normalized,buffers_.gate,tokens,stream);
            project(weights_.up,buffers_.normalized,buffers_.up,tokens,stream);
            gelu_product(buffers_.gate,buffers_.up,buffers_.activation,size_t(tokens)*intermediate_,stream);
        }
        project(weights_.down,buffers_.activation,buffers_.down,tokens,stream);
    }
    primitive::rms_norm(buffers_.down,weights_.post_norm,buffers_.post,hidden_,tokens,epsilon_,stream);
    residual_scale(buffers_.post,input,weights_.layer_scale,output,size_t(tokens)*hidden_,stream);
}
}
