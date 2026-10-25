#include "strata/gemma4/native/pp3_q4_panel.hpp"
#include <limits>
namespace strata::gemma4::native::detail298 {
void validate(primitive::Matrix gate,primitive::Matrix up,const Q4PpPanelLease& lease,Q4PpPanelBinding live){
    constexpr size_t H=5376,I=21504,T=1280;
    const auto& p=lease.operands;const auto b=lease.binding;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream||
       p.stream!=live.stream||!p.handle||p.output.rows!=I||p.output.tokens!=T||
       p.weight_bytes<2*H*I*2||p.input_bytes<T*H*2||p.workspace_bytes!=32u*1024*1024||
       p.output.interleaved_values<2*T*I||p.output.activation_values<T*I||
       p.output.interleaved_values>(std::numeric_limits<size_t>::max)()/4||
       p.output.activation_values>(std::numeric_limits<size_t>::max)()/4)
        throw std::invalid_argument("298 fixed PP3/live lease");
    for(auto m:{gate,up})if(!m.data||m.rows!=I||m.columns!=H||m.quant!=primitive::Quant::Q4K||
       m.layout!=primitive::BlockLayout::Gguf)throw std::invalid_argument("298 rawQ4 PP3 inventory");
    struct Span{const void* p;size_t n,align;};
    const Span spans[]={{gate.data,H*I/256*144,16},{up.data,H*I/256*144,16},
        {p.input,T*H*4,4},{p.weight_arena,p.weight_bytes,16},{p.input_arena,p.input_bytes,16},
        {p.workspace,p.workspace_bytes,256},{p.output.interleaved,p.output.interleaved_values*4,16},
        {p.output.activation,p.output.activation_values*4,16}};
    for(size_t i=0;i<8;++i){
        const auto x=reinterpret_cast<std::uintptr_t>(spans[i].p);
        if(!x||!spans[i].n||x%spans[i].align||spans[i].n>(std::numeric_limits<std::uintptr_t>::max)()-x)
            throw std::invalid_argument("298 span extent/alignment/overflow");
        for(size_t j=0;j<i;++j){const auto y=reinterpret_cast<std::uintptr_t>(spans[j].p);
            if(x<y+spans[j].n&&y<x+spans[i].n)throw std::invalid_argument("298 advertised span alias");}
    }
}
}
