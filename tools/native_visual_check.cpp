#include "strata/qwen35/native/visual.hpp"
#include "strata/qwen35/native/attention_ops.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace strata::qwen35::native;
int main() {
    try {
        cudaStream_t stream; check_cuda(cudaStreamCreate(&stream),"visual check stream");
        DeviceStorage storage;
        constexpr int tokens=2,heads=2,dim=256,rot=64;
        const float base=10000000.f;
        const size_t size=tokens*heads*dim;
        std::vector<float> input(size),text(size),image(size);
        for(size_t i=0;i<size;++i)input[i]=std::sin(float(i)*.17f);
        auto x=static_cast<float*>(storage.allocate(size*4));
        auto y=static_cast<float*>(storage.allocate(size*4));
        auto p=static_cast<int*>(storage.allocate(tokens*4));
        auto tab=static_cast<int*>(storage.allocate(16*3*4));
        const int positions[tokens]={5,10};std::vector<int> table(16*3);
        for(int i=0;i<16;++i)for(int a=0;a<3;++a)table[i*3+a]=i;
        check_cuda(cudaMemcpyAsync(x,input.data(),size*4,cudaMemcpyHostToDevice,stream),"check input");
        check_cuda(cudaMemcpyAsync(p,positions,tokens*4,cudaMemcpyHostToDevice,stream),"check positions");
        check_cuda(cudaMemcpyAsync(tab,table.data(),table.size()*4,cudaMemcpyHostToDevice,stream),"check identity table");
        rotary_text(x,p,y,dim,rot,heads,tokens,base,stream);
        check_cuda(cudaMemcpyAsync(text.data(),y,size*4,cudaMemcpyDeviceToHost,stream),"check text output");
        rotary_visual(x,p,y,dim,rot,heads,tokens,base,stream,tab);
        check_cuda(cudaMemcpyAsync(image.data(),y,size*4,cudaMemcpyDeviceToHost,stream),"check identity output");
        check_cuda(cudaStreamSynchronize(stream),"check identity completion");
        float identity_error=0;
        for(size_t i=0;i<size;++i)identity_error=(std::max)(identity_error,std::abs(text[i]-image[i]));
        if(identity_error!=0)throw std::runtime_error("equal-axis visual rotary changed text math");
        table[5*3+1]=7;table[5*3+2]=9;table[10*3+1]=12;table[10*3+2]=14;
        check_cuda(cudaMemcpyAsync(tab,table.data(),table.size()*4,cudaMemcpyHostToDevice,stream),"check 2D table");
        rotary_visual(x,p,y,dim,rot,heads,tokens,base,stream,tab);
        check_cuda(cudaMemcpyAsync(image.data(),y,size*4,cudaMemcpyDeviceToHost,stream),"check visual output");
        check_cuda(cudaStreamSynchronize(stream),"check visual completion");
        for(float value:image)if(!std::isfinite(value))throw std::runtime_error("nonfinite visual rotary output");
        double maximum_error=0;const float step=std::pow(base,-2.f/rot);
        for(int row=0;row<tokens*heads;++row)for(int pair=0;pair<dim/2;++pair){
            const size_t offset=size_t(row)*dim;
            if(pair>=rot/2){
                const int d=rot+(pair-rot/2)*2;
                if(image[offset+d]!=input[offset+d]||image[offset+d+1]!=input[offset+d+1])
                    throw std::runtime_error("visual rotary changed unrotated channels");
                continue;
            }
            const double angle=table[positions[row/heads]*3+pair%3]*std::pow(double(step),pair);
            const double a=input[offset+pair],b=input[offset+pair+rot/2];
            maximum_error=(std::max)(maximum_error,std::abs(image[offset+pair]-(a*std::cos(angle)-b*std::sin(angle))));
            maximum_error=(std::max)(maximum_error,std::abs(image[offset+pair+rot/2]-(a*std::sin(angle)+b*std::cos(angle))));
        }
        if(maximum_error>0.00005)throw std::runtime_error("visual rotary differs from same-input 2D formula");
        DenseVisualInput visual;visual.width=5120;visual.positions={{{0,0,0}},{{1,1,1}},{{2,2,2}},{{2,3,2}},{{4,4,4}}};
        VisualSpan span{2,2,std::vector<float>(2*5120)};
        for(size_t i=0;i<span.rows.size();++i)span.rows[i]=float(i%97)*.01f;
        visual.spans.push_back(std::move(span));visual.upload(16,stream);
        std::vector<float> overlay(3*5120,-7.f);
        auto dest=static_cast<float*>(storage.allocate(overlay.size()*4));
        check_cuda(cudaMemcpyAsync(dest,overlay.data(),overlay.size()*4,cudaMemcpyHostToDevice,stream),"check overlay input");
        visual.overlay(dest,1,3,stream);
        check_cuda(cudaMemcpyAsync(overlay.data(),dest,overlay.size()*4,cudaMemcpyDeviceToHost,stream),"check overlay result");
        std::vector<int> extended(16*3);
        check_cuda(cudaMemcpyAsync(extended.data(),visual.table,extended.size()*4,cudaMemcpyDeviceToHost,stream),"check extended positions");
        check_cuda(cudaStreamSynchronize(stream),"check overlay completion");
        for(int i=0;i<5120;++i)if(overlay[i]!=-7.f)throw std::runtime_error("visual span overwrote text embedding");
        for(size_t i=0;i<visual.spans[0].rows.size();++i)
            if(overlay[5120+i]!=visual.spans[0].rows[i])throw std::runtime_error("visual payload row changed");
        if(extended[5*3]!=5||extended[15*3+2]!=15)throw std::runtime_error("decode visual position extension changed");
        check_cuda(cudaStreamDestroy(stream),"visual check stream release");
        std::cout<<"{\"passed\":true,\"identity_max_error\":"<<identity_error
                 <<",\"visual_max_error\":"<<maximum_error
                 <<",\"payload_rows_exact\":true,\"decode_position_extension_exact\":true}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
