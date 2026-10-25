#pragma once
// Included once inside the native namespace after the owned Q4 kernels.
void configure_small_prefill_q4_pair(){
    int device=0;check_cuda(cudaGetDevice(&device),"paired Q4 setup device");const auto caps=capabilities(device);
    if(caps.major!=7||caps.minor!=0||caps.multiprocessors!=80||caps.shared_per_block<sizeof(Shared<2>))throw std::invalid_argument("paired Q4 requires validated SM70 resources");
    check_cuda(cudaFuncSetAttribute(grouped_q4<2,1,false,false,false,false,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"segmented paired Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<2,1,false,false,false,false,false,true,64,256,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"tiled paired Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<2>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"paired Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<2,1,false,false,false,false,false,false,64,256,false,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"block affine paired Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(paired_q4_tensor_bias,cudaFuncAttributePreferredSharedMemoryCarveout,100),"paired Q4 bias shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"single Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"split single Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<1>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"register Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<0>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"single-block Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<2>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"row-cell two-part Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<3>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"row-cell three-part Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<5>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"split register Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<6>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"balanced register Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"prepared split Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"tensor bias Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5,false,true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"fused tensor bias Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5,true,true,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"scaled integer Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(grouped_q4<1,5,true,true,false,true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"compact scaled Q4 shared policy");
    check_cuda(cudaFuncSetAttribute(register_q4<0,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"small pair Q4 shared policy");
    configured=true;
}
int q4_single_partitions(Matrix weight){
    matrix_bytes(weight);
    if(weight.quant!=Quant::Q4K||weight.layout!=BlockLayout::Q4Pairs)
        throw std::invalid_argument("raw single partition schedule requires Q4Pairs");
    const int blocks=weight.columns/256;
    if(weight.rows<Rows&&blocks<=32)return blocks;
    const int tiles=(weight.rows+127)/128;
    if(weight.rows>=Rows&&tiles<=80&&(int64_t(tiles)*blocks)%80==0){
        const int width=int(int64_t(tiles)*blocks/80);int maximum=0;
        for(int tile=0;tile<tiles;++tile){const int start=tile*blocks;
            maximum=std::max(maximum,(start+blocks-1)/width-start/width+1);}
        if(maximum==2||maximum==3)return maximum;
    }
    return weight.rows>=Rows&&blocks>=6&&blocks%6==0?6:5;
}
template<bool Prepared,bool TensorBias=false,bool Fused=false,bool Scaled=false,bool Compact=false>
void single_linear(Matrix weight,size_t wb,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    const size_t ib=small_prefill_half_input_bytes(weight.columns,tokens),ob=size_t(weight.rows)*tokens*4;
    if(!configured||weight.quant!=Quant::Q4K||weight.layout!=BlockLayout::Q4Pairs||tokens!=32||
       !input||!output||reinterpret_cast<uintptr_t>(weight.data)%16||reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(output)%16||
       detail::split_overlap(input,ib,weight.data,wb)||detail::split_overlap(output,ob,weight.data,wb)||detail::split_overlap(output,ob,input,ib))
        throw std::invalid_argument("single grouped Q4 requires distinct aligned PP32 buffers");
    if(parts){
        const int partitions=!Prepared&&!TensorBias?q4_single_partitions(weight):5;
        const size_t required=partitions*ob+(Fused?size_t((weight.rows+Rows-1)/Rows)*sizeof(unsigned):0);
        if(weight.columns/256<partitions||bytes<required||reinterpret_cast<uintptr_t>(parts)%16||
           detail::split_overlap(parts,required,weight.data,wb)||detail::split_overlap(parts,required,input,ib)||detail::split_overlap(parts,required,output,ob))
            throw std::invalid_argument("single Q4 partitioned service requires distinct aligned sufficient scratch");
        auto *counters=Fused?reinterpret_cast<unsigned *>(static_cast<unsigned char *>(parts)+5*ob):nullptr;
        if constexpr(!Prepared&&!TensorBias){
            if(weight.rows<Rows&&partitions==weight.columns/256)register_q4<0><<<dim3(1,1,partitions),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts));
            else if(partitions==2)register_q4<2><<<dim3((weight.rows+Rows-1)/Rows,1,2),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts));
            else if(partitions==3)register_q4<3><<<dim3((weight.rows+Rows-1)/Rows,1,3),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts));
            else if(partitions==6)register_q4<6><<<dim3((weight.rows+Rows-1)/Rows,1,6),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts));
            else register_q4<5><<<dim3((weight.rows+Rows-1)/Rows,1,5),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts));
        }
        else grouped_q4<1,5,Prepared,TensorBias,Fused,Scaled,Compact><<<dim3((weight.rows+Rows-1)/Rows,1,5),Threads,0,stream>>>(weight,weight,static_cast<const detail::HalfAffineInputBlock *>(input),static_cast<float *>(parts),nullptr,nullptr,false,nullptr,Fused?output:nullptr,counters);
        check_cuda(cudaGetLastError(),"owned partitioned Q4 projection");if constexpr(Fused)return;
        const size_t vectors=ob/sizeof(float4);
        if constexpr(!Prepared&&!TensorBias){
            if(weight.rows<Rows&&partitions==weight.columns/256){
                reduce_small_cells<<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(parts),output,vectors,partitions);
                check_cuda(cudaGetLastError(),"owned small Q4 cell reduction");return;
            }
        }
        if(partitions==2)reduce_single<2><<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(parts),output,vectors);
        else if(partitions==3)reduce_single<3><<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(parts),output,vectors);
        else if(partitions==6)reduce_single<6><<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(parts),output,vectors);
        else reduce_single<5><<<unsigned((vectors+Threads-1)/Threads),Threads,0,stream>>>(static_cast<const float *>(parts),output,vectors);
        check_cuda(cudaGetLastError(),"owned partitioned Q4 reduction");return;
    }
    if constexpr(Prepared||TensorBias||Fused)throw std::invalid_argument("special Q4 single service requires split5 scratch");
    else{
        register_q4<1><<<unsigned((weight.rows+Rows-1)/Rows),128,0,stream>>>(weight,static_cast<const detail::HalfAffineInputBlock *>(input),output);
        check_cuda(cudaGetLastError(),"owned grouped single Q4 projection");
    }
}
void small_prefill_q4_single_linear(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    single_linear<false>(weight,matrix_bytes(weight),input,output,tokens,stream,parts,bytes);
}
void small_prefill_q4_tensor_bias_linear(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    single_linear<false,true>(weight,matrix_bytes(weight),input,output,tokens,stream,parts,bytes);
}
size_t q4_single_workspace_bytes(Matrix weight){
    matrix_bytes(weight);
    if(weight.quant!=Quant::Q4K||weight.layout!=BlockLayout::Q4Pairs||weight.columns/256<5||weight.rows%4)
        throw std::invalid_argument("fused Q4 workspace requires compact aligned row vectors and five K ranges");
    return size_t(5)*32*weight.rows*4+size_t((weight.rows+Rows-1)/Rows)*sizeof(unsigned);
}
Q4SingleWorkspace initialize_q4_single_workspace(Matrix weight,void *data,size_t bytes,cudaStream_t stream){
    const size_t required=q4_single_workspace_bytes(weight),partial=size_t(5)*32*weight.rows*4;
    int device=0;check_cuda(cudaGetDevice(&device),"cold Q4 workspace device");
    if(!configured)throw std::invalid_argument("fused Q4 workspace requires configured SM70 setup");
    if(!data||reinterpret_cast<uintptr_t>(data)%16||bytes<required||detail::split_overlap(data,required,weight.data,matrix_bytes(weight)))
        throw std::invalid_argument("fused Q4 initialization requires distinct aligned complete scratch");
    check_cuda(cudaMemsetAsync(static_cast<unsigned char *>(data)+partial,0,required-partial,stream),"cold Q4 producer ticket initialization");
    return {data,bytes,weight,stream,true,device,std::this_thread::get_id()};
}
void small_prefill_q4_fused_bias_linear(Matrix weight,const void *input,float *output,int tokens,cudaStream_t stream,Q4SingleWorkspace lease){
    q4_single_workspace_bytes(weight);
    int device=0;check_cuda(cudaGetDevice(&device),"fused Q4 execution device");
    if(!lease.ready||lease.stream!=stream||lease.device!=device||lease.owner!=std::this_thread::get_id()||lease.weight.data!=weight.data||lease.weight.rows!=weight.rows||lease.weight.columns!=weight.columns)
        throw std::invalid_argument("fused Q4 workspace stream or inventory lease changed");
    single_linear<false,true,true>(weight,matrix_bytes(weight),input,output,tokens,stream,lease.data,lease.bytes);
}
void small_prefill_q4_half_linear(Q4HalfCodeMatrix weight,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    const size_t wb=q4_half_code_matrix_bytes(weight);
    // Geometry descriptor is private to this kernel; never passed to a raw Matrix operator.
    single_linear<true>({weight.data,Quant::Q4K,weight.rows,weight.columns,BlockLayout::Q4Pairs},wb,input,output,tokens,stream,parts,bytes);
}
void small_prefill_q4_scaled_linear(Q4ScaledHalfCodeMatrix scaled,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    const auto weight=scaled.storage;const size_t wb=q4_half_code_matrix_bytes(weight);
    single_linear<true,true,false,true>({weight.data,Quant::Q4K,weight.rows,weight.columns,BlockLayout::Q4Pairs},wb,input,output,tokens,stream,parts,bytes);
}
void small_prefill_q4_compact_scaled_linear(Q4ScaledCodeMatrix weight,const void *input,float *output,int tokens,cudaStream_t stream,void *parts,size_t bytes){
    const size_t wb=q4_scaled_code_matrix_bytes(weight);
    single_linear<true,true,false,true,true>({weight.data,Quant::Q4K,weight.rows,weight.columns,BlockLayout::Q4Pairs},wb,input,output,tokens,stream,parts,bytes);
}
template<bool Tiled>
void q4_pair_linear(Matrix gate,Matrix up,const void *input,float *gout,float *uout,float *activation,int tokens,cudaStream_t stream,bool materialize,void *q6_codes,int segmented_processors,bool block_affine=false,bool tensor_bias=false){
    const size_t gb=matrix_bytes(gate),ub=matrix_bytes(up),ib=small_prefill_half_input_bytes(gate.columns,tokens),ob=size_t(gate.rows)*tokens*4;
    if(q6_codes){
        const size_t cb=small_prefill_half_input_bytes(gate.rows,tokens);
        if(reinterpret_cast<uintptr_t>(q6_codes)%16||detail::split_overlap(q6_codes,cb,input,ib)||
           detail::split_overlap(q6_codes,cb,gate.data,gb)||detail::split_overlap(q6_codes,cb,up.data,ub)||
           detail::split_overlap(q6_codes,cb,gout,ob)||detail::split_overlap(q6_codes,cb,uout,ob)||
           detail::split_overlap(q6_codes,cb,activation,ob))throw std::invalid_argument("fused Q6 producer requires distinct aligned complete descriptor");
    }
    if(!configured||gate.quant!=Quant::Q4K||up.quant!=Quant::Q4K||
       gate.layout!=BlockLayout::Q4Pairs||up.layout!=BlockLayout::Q4Pairs||gate.rows!=up.rows||gate.columns!=up.columns||tokens!=32||
       !input||!gout||!uout||!activation||reinterpret_cast<uintptr_t>(activation)%16||reinterpret_cast<uintptr_t>(gate.data)%16||reinterpret_cast<uintptr_t>(up.data)%16||
       reinterpret_cast<uintptr_t>(input)%16||reinterpret_cast<uintptr_t>(gout)%16||reinterpret_cast<uintptr_t>(uout)%16||
       detail::split_overlap(input,ib,gate.data,gb)||detail::split_overlap(input,ib,up.data,ub)||
       detail::split_overlap(activation,ob,gout,ob)||detail::split_overlap(activation,ob,uout,ob)||
       detail::split_overlap(activation,ob,input,ib)||detail::split_overlap(activation,ob,gate.data,gb)||detail::split_overlap(activation,ob,up.data,ub)||
       detail::split_overlap(gout,ob,uout,ob)||detail::split_overlap(gout,ob,input,ib)||detail::split_overlap(uout,ob,input,ib)||
       detail::split_overlap(gout,ob,gate.data,gb)||detail::split_overlap(gout,ob,up.data,ub)||
       detail::split_overlap(uout,ob,gate.data,gb)||detail::split_overlap(uout,ob,up.data,ub))
        throw std::invalid_argument("paired Q4 requires equal raw shapes and distinct aligned input/outputs");
    if(tensor_bias){
        if(segmented_processors||!block_affine||Tiled||gate.rows%64)throw std::invalid_argument("tensor bias requires continuous raw Q4 full tiles");
        paired_q4_tensor_bias<<<unsigned(gate.rows/64),Threads,0,stream>>>(gate,up,static_cast<const detail::HalfAffineInputBlock *>(input),gout,uout,activation,materialize,static_cast<detail::HalfInputBlock *>(q6_codes));
    }else if(segmented_processors){
        if(segmented_processors<1)throw std::invalid_argument("invalid paired Q4 processors");
        const int tiles=(gate.rows+127)/128,waves=(tiles+segmented_processors-1)/segmented_processors;
        const int units=100*tiles/(segmented_processors*waves)>=90?tiles:segmented_processors;
        if(units<=0||units>tiles)throw std::invalid_argument("paired Q4 two-interval capacity exceeded");
        grouped_q4<2,1,false,false,false,false,false,true,64,256,false,Tiled><<<unsigned((gate.rows+Rows-1)/Rows),Threads,0,stream>>>(gate,up,static_cast<const detail::HalfAffineInputBlock *>(input),gout,uout,activation,materialize,static_cast<detail::HalfInputBlock *>(q6_codes),nullptr,nullptr,units);
    }else{
        if constexpr(Tiled)throw std::invalid_argument("tiled Q4 requires admitted segmented schedule");
        else if(block_affine)grouped_q4<2,1,false,false,false,false,false,false,64,256,false,false,true><<<unsigned((gate.rows+Rows-1)/Rows),Threads,0,stream>>>(gate,up,static_cast<const detail::HalfAffineInputBlock *>(input),gout,uout,activation,materialize,static_cast<detail::HalfInputBlock *>(q6_codes));
        else grouped_q4<2><<<unsigned((gate.rows+Rows-1)/Rows),Threads,0,stream>>>(gate,up,static_cast<const detail::HalfAffineInputBlock *>(input),gout,uout,activation,materialize,static_cast<detail::HalfInputBlock *>(q6_codes));
    }
    check_cuda(cudaGetLastError(),"owned paired Q4 projection");
}
void small_prefill_q4_pair_linear(Matrix gate,Matrix up,const void *input,float *gout,float *uout,float *activation,int tokens,cudaStream_t stream,bool materialize,void *q6_codes,int segmented_processors,bool block_affine,bool tensor_bias){
    if(block_affine&&segmented_processors)throw std::invalid_argument("block affine pair requires independent continuous K");
    q4_pair_linear<false>(gate,up,input,gout,uout,activation,tokens,stream,materialize,q6_codes,segmented_processors,block_affine,tensor_bias);
}
void small_prefill_q4_tiled_pair(Q4TileCodes gate,Q4TileCodes up,const void *input,float *gout,float *uout,float *activation,cudaStream_t stream,bool materialize,void *q6_codes,int processors){
    q4_tile_codes_bytes(gate);q4_tile_codes_bytes(up);
    // Internal geometry carrier only; these private views never escape to raw operators.
    q4_pair_linear<true>({gate.data,Quant::Q4K,gate.rows,gate.columns,BlockLayout::Q4Pairs},
        {up.data,Quant::Q4K,up.rows,up.columns,BlockLayout::Q4Pairs},input,gout,uout,activation,32,stream,materialize,q6_codes,processors);
}
