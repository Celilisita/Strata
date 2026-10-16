#pragma once
// Owned shared operand and MMA routines, included after Shared.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ inline void mma_step(unsigned ax,unsigned ay,unsigned bx,unsigned by,float (&sum)[8]){
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(sum[0]),"+f"(sum[1]),"+f"(sum[2]),"+f"(sum[3]),
                   "+f"(sum[4]),"+f"(sum[5]),"+f"(sum[6]),"+f"(sum[7])
                 : "r"(ax),"r"(ay),"r"(bx),"r"(by));
}
template<int Matrices,int TileRows>
__device__ inline void grouped_dot16(const Shared<Matrices,TileRows> &s,int group,int part,int warp,int lane,
                                   float (&sum)[Matrices][8]){
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=(warp/2)*16+(matrix/2)*8+local,token=(warp%2)*16+(matrix%2)*8+local;
    const int word=group*16+part*8;
    if constexpr(Matrices==1){
        #pragma unroll
        for(int half=0;half<2;++half){
            const uint4 a=detail::half_operand4(&s.weight[0][0][0],row,word+half*4);
            const uint4 b=detail::input_operand4<true>(&s.input[0][0],token,word+half*4);
            mma_step(a.x,a.y,b.x,b.y,sum[0]);mma_step(a.z,a.w,b.z,b.w,sum[0]);
        }
        return;
    }
    const uint4 b0=detail::input_operand4<true>(&s.input[0][0],token,word),b1=detail::input_operand4<true>(&s.input[0][0],token,word+4);
    uint4 a0[Matrices],a1[Matrices];
    #pragma unroll
    for(int m=0;m<Matrices;++m){a0[m]=detail::half_operand4(&s.weight[m][0][0],row,word);a1[m]=detail::half_operand4(&s.weight[m][0][0],row,word+4);}
    // Independent matrix accumulators alternate at each K4.
    #pragma unroll
    for(int step=0;step<4;++step){
        const unsigned bx=step==0?b0.x:step==1?b0.z:step==2?b1.x:b1.z;
        const unsigned by=step==0?b0.y:step==1?b0.w:step==2?b1.y:b1.w;
        #pragma unroll
        for(int m=0;m<Matrices;++m){
            const unsigned ax=step==0?a0[m].x:step==1?a0[m].z:step==2?a1[m].x:a1[m].z;
            const unsigned ay=step==0?a0[m].y:step==1?a0[m].w:step==2?a1[m].y:a1[m].w;
            mma_step(ax,ay,bx,by,sum[m]);
        }
    }
}
__device__ inline void tensor_bias8(const Shared<1> &s,int warp,int lane,float (&sum)[8]){
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=(warp/2)*16+(matrix/2)*8+local,token=(warp%2)*16+(matrix%2)*8+local;
    const auto *a=reinterpret_cast<const unsigned *>(s.coefficient[0][row]);
    #pragma unroll
    for(int step=0;step<2;++step){
        const int g=step*4;
        const unsigned bx=(s.ds[g][token]>>16)|(s.ds[g+1][token]&0xffff0000u);
        const unsigned by=(s.ds[g+2][token]>>16)|(s.ds[g+3][token]&0xffff0000u);
        mma_step(a[step*2],a[step*2+1],bx,by,sum);
    }
}
#endif
