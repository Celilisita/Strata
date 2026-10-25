#pragma once
// Four exact integer dots share the same outer Q4 delta/minimum.
// Included beside the owned Volta MMA helpers; no reference-framework path.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ inline void pair_block_affine(const Shared<2> &s,int chunk,int warp,int lane,
                                       const float (&delta)[2][2],const float (&minimum)[2][2],float (&result)[2][8]){
    float scaled[2][8]{},correction[2][8]{};
    // Bound cross-group operand liveness; retain identical FP32 association.
    #pragma unroll 1
    for(int local=0;local<4;++local){
        const int group=chunk*4+local;
        float dot[2][8]{};
        grouped_dot16(s,local,0,warp,lane,dot);
        grouped_dot16(s,local,1,warp,lane,dot);
        float activation_delta[4],activation_sum[4],scale[2][2],minimum_scale[2][2];
        #pragma unroll
        for(int t=0;t<4;++t){
            const int token=detail::result_token(warp,lane,(t&1)+(t/2)*4);
            const auto *h=reinterpret_cast<const half *>(&s.ds[group][token]);
            activation_delta[t]=__half2float(h[0]);activation_sum[t]=__half2float(h[1]);
        }
        #pragma unroll
        for(int m=0;m<2;++m){
            #pragma unroll
            for(int r=0;r<2;++r){
                const unsigned meta=s.coefficient[m][detail::result_row(warp,lane,r*2)][group];
                scale[m][r]=float(meta&63);minimum_scale[m][r]=float((meta>>6)&63);
            }
        }
        #pragma unroll
        for(int i=0;i<8;++i){
            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
            #pragma unroll
            for(int m=0;m<2;++m){
                scaled[m][i]=__fmaf_rn(activation_delta[t],__fmul_rn(scale[m][r],dot[m][i]),scaled[m][i]);
                correction[m][i]=__fmaf_rn(activation_sum[t],minimum_scale[m][r],correction[m][i]);
            }
        }
    }
    #pragma unroll
    for(int m=0;m<2;++m){
        #pragma unroll
        for(int i=0;i<8;++i){
            const int r=(i/2)%2;
            result[m][i]=__fadd_rn(result[m][i],__fmaf_rn(delta[m][r],scaled[m][i],-__fmul_rn(minimum[m][r],correction[m][i])));
        }
    }
}
#endif
