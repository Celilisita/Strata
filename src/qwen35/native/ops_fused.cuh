#pragma once
// Included inside the owned ops kernel namespace after sum_warp.
__global__ void residual_rms(const float *a,const float *b,const float *w,float *residual_out,
                             float *out,int features,float epsilon) {
    __shared__ float partial[32];
    const size_t base=size_t(blockIdx.x)*features;
    float square=0;
    for(int i=threadIdx.x;i<features;i+=blockDim.x) {
        const float value=__fadd_rn(a[base+i],b[base+i]);
        square=__fmaf_rn(value,value,square);
    }
    square=sum_warp(square);
    if(threadIdx.x%32==0)partial[threadIdx.x/32]=square;
    __syncthreads();
    square=sum_warp(threadIdx.x%32<blockDim.x/32?partial[threadIdx.x%32]:0.0f);
    const float scale=rsqrtf(__fmaf_rn(square,__fdividef(1.0f,float(features)),epsilon));
    for(int i=threadIdx.x;i<features;i+=blockDim.x) {
        const float value=__fadd_rn(a[base+i],b[base+i]);
        residual_out[base+i]=value;
        out[base+i]=__fmul_rn(__fmul_rn(value,scale),w[i]);
    }
}
__global__ void prepared_gdn(const float *mixed,const float *alpha,const float *br,
                            const float *bias,const float *a,float *q,float *k,float *v,
                            float *g,float *beta,int dim,int hq,int hv,int tokens,float epsilon) {
    const int row=int(blockIdx.x)*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=hq*tokens)return;
    const int head=row%hq,token=row/hq,channels=(2*hq+hv)*dim;
    const size_t base=size_t(token)*channels+head*dim;
    float qs=0,ks=0;
    for(int j=lane;j<dim;j+=32) {
        const float x=mixed[base+j],y=mixed[base+dim*hq+j];
        qs=__fmaf_rn(x,x,qs);ks=__fmaf_rn(y,y,ks);
    }
    qs=rsqrtf(fmaxf(sum_warp(qs),epsilon));ks=rsqrtf(fmaxf(sum_warp(ks),epsilon));
    for(int j=lane;j<dim;j+=32) {
        q[size_t(row)*dim+j]=__fmul_rn(mixed[base+j],qs);
        k[size_t(row)*dim+j]=__fmul_rn(mixed[base+dim*hq+j],ks);
    }
    for(int vh=head;vh<hv;vh+=hq) {
        const size_t at=size_t(token)*hv+vh;
        for(int j=lane;j<dim;j+=32)v[at*dim+j]=mixed[size_t(token)*channels+2*dim*hq+vh*dim+j];
        if(!lane) {
            const float x=__fadd_rn(alpha[at],bias[vh]);
            const float softplus=x>20?x:__logf(__fadd_rn(1.0f,__expf(x)));
            g[at]=__fmul_rn(softplus,a[vh]);
            beta[at]=__fdividef(1.0f,__fadd_rn(1.0f,__expf(-br[at])));
        }
    }
}
