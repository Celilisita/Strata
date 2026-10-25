#pragma once
// Fixed sixteen packed32bit fields/thread; constant row access cannot require a dynamic local-array index.
namespace strata::qwen35::native::kreg_detail {
struct PackedK {
    unsigned p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15;
    template<int Pair> __device__ __forceinline__ void set(__half a,__half b){
        const unsigned bits=unsigned(__half_as_ushort(a))|(unsigned(__half_as_ushort(b))<<16);
        if constexpr(Pair==0)p0=bits;
        else if constexpr(Pair==1)p1=bits;
        else if constexpr(Pair==2)p2=bits;
        else if constexpr(Pair==3)p3=bits;
        else if constexpr(Pair==4)p4=bits;
        else if constexpr(Pair==5)p5=bits;
        else if constexpr(Pair==6)p6=bits;
        else if constexpr(Pair==7)p7=bits;
        else if constexpr(Pair==8)p8=bits;
        else if constexpr(Pair==9)p9=bits;
        else if constexpr(Pair==10)p10=bits;
        else if constexpr(Pair==11)p11=bits;
        else if constexpr(Pair==12)p12=bits;
        else if constexpr(Pair==13)p13=bits;
        else if constexpr(Pair==14)p14=bits;
        else if constexpr(Pair==15)p15=bits;
    }
    template<int Row> __device__ __forceinline__ __half get()const{
        unsigned bits=0;
        if constexpr(Row/2==0)bits=p0;
        else if constexpr(Row/2==1)bits=p1;
        else if constexpr(Row/2==2)bits=p2;
        else if constexpr(Row/2==3)bits=p3;
        else if constexpr(Row/2==4)bits=p4;
        else if constexpr(Row/2==5)bits=p5;
        else if constexpr(Row/2==6)bits=p6;
        else if constexpr(Row/2==7)bits=p7;
        else if constexpr(Row/2==8)bits=p8;
        else if constexpr(Row/2==9)bits=p9;
        else if constexpr(Row/2==10)bits=p10;
        else if constexpr(Row/2==11)bits=p11;
        else if constexpr(Row/2==12)bits=p12;
        else if constexpr(Row/2==13)bits=p13;
        else if constexpr(Row/2==14)bits=p14;
        else if constexpr(Row/2==15)bits=p15;
        return __ushort_as_half(static_cast<unsigned short>(Row%2?bits>>16:bits));
    }
};
static_assert(sizeof(PackedK)==64,"K cache must remain sixteen packed32bit words");
template<int Pair,class P> __device__ __forceinline__ void load(PackedK &cache,P &p,GdnInputs x,GdnShape s,int query,int first,int tid){
    constexpr int row=Pair*2;float a=0,b=0;
    if(first+row<s.tokens)a=x.k[(size_t(first+row)*s.query_heads+query)*128+tid];
    if(first+row+1<s.tokens)b=x.k[(size_t(first+row+1)*s.query_heads+query)*128+tid];
    const __half lo=__float2half_rn(a),hi=__float2half_rn(b);cache.template set<Pair>(lo,hi);
    p.qk[32*136+row*136+tid]=lo;p.qk[32*136+(row+1)*136+tid]=hi;
    if constexpr(Pair<15)load<Pair+1>(cache,p,x,s,query,first,tid);
}
template<int Row,class P> __device__ __forceinline__ void rhs(P &p,const PackedK &cache,GdnInputs x,GdnShape s,int head,int first,int tid,__half *low){
    const int actual=first+Row;float k=0,v=0;
    if(actual<s.tokens){k=p.beta_exp[Row]*__half2float(cache.template get<Row>());v=p.beta[Row]*x.v[(size_t(actual)*s.value_heads+head)*128+tid];}
    const int at=Row*264+tid;p.input[at]=__float2half_rn(k);low[at]=__float2half_rn(k-__half2float(p.input[at]));
    p.input[at+128]=__float2half_rn(v);low[at+128]=__float2half_rn(v-__half2float(p.input[at+128]));
    if constexpr(Row<31)rhs<Row+1>(p,cache,x,s,head,first,tid,low);
}
template<int Row,class P> __device__ __forceinline__ void qk(P &p,const PackedK &cache,GdnInputs x,GdnShape s,int query,int first,int tid){
    float q=0;if(first+Row<s.tokens)q=x.q[(size_t(first+Row)*s.query_heads+query)*128+tid];
    p.qk[Row*136+tid]=__float2half_rn(q);p.qk[32*136+Row*136+tid]=cache.template get<Row>();
    if constexpr(Row<31)qk<Row+1>(p,cache,x,s,query,first,tid);
}
}
