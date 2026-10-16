// Non-owning per-column input views for canonical row reuse.
template<typename F> struct MultiInputReuse {
    struct Input { const Q81Block* x; int iqs; };
    __device__ static Input load(const typename F::W&, const Q81Block* x, int iqs) {
        return {x, iqs};
    }
    __device__ static float apply(const typename F::W& w, const Input& in) {
        return F::apply(w, in.x, in.iqs);
    }
};
template<> struct MultiInputReuse<Q6KTraits> {
    using W = Q6KTraits::W;
    struct Input { int u[2]; float d8[2]; };
    __device__ static Input load(const W& r, const Q81Block* __restrict__ x, int iqs) {
        Input in;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            in.u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            in.d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return in;
    }
    __device__ static float apply(const W& r, const Input& in) {
        return q6_q8_dot_impl(r.vl, r.vh, in.u, r.scales, r.d, in.d8);
    }
};
template<> struct MultiInputReuse<Q4KTraits> {
    using W = Q4KTraits::W;
    struct Input { int u[4]; float d8[2]; };
    __device__ static Input load(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        Input in;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            in.d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            in.u[2 * i] = q8[0];
            in.u[2 * i + 1] = q8[4];
        }
        return in;
    }
    __device__ static float apply(const W& r, const Input& in) {
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, in.u, sc, sc + 2, r.dm, in.d8);
    }
};
