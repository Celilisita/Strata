#pragma once
namespace strata::prefill::mmq {
template<class Args>
auto set_tile_hint(Args& args, int value, int) -> decltype(args.J_best = value, void()) {
    args.J_best = value;
}
template<class Args>
void set_tile_hint(Args&, int, ...) {}
}
