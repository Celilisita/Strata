#pragma once
namespace strata::glm5_next {
struct MtpMatchResult {bool accepted=false,continue_generation=false;};
// A greedy proposal spends no target RNG/history. Sample row0 once; on reject
// retain that sample after rollback. Row1 is sampled only after an accepted
// row0 was emitted/committed and generation still has room to continue.
template<class Token,class Sample,class Output,class Reject,class Catchup,class Continue>
MtpMatchResult mtp_sample_and_match(Token draft,Token& pending,Sample&& sample,
                                  Output&& output,Reject&& reject,Catchup&& catchup,Continue&& continuation) {
    const auto target=sample(0);
    if(target!=draft) {
        reject();pending=target;return {false,output(pending)};
    }
    if(!output(target))return {true,false};
    catchup(draft);if(!continuation())return {true,false};
    pending=sample(1);
    return {true,output(pending)};
}
} // namespace strata::glm5_next
