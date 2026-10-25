"use strict";
// Each rate keeps its measurement scope; no planned tokens / elapsed estimate.
globalThis.StrataSpeed = {
  render({live, last, history}, {fmt, setMetric, spark}) {
    const valid = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
    const reading = live.state === "reading";
    const generating = live.state === "generating";
    const active = reading || generating;
    const lastPP = last && last.prompt_ms > 0
      ? Math.max(0, last.prompt_tokens - (last.reused || 0)) * 1000 / last.prompt_ms : null;
    const ppMean = active ? live.prefill_tok_s_mean : lastPP;
    const ppLive = reading ? live.prefill_tok_s : null;
    const tgLive = generating ? live.tok_s : null;
    const tgMean = generating ? live.tok_s_mean : last?.decode_tok_s;
    const meanText = value => valid(value) ? `mean ${fmt(value, 1)} t/s` : "mean —";
    let ppLabel = active ? "PP · this request" : "PP · last request";
    if (reading) ppLabel = valid(ppLive) ? "PP · live chunk"
      : valid(ppMean) ? "PP · native mean" : "PP · measuring first chunk";
    const pp = valid(ppLive) ? ppLive : valid(ppMean) ? ppMean : null;
    const tg = valid(tgLive) ? tgLive : !active && valid(tgMean) ? tgMean : null;
    setMetric("speed", tg == null ? null : fmt(tg, 1), "t/s",
      `${generating ? "TG · live" : "TG · last request"} · ${meanText(tgMean)}`);
    setMetric("prefill", pp == null ? null : fmt(pp, 1), "t/s",
      `${ppLabel} · ${meanText(ppMean)}`);
    spark("sp-speed", history.tok_s);
    spark("sp-prefill", history.prefill_tok_s || history.prefill_tok_s_mean);
  }
};
