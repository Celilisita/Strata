# Cross-request prefix resume (native dense)

## Problem

OpenAI-style clients resend the whole conversation every turn. The native dense
service (`qwen35_dense`: Qwen3.5/3.6/3.8) had no cross-request prefix cache:
`serve/native_service.py` rejects `strata_prefix`, and `DenseService::release()`
called `reset_prefill()` after every request. A 90k-116k prompt was therefore
read from token 0 on every turn, which cost 33-55 s per message.

## Behaviour

`DenseService` keeps `snapshot_` (the previous prompt plus every generated id)
and `resumable_`. On `PREPARE` it computes the longest common prefix with the
new prompt:

- `LCP == snapshot.size()` and `LCP < prompt.size()`: resume at `LCP` (exact).
- otherwise, if `LCP` reaches the most recent recurrent checkpoint: resume at
  that chunk boundary (bounded rewind of at most one prefill chunk).
- otherwise: full prefill.

`release(true)` keeps the KV and recurrence alive. Protocol errors clear the
snapshot and reset; visual (`PREPAREV`) requests never resume, because their
embedding is bound to absolute positions.

## State that survives

- KV: `RamKvCache::resume(rows)` keeps the authoritative host rows and restores
  identity resident rows from them, then rebuilds the directory with
  `ram_reset`. Above the resident window it invalidates the resident directory
  so the decode pass reloads pages on demand.
- Recurrence: GatedDeltaNet state cannot be rewound token by token, so
  `DenseModel::save_checkpoint` copies `state`, `history[0]`, `history[1]` and
  the ping-pong parity at every prefill chunk boundary (skipped during graph
  capture). This costs about 156 MB for the 27B hybrid at 48 recurrent layers.
- Positions and the sampler are rebuilt per request. `DenseModel::prefill` takes
  a `start` argument and the `PP` progress line reports absolute offsets.

## MTP

`NativeMtp::prefill(ids, tokens, stream, offset)` keeps the draft KV and its
positions across requests (`ram_->reset` only runs when `offset == 0`).
`prepare_service_mtp_prefix(..., offset)` takes a resume branch that prefills
only the tail and re-captures the decode graph uncached
(`enable_graphs(..., capture_prefix = false)`), so speculative decode stays
enabled on resumed requests.

## Limits

- One snapshot, slot 0, previous request only.
- Rewind granularity is the 8k prefill chunk; a mismatch inside the last chunk
  re-reads at most 8192 tokens.
- A byte-identical resend resumes at the chunk checkpoint rather than skipping
  all prefill, because the last token's logits are not retained.
- Resident window is 40960 tokens; longer prefixes are read from host memory
  over PCIe on both the full and resumed paths.
- Draft KV may have been remapped by the preceding decode pass, so a resumed
  draft can propose less often. Verification still guarantees the output.

## Measurements

RTX 5090, Qwen3.8-27B NVFP4, `kv_policy = ram8k-reuse`, `mtp = 2`:

| case | full | resumed |
| --- | --- | --- |
| 90k prompt, identical prefix | 33.0 s | 1.72 s |
| 30k branch, 5 divergent tokens | 4.24 s | 1.08 s (rewind to 24576) |
| 116k turn over HTTP | 55 s | prefill 1-2 s, generation ~85 tok/s |

Resumed and full runs matched on the first sampled token and the next seven in
every check (exact, bounded rewind, with and without MTP).

## Files

- `src/qwen35/native/serve.cpp`: snapshot, resume decision, `release(true)`.
- `src/qwen35/native/model.cpp`, `include/strata/qwen35/native/model.hpp`:
  `prefill` start offset, `resume_kv`, checkpoints.
- `src/qwen35/native/kv_ram.cpp`, `src/qwen35/native/kv_ram_write.cu`:
  `RamKvCache::resume`, `ram_invalidate`.
- `src/qwen35/native/mtp_condition.cpp`,
  `src/qwen35/native/service_mtp_prefill.hpp`,
  `include/strata/qwen35/native/mtp.hpp`: draft offset and resume path.
