# DeepSeek V4 Flash Vision in Strata

DeepSeek V4 Flash Vision runs through the same `strata.exe` and
`serve/server.py` entry points used by Strata's other model families. Its
architecture-specific mathematics and expert scheduling live under
`engines/deepseek_v4`; model selection remains in the common executable.
The DeepSeek launcher is `run-strata-deepseek-v4.bat`, and its server config is
`strata-deepseek-v4.json`.

## Image requests

The config loads `mmproj-model-bf16.gguf` from the same directory as the four
main GGUF shards. `strata-vision.exe` uses the model's projector to turn each
image into float embedding rows. The common HTTP service inserts image
placeholders into the DeepSeek chat template, then sends the rows and text
tokens to the DeepSeek module in their original order. Each image is decoded
as a bidirectional image block; following text uses causal attention. Several
adjacent images retain separate block boundaries. The projector currently runs
on CPU so it does not displace the main model's GPU expert cache.

DeepSeek requests have thinking enabled by default. An API request can pass
`"chat_template_kwargs": {"enable_thinking": false}` to disable it. The
template and response parser use the same setting, so the final answer goes
to `content` and thinking text goes to `reasoning_content`.

MTP is used for text requests when available. Image requests use ordinary
decode because the MTP drafter has not been validated with image embeddings.

## Validation and performance

The 5090 one-load HTTP canary used the actual four-shard model and local
`mmproj-model-bf16.gguf`: a solid red image returned `Red`, a text arithmetic
request returned `4`, and two images returned `2`. All three responses had
final `content`; the two image responses also had `reasoning_content`. The
same candidate binaries and service source were then installed in
`Y:\Strata_moe`. Test inputs, logs, build outputs, and backups stay under
`Y:\temp`.

The last measured unified text run before the Vision addition was hot PP
966.31 tok/s and TG 23.995 tok/s on 8192 input and 2048 output tokens. It
exceeded the historical llama.cpp PP baseline of 419.84 tok/s but remained
below the requested 2x TG threshold of 25.97 tok/s (llama.cpp TG 12.98
tok/s). Text PP/TG and full-model Qwen regressions have not been remeasured
after the Vision addition; do not present the earlier rates as a new benchmark.

Correctness checks use real quantized payloads, layer state, and bounded task
quality. Cross-framework token-ID or logits identity is not a requirement.
