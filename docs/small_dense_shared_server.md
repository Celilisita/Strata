# Small dense families in the shared Strata server

`family: small_dense` selects the source-built `strata-dense-engine` through
`serve.server`, its existing HTTP/SSE handlers, configuration, authentication,
request FIFO, sampling defaults, metrics, cancellation and unload/reload lifecycle.
The independent historical `tools/dense_server.py` remains readable source; it is
not the factory used by this integration.

The variants are `minicpm5_2b` (GGUF llama / minicpm5 tokenizer, text only) and
`qwen35_08b` (GGUF qwen35 / qwen35 tokenizer, optional real SDV1/DVI1 images).
These are separate from 35B MoE and 27B Dense families. Both variants use one
`strata::dense` source implementation across the compiled GGML backends.

`tools/configure_small_dense.py` writes an identity-bound candidate configuration.
It accepts `--variant`, `--exe`, `--model`, `--gpu-uuid`, `--temp-dir`, `--output`,
optional `--context`, and the paired `--vision-exe` / `--mmproj`. The helper hashes
files and reads GGUF headers; it does not launch a decoder, encoder or HTTP server.
Its output is a new file below the explicit task directory. No config is generated
or activated by CMake. The ordinary shared-server command consumes that config:
`python -m serve.server --engine strata --config <candidate-config>`.

The model's actual GGUF carries the tokenizer and template. No unrelated pack
vocabulary is substituted. The factory checks the variant's architecture,
pre-tokenizer, template SHA and file binding before a decoder starts. Cheap
path/size/mtime checks at later starts reuse the static full-file SHA binding.
A replaced asset requires a newly generated binding. The Qwen visual projector
has its own exact binding; MiniCPM5 does not advertise image support.

All four optional mechanisms and device greedy sampling default to off. The
configuration helper accepts `--profile sm70` or `--profile sm75` only as explicit
historical selection. The corresponding profiles remain recorded in
`config/profiles/small-dense-frozen.json`; an SM120 build does not inherit their
performance or quality results. Norm fusion retains its Qwen prefill-only scope.
Feature-off uses the accepted original projection, readback and graph path.

The adapter supports the native sampler's temperature/top-k/top-p/repetition
parameters and 32-bit seed. Unsupported min-p/frequency/presence penalties are
rejected instead of silently ignored. Each RUN resets request state. One slot,
CANCEL followed by DONE drain, and actual native stop IDs remain authoritative.
The shared frontend's optional repetition cutoff defaults to zero for this family,
preserving the accepted native EOS/explicit output-limit behavior.

Visual requests use the source-built `strata-dense-vision`, the accepted SDV1 to
DVI1 preparation, the same decoder's RUNV, and the shared FIFO. Request artifacts
are owned by their preparation record and removed by the shared handler's cleanup,
including when generation never starts. Windows task artifacts require an
explicit directory below `Y:\temp`; no model assets are moved.

This source integration has not been exercised with HTTP requests, decoder or
encoder runs. The new source-built outputs have no inherited performance or
quality claim. Historical long tutorial, translation, explicit-thinking and
multi-image quality limits remain in the source acceptance records.
