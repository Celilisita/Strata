# Native dense image bridge: SDV1

`strata-dense-vision` owns image preprocessing, the vision encoder and the
projector. It opens the Qwen3.5-0.8B text GGUF with `vocab_only=true` for the
tokenizer metadata required by mtmd. It does not create a llama text context or
execute a llama text model graph. Projector and text embedding widths must both
be 1024. The native Strata dense runtime consumes the resulting records.

Build the independent image target with the existing pinned mtmd dependency:

```text
cmake -S tools/vision -B <task-build> -DLLAMA_DIR=<dependency-source> -DSTRATA_VISION_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build <task-build> --config Release --target strata-dense-vision
```

Start one resident encoder per validated GPU allocation:

```text
strata-dense-vision --model <qwen35-0.8b.gguf> --mmproj <mmproj.gguf> --output-dir G:\temp\strata-minicpm-v47-titan-20261014\small-dense --gpu --max-tokens 1024
```

The caller sets `CUDA_VISIBLE_DEVICES` to the authorized Titan V UUID before
starting this process and budgets the encoder together with the native dense
process. `--gpu` enables GPU image encoding; without it the encoder uses CPU.
`--threads N` selects mtmd CPU threads. `--max-tokens N` sets mtmd's image token
cap; omitting it preserves projector metadata defaults. `--max-request-tokens N`
bounds the sum of text wrapper and image tokens across one request (default
131072). At most 32 independent still images are accepted per request. Temporal
frame merging is disabled for these images.

The encoder prints `READY 1024 SDV1` after metadata and vision weights load.
There is no synthetic warmup. The first request therefore includes any first
encoder workspace/kernel cost. Startup time and first request time should be
reported separately.

The stdin protocol uses literal TAB separators. Spaces in paths are preserved;
TAB and newline cannot appear in a path. The output path may be relative to the
required existing `--output-dir`, or an absolute path inside that directory.
Its parent must exist. Existing artifacts and `.partial` files are rejected.
No system temporary directory is used.

```text
ENC<TAB>artifact.sdv<TAB>image-one.png<TAB>image-two.jpg
QUIT
```

Successful requests print:

```text
OK <images> <records> <image_tokens> <text_tokens> <elapsed_ms>
```

The elapsed time includes image file reads, preprocessing, every encoder and
projector pass, finite-value checks and artifact writes. An error prints
`ERR <message>` and keeps the encoder available for another request. A failed
request removes its own partial artifact. Startup errors go to stderr and
return a nonzero exit status.

Each image is tokenized through mtmd's media marker independently, without
adding BOS. Every returned TEXT and IMAGE chunk is retained in original order.
TEXT chunks contain model-specific image boundary/separator token IDs; callers
must consume these records rather than adding duplicate vision wrappers.
IMAGE chunks include all encoder output rows, even when mtmd produces several
chunks or a layout that is not a rectangular grid. Each record carries its
input image index and local chunk index, so no image boundary is inferred from
token geometry.

All SDV1 fields are little endian. Floats are IEEE 754 float32. The fixed file
header is eight unsigned 32-bit integers (`<8I` in Python):

| Field | Meaning |
| --- | --- |
| magic | `0x31564453`, file bytes `SDV1` |
| version | `1` |
| embedding_width | `1024` |
| image_count | Number of input image files |
| record_count | Total TEXT and IMAGE records |
| text_tokens | Aggregate TEXT token count |
| image_tokens | Aggregate IMAGE embedding row count |
| flags | Bit 0: GPU image encoder selected; remaining bits zero |

Exactly `record_count` records follow. Every fixed record header is eight
unsigned 32-bit integers (`<8I`):

| Field | Meaning |
| --- | --- |
| kind | `0` = TEXT; `1` = IMAGE |
| image_index | Zero-based input image index |
| chunk_index | Zero-based chunk index within that image's mtmd result |
| token_count | Number of token IDs or embedding rows |
| position_count | `mtmd_input_chunk_get_n_pos(chunk)` |
| source_width | Original bitmap width in pixels |
| source_height | Original bitmap height in pixels |
| flags | IMAGE bit 0: M-RoPE positions; bit 1: noncausal decoder chunk; TEXT zero |

A TEXT payload is `token_count` signed 32-bit token IDs, with no padding.
An IMAGE payload consists of two arrays, with no padding between them:

1. `uint32 positions[token_count][4]` in **t, x, y, z** order, read from
   `mtmd_image_tokens_get_decoder_pos(image_tokens, 0, row)`.
2. `float32 embeddings[token_count][embedding_width]` in token-major order,
   copied from that chunk's real encoder/projector output before the next
   encode call.

Positions are chunk-relative because the public API is called with `pos_0=0`.
For Qwen image decoding, convert the saved order to the native ggml position
axes **t, y, x, z**, stored section-major. Add the chunk's current logical base
to t, y and x; leave the unused z axis at the API value. Advance the logical base
by `position_count` after the chunk. TEXT records use ordinary sequential
positions and advance by their token count. Subsequent text decoding continues
from this logical base, while physical KV indices continue to count every
TEXT token and IMAGE embedding row. The two counters differ for images.

For example, an IMAGE chunk with 16×16 patch rows has 256 physical rows and
usually 16 logical positions. It must advance the KV row count by 256 and the
following text's logical position by 16. Use the exported API values for actual
inputs; the example is not a grid requirement.

Consumers validate header totals, record order, bounded lengths, token IDs,
position overflow, finite embeddings and end-of-file before applying a request.
Consumers that only support causal Qwen image chunks reject IMAGE flag bit 1.
This bridge exports metadata and embeddings; successful image understanding is
verified only after those rows and positions run through Strata's native text
decoder with the normal chat template.
