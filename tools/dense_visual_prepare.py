"""Expand real SDV1 image records through the GGUF chat template into native DVI1.

This prepares decoder inputs only. The image encoder must have produced SDV1
already; this tool neither encodes images nor runs a text model.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import pathlib
import struct
import sys

try:
    import numpy as _np
except ImportError:
    _np = None

from dense_prepare import DenseFrontend


SDV_MAGIC = 0x31564453
DVI_MAGIC = 0x31495644
WIDTH = 1024
INT_MAX = (1 << 31) - 1
SPECIAL_TEXT = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")


def _all_finite(values):
    if _np is not None:
        # View the already validated native-endian array without copying floats.
        return bool(_np.isfinite(_np.frombuffer(values, dtype=_np.float32)).all())
    return all(math.isfinite(value) for value in values)


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)
        self.offset = 0
        self.sha256 = hashlib.sha256()

    def take(self, count):
        if count < 0 or count > self.size - self.offset:
            raise ValueError("truncated SDV1 payload")
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("short read from SDV1")
        self.offset += count
        self.sha256.update(data)
        return data

    def integers(self):
        return struct.unpack("<8I", self.take(32))

    def values(self, code, count):
        values = array.array(code)
        if values.itemsize != (8 if code == "q" else 4):
            raise ValueError("DVI1 requires 32-bit int/float and 64-bit indices")
        values.frombytes(self.take(values.itemsize * count))
        if sys.byteorder != "little":
            values.byteswap()
        return values


def read_sdv(path, vocab_size, context):
    with pathlib.Path(path).open("rb") as stream:
        reader = Reader(stream)
        magic, version, width, n_images, n_records, n_text, n_image, flags = reader.integers()
        if (magic, version, width) != (SDV_MAGIC, 1, WIDTH):
            raise ValueError("SDV1 magic/version/projector width mismatch")
        if flags & ~1 or not 1 <= n_images <= 32:
            raise ValueError("invalid SDV1 file flags or image count")
        if not n_records or n_records > n_text + n_image or n_text + n_image > context:
            raise ValueError("SDV1 record/token totals exceed context or are empty")
        if n_image < n_images:
            raise ValueError("SDV1 is missing image embeddings")
        groups = [[] for _ in range(n_images)]
        seen_text = seen_image = 0
        current_image = 0
        for _ in range(n_records):
            kind, image, chunk, tokens, n_pos, source_w, source_h, record_flags = reader.integers()
            if image == current_image + 1 and groups[current_image]:
                current_image = image
            if image != current_image or image >= n_images or chunk != len(groups[image]):
                raise ValueError("SDV1 image/chunk indices must be contiguous and increasing")
            if kind not in (0, 1) or not 1 <= tokens <= context or not 1 <= n_pos <= tokens:
                raise ValueError("invalid SDV1 record kind/token/position count")
            if not 1 <= source_w <= INT_MAX or not 1 <= source_h <= INT_MAX:
                raise ValueError("invalid SDV1 source image dimensions")
            if groups[image] and (source_w, source_h) != groups[image][0]["source_size"]:
                raise ValueError("SDV1 source dimensions changed within one image")
            record = {"kind": kind, "image_index": image, "chunk_index": chunk,
                      "token_count": tokens, "position_count": n_pos,
                      "source_size": (source_w, source_h), "flags": record_flags}
            if kind == 0:
                if record_flags or n_pos != tokens:
                    raise ValueError("TEXT records must have zero flags and sequential positions")
                ids = reader.values("i", tokens)
                if any(token < 0 or token >= vocab_size for token in ids):
                    raise ValueError("SDV1 text token outside the GGUF vocabulary")
                record["ids"] = ids
                seen_text += tokens
            else:
                if record_flags & 2:
                    raise ValueError("noncausal visual chunks are not supported by the dense decoder")
                if record_flags != 1:
                    raise ValueError("Qwen SDV1 IMAGE record must provide M-RoPE positions only")
                coords = reader.values("I", tokens * 4)
                for start in range(0, len(coords), 4):
                    t, x, y, z = coords[start:start + 4]
                    if t != 0 or z != 0 or x >= n_pos or y >= n_pos:
                        raise ValueError("invalid relative Qwen still-image coordinates")
                embeddings = reader.values("f", tokens * WIDTH)
                if not _all_finite(embeddings):
                    raise ValueError("non-finite SDV1 visual embedding")
                record.update(coords=coords, embeddings=embeddings)
                seen_image += tokens
            if seen_text > n_text or seen_image > n_image:
                raise ValueError("SDV1 records exceed their declared token totals")
            groups[image].append(record)
        if seen_text != n_text or seen_image != n_image or reader.offset != reader.size:
            raise ValueError("SDV1 totals or end-of-file mismatch")
        if any(not group or not any(record["kind"] == 1 for record in group) for group in groups):
            raise ValueError("SDV1 image has no IMAGE record")
        return groups, {"path": str(pathlib.Path(path).resolve()),
                        "sha256": reader.sha256.hexdigest(), "version": version,
                        "width": width, "images": n_images, "records": n_records,
                        "text_tokens": n_text, "image_tokens": n_image,
                        "gpu_encoder": bool(flags & 1),
                        "finite_validation": "numpy_isfinite" if _np is not None else "python_math_isfinite"}


def special_ids(frontend):
    tokenizer = frontend.tokenizer
    cached = getattr(frontend, "_dense_visual_special_ids", None)
    if cached is not None and cached[0] is tokenizer:
        return list(cached[1])
    result = []
    for text in SPECIAL_TEXT:
        # Tokenizer.__init__ rejects duplicate vocabulary strings before exposing
        # this map; reuse that invariant instead of scanning the full vocabulary.
        token = tokenizer.ids.get(text)
        if token is None:
            raise ValueError(f"vocabulary must contain exactly one {text}")
        if tokenizer.encode(text, parse_special=True) != [token]:
            raise ValueError(f"tokenizer does not recognize the special token {text}")
        result.append(token)
    frontend._dense_visual_special_ids = (tokenizer, tuple(result))
    return result


def render_messages(frontend, source, max_new, context):
    if isinstance(source, list):
        messages, options = source, {}
    elif isinstance(source, dict) and isinstance(source.get("messages"), list):
        messages = source["messages"]
        options = {key: source[key] for key in ("template_kwargs", "tools", "add_generation_prompt")
                   if key in source}
    else:
        raise ValueError("messages JSON must be a list or an object with a messages list")
    return frontend.prepare(messages, max_new=max_new, context=context, **options)


def expand(prepared, groups, triple, context, max_new):
    initial = prepared["input_ids"]
    images = prepared["metadata"]["image_count"]
    if images != len(groups):
        raise ValueError(f"template messages have {images} images but SDV1 has {len(groups)}")
    matches = [index for index in range(len(initial) - 2) if initial[index:index + 3] == triple]
    if len(matches) != images or any(initial.count(token) != images for token in triple):
        raise ValueError("each image must render one exact consecutive vision_start/image_pad/vision_end triple")
    ids = array.array("i")
    axes = [array.array("i") for _ in range(4)]
    image_indices = array.array("q")
    embeddings = array.array("f")
    logical = cursor = 0
    spans = []

    def append_text(tokens):
        nonlocal logical
        if logical + len(tokens) > INT_MAX:
            raise ValueError("logical text position overflow")
        ids.extend(tokens)
        positions = range(logical, logical + len(tokens))
        for axis in axes[:3]:
            axis.extend(positions)
        axes[3].extend([0] * len(tokens))
        logical += len(tokens)

    for image, marker in enumerate(matches):
        append_text(initial[cursor:marker])
        for record in groups[image]:
            physical = len(ids)
            before = logical
            if record["kind"] == 0:
                append_text(record["ids"])
            else:
                count = record["token_count"]
                if logical + record["position_count"] > INT_MAX:
                    raise ValueError("logical image position overflow")
                ids.extend([triple[1]] * count)
                image_indices.extend(range(physical, physical + count))
                embeddings.extend(record["embeddings"])
                coords = record["coords"]
                for row in range(count):
                    t, x, y, z = coords[4 * row:4 * row + 4]
                    axes[0].append(logical + t)
                    axes[1].append(logical + y)
                    axes[2].append(logical + x)
                    axes[3].append(z)
                logical += record["position_count"]
            spans.append({key: value for key, value in record.items()
                          if key not in ("coords", "embeddings", "ids")} |
                         {"physical_start": physical, "logical_start": before})
            if len(ids) + max_new > context or logical + max_new > context:
                raise ValueError("expanded image prompt plus max-new exceeds physical/logical context")
        cursor = marker + 3
    append_text(initial[cursor:])
    if len(ids) + max_new > context or logical + max_new > context:
        raise ValueError("expanded prompt plus max-new exceeds physical/logical context")
    if (len(image_indices) * WIDTH != len(embeddings) or any(len(axis) != len(ids) for axis in axes)
            or any(axes[3])
            or any(right <= left for left, right in zip(image_indices, image_indices[1:]))):
        raise ValueError("invalid expanded DVI1 ownership or row layout")
    return ids, axes, image_indices, embeddings, logical, spans


def little_bytes(values):
    if sys.byteorder == "little":
        return values.tobytes()
    copy = values[:]
    copy.byteswap()
    return copy.tobytes()


def prepare_frontend(frontend, messages, sdv, output, max_new, context, max_output=None):
    """Reuse resident tokenizer metadata; max_new=None uses the expanded input's actual room."""
    gguf = frontend.path
    if (frontend.md.get("general.architecture") != "qwen35" or
            frontend.md.get("qwen35.embedding_length") != WIDTH):
        raise ValueError("DVI1 image preparation requires qwen35 hidden width 1024")
    trained = frontend.md.get("qwen35.context_length")
    if not isinstance(trained, int) or context > trained or context < 256 or context % 256:
        raise ValueError("context must be a trained-context multiple of 256")
    if max_new is not None and (type(max_new) is not int or max_new < 1 or max_new >= context):
        raise ValueError("max-new must be positive and smaller than context")
    if max_output is not None and (type(max_output) is not int or max_output < 1):
        raise ValueError("max-output must be a positive integer")
    reserve = max_new if max_new is not None else 1
    source = json.loads(pathlib.Path(messages).read_text(encoding="utf-8-sig"))
    prepared = render_messages(frontend, source, reserve, context)
    triple = special_ids(frontend)
    groups, sdv_metadata = read_sdv(sdv, len(frontend.tokenizer.tokens), context)
    ids, axes, rows, embeddings, logical, spans = expand(prepared, groups, triple, context, reserve)
    if max_new is None:
        max_new = min(max_output or context, context-len(ids), context-logical)
        if max_new < 1:
            raise ValueError("expanded visual prompt leaves no output room")
    prepared["max_new_tokens"] = max_new
    target = pathlib.Path(output).resolve()
    sidecar = target.with_name(target.name + ".json")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if target in (pathlib.Path(messages).resolve(), pathlib.Path(sdv).resolve(), pathlib.Path(gguf).resolve()):
        raise ValueError("output must not replace a source asset")
    partial = target.with_name(target.name + ".partial")
    if any(path.exists() for path in (target, sidecar, partial)):
        raise ValueError("output, sidecar or partial already exists")
    metadata = dict(prepared)
    metadata.update({"format": "DVI1", "version": 1, "output": str(target),
                     "messages_source": str(pathlib.Path(messages).resolve()), "source_request": source,
                     "sdv": sdv_metadata, "template_input_ids": prepared["input_ids"],
                     "input_ids": ids.tolist(), "prompt_tokens": len(ids), "image_rows": len(rows),
                     "image_indices": rows.tolist(), "position_axes": ["t", "y", "x", "z"],
                     "positions": [axis.tolist() for axis in axes], "next_logical_position": logical,
                     "chunks": spans, "validation": "prepared only; encoder and native decoding not run here"})
    metadata["metadata"] = dict(prepared["metadata"], requires_visual_expansion=False,
                                 real_visual_rows=True, template_prompt_tokens=prepared["prompt_tokens"])
    try:
        with partial.open("xb") as stream:
            stream.write(struct.pack("<8I", DVI_MAGIC, 1, WIDTH, len(ids), len(rows), 4, logical, 0))
            for values in (ids, *axes, rows, embeddings):
                stream.write(little_bytes(values))
        partial.replace(target)
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        partial.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return {"output": str(target), "sidecar": str(sidecar), "physical_tokens": len(ids),
            "image_rows": len(rows), "next_logical_position": logical,
            "max_new_tokens": max_new, "context": context}


def prepare(gguf, messages, sdv, output, max_new, context):
    return prepare_frontend(DenseFrontend(gguf), messages, sdv, output, max_new, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "messages", "sdv", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-new", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.gguf, args.messages, args.sdv, args.output, args.max_new, args.context)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, struct.error) as error:
        print(f"dense_visual_prepare: {error}", file=sys.stderr)
        raise SystemExit(1)
