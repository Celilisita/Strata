"""Qwen rectangular image embeddings and IMRoPE positions at the native boundary."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_image(path: Path, width: int):
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("truncated Qwen image embedding header")
    magic, count, nx, ny, dim = struct.unpack_from("<5i", raw)
    if (magic != 0x31455653 or not 1 <= count <= 16384 or nx < 1 or ny < 1 or
            nx * ny != count or dim != width or len(raw) != 20 + count * width * 4):
        raise ValueError("Qwen image grid, embedding width or size mismatch")
    rows = np.frombuffer(raw, dtype="<f4", offset=20).reshape(count, width)
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite Qwen image embeddings")
    return rows, nx, ny


def build_visual_prompt(tokenizer, rendered: str, sources, vision, destination: Path,
                        *, width: int, context: int):
    pieces = rendered.split("<|image_pad|>")
    if not 1 <= len(sources) <= 64 or len(pieces) != len(sources) + 1:
        raise ValueError("Qwen image markers and sources do not match")
    image_pad = tokenizer.ids["<|image_pad|>"]
    ids, positions, spans = [], [], []
    logical = 0

    def text(piece):
        nonlocal logical
        tokens = tokenizer.encode(piece, parse_special=True)
        ids.extend(tokens)
        positions.extend((logical + i,) * 3 for i in range(len(tokens)))
        logical += len(tokens)

    # Pin all request images in the encoder cache until this payload is written.
    encoded = vision.encode_all(sources)
    if len(encoded) != len(sources):
        raise ValueError("Qwen encoder returned an incomplete image list")
    for index, (path, count) in enumerate(encoded):
        text(pieces[index])
        rows, nx, ny = read_image(path, width)
        if len(rows) != count or len(ids) + count >= context:
            raise ValueError("Qwen visual prompt exceeds context or encoder count")
        spans.append((len(ids), rows))
        ids.extend([image_pad] * count)
        positions.extend((logical, logical + i // nx, logical + i % nx) for i in range(count))
        logical += max(nx, ny)
    text(pieces[-1])
    if len(ids) >= context or logical >= 1_000_000_000:
        raise ValueError("Qwen visual prompt exceeds native position bounds")
    with destination.open("wb") as out:
        out.write(struct.pack("<3i", 0x31505644, len(ids), len(spans)))
        out.write(np.asarray(ids, dtype="<i4").tobytes())
        out.write(np.asarray(positions, dtype="<i4").tobytes())
        for first, rows in spans:
            out.write(struct.pack("<2i", first, len(rows)))
            out.write(rows.tobytes())
    return ids
