"""Convert MiniCPM 4.7 canvas embeddings into Strata visual prompt payloads."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


SVE2 = 0x32455653
DVP1 = 0x31505644


def read_canvas(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("truncated MiniCPM canvas")
    magic, count, n_pos, width, axes = struct.unpack_from("<5i", data)
    if (magic != SVE2 or not 1 <= count <= 8192 or not 1 <= n_pos <= count or
        width != 2048 or axes != 3 or len(data) != 20 + count * (width * 4 + 12)):
        raise ValueError("invalid MiniCPM canvas header or size")
    embeddings = np.frombuffer(data, dtype="<f4", count=count * width,
                               offset=20).reshape(count, width)
    positions = np.frombuffer(data, dtype="<i4", count=count * 3,
                              offset=20 + count * width * 4).reshape(count, 3)
    if not np.isfinite(embeddings).all() or (positions < 0).any():
        raise ValueError("invalid MiniCPM canvas values")
    return embeddings, positions, n_pos


def build_visual_prompt(tokenizer, rendered: str, image_sources: list[str],
                        vision, destination: Path) -> list[int]:
    pieces = rendered.split("<|image_pad|>")
    if len(pieces) != len(image_sources) + 1 or not image_sources:
        raise ValueError("MiniCPM image markers and payloads do not match")
    image_pad = tokenizer.ids["<|image_pad|>"]
    ids: list[int] = []
    positions: list[tuple[int, int, int]] = []
    spans: list[tuple[int, np.ndarray]] = []
    logical = 0

    def add_text(text: str) -> None:
        nonlocal logical
        encoded = tokenizer.encode(text, parse_special=True)
        ids.extend(encoded)
        positions.extend((logical + i,) * 3 for i in range(len(encoded)))
        logical += len(encoded)

    for index, source in enumerate(image_sources):
        add_text(pieces[index])
        add_text(f"<image_id>{index}</image_id><image>")
        path, count = vision.encode(source)
        embeddings, relative, n_pos = read_canvas(path)
        if len(embeddings) != count or logical < 1:
            raise ValueError("MiniCPM canvas position or length mismatch")
        first = len(ids)
        ids.extend([image_pad] * count)
        absolute = relative.astype(np.int64) + logical - 1
        if (absolute < 0).any() or (absolute > 1_000_000_000).any():
            raise ValueError("MiniCPM canvas position outside range")
        positions.extend(map(tuple, absolute.tolist()))
        spans.append((first, embeddings))
        logical += n_pos
    add_text(pieces[-1])
    if len(ids) != len(positions) or len(ids) > 16384:
        raise ValueError("MiniCPM visual prompt exceeds supported length")
    with destination.open("wb") as file:
        file.write(struct.pack("<3i", DVP1, len(ids), len(spans)))
        file.write(np.asarray(ids, dtype="<i4").tobytes())
        file.write(np.asarray(positions, dtype="<i4").tobytes())
        for first, embeddings in spans:
            file.write(struct.pack("<2i", first, len(embeddings)))
            file.write(embeddings.tobytes())
    return ids
