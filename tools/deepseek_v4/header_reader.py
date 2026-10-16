"""Bounded, metadata-only readers. Tensor payloads are never read or mapped."""
from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

MAX_HEADER = 64 * 1024 * 1024
MAX_ENTRIES = 200_000
MAX_STRING = 16 * 1024 * 1024
META_FORMAT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON descriptor key: {key}")
        result[key] = value
    return result


class HeaderReader:
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size):
        if size < 0 or self.bytes + size > MAX_HEADER:
            raise ValueError("header read budget exceeded")
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("truncated header")
        self.digest.update(data)
        self.bytes += size
        return data

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))

    def string(self):
        size, = self.unpack("Q")
        if size > MAX_STRING:
            raise ValueError("header string exceeds limit")
        return self.read(size).decode("utf-8", errors="strict")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item_kind, count = self.unpack("IQ")
            if count > MAX_ENTRIES or item_kind == 9:
                raise ValueError("unsupported or oversized metadata array")
            return [self.value(item_kind) for _ in range(count)]
        if kind not in META_FORMAT:
            raise ValueError(f"unknown GGUF metadata type {kind}")
        return self.unpack(META_FORMAT[kind])[0]


def safetensors(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        reader = HeaderReader(handle)
        header_len, = reader.unpack("Q")
        if header_len > MAX_HEADER - 8 or header_len > size - 8:
            raise ValueError("invalid safetensors header length")
        header = json.loads(reader.read(header_len), object_pairs_hook=unique_object)
    if len(header) > MAX_ENTRIES:
        raise ValueError("tensor count exceeds limit")
    tensors, spans = [], []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = entry["shape"]
        begin, end = entry["data_offsets"]
        if (not isinstance(shape, list) or len(shape) > 8 or
                any(not isinstance(d, int) or d < 0 for d in shape) or
                not 0 <= begin <= end <= size - 8 - header_len):
            raise ValueError(f"invalid tensor descriptor: {name}")
        spans.append((begin, end, name))
        tensors.append({"name": name, "shape": shape, "dtype": entry["dtype"],
                        "offset": begin + 8 + header_len, "stored_bytes": end - begin,
                        "shape_order": "row_major"})
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if previous[1] > current[0]:
            raise ValueError(f"overlapping tensors: {previous[2]}, {current[2]}")
    return {"format": "safetensors", "metadata": header.get("__metadata__", {}),
            "header_bytes_read": reader.bytes, "header_sha256": reader.digest.hexdigest(),
            "tensors": tensors}


def gguf(path: Path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        reader = HeaderReader(handle)
        if reader.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, tensor_count, kv_count = reader.unpack("IQQ")
        if version not in (2, 3) or max(tensor_count, kv_count) > MAX_ENTRIES:
            raise ValueError("unsupported GGUF version or oversized directory")
        metadata = {}
        for _ in range(kv_count):
            key = reader.string()
            if key in metadata:
                raise ValueError(f"duplicate metadata key: {key}")
            kind, = reader.unpack("I")
            metadata[key] = reader.value(kind)
        tensors, names = [], set()
        for _ in range(tensor_count):
            name = reader.string()
            dims, = reader.unpack("I")
            if dims > 8 or name in names:
                raise ValueError(f"invalid tensor directory: {name}")
            shape = list(reader.unpack("Q" * dims))
            kind, offset = reader.unpack("IQ")
            names.add(name)
            tensors.append({"name": name, "shape": shape, "ggml_type_id": kind,
                            "dtype": f"ggml_type_{kind}", "offset": offset,
                            "elements": math.prod(shape), "shape_order": "ggml_ne"})
        alignment = metadata.get("general.alignment", 32)
        if not isinstance(alignment, int) or alignment <= 0 or alignment > 65536:
            raise ValueError("invalid GGUF alignment")
        data_start = (reader.bytes + alignment - 1) // alignment * alignment
        ordered = sorted(tensors, key=lambda t: t["offset"])
        for i, tensor in enumerate(ordered):
            begin = data_start + tensor["offset"]
            end = data_start + ordered[i + 1]["offset"] if i + 1 < len(ordered) else size
            if not data_start <= begin <= end <= size:
                raise ValueError(f"invalid tensor offset: {tensor['name']}")
            tensor["offset"] = begin
            tensor["extent_with_padding"] = end - begin
    return {"format": "gguf", "version": version, "metadata": metadata,
            "header_bytes_read": reader.bytes, "header_sha256": reader.digest.hexdigest(),
            "tensors": tensors}
