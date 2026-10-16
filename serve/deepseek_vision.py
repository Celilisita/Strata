"""DeepSeek4V's image blocks at the Strata service protocol boundary.

The projector emits a complete block with learned PAD/START/END embeddings;
the base text vocabulary has no Qwen image-pad token. A -1 service placeholder
is expanded only after the encoder reports the block's token count.
"""
from __future__ import annotations

import os
import struct
import uuid
from pathlib import Path

from serve.frontend import images_of, unmark_think_literals


def encode_prompt(service, messages, tools, kwargs, marked, marked_tools, changed) -> list[int]:
    marker = f"__STRATA_DSV4_IMAGE_{uuid.uuid4().hex}__"
    rendered = []
    for message in marked:
        content = message.get("content")
        if not isinstance(content, list):
            rendered.append(message)
            continue
        pieces = []
        for part in content:
            if part.get("type") == "image":
                pieces.append(marker)
            elif part.get("type") == "text":
                pieces.append(part.get("text", ""))
            else:
                raise ValueError("unsupported DeepSeek image message part")
        rendered.append(dict(message, content="".join(pieces)))
    prompt = service.render_prompt(rendered, marked_tools, kwargs)
    prompt, plain = unmark_think_literals(prompt, service.literals) if changed else (prompt, [])
    pieces = prompt.split(marker)
    if len(pieces) - 1 != len(images_of(messages)):
        raise ValueError("the DeepSeek prompt and its images do not match")
    ids, offset = [], 0
    for index, piece in enumerate(pieces):
        spans = [(max(a, offset) - offset, min(b, offset + len(piece)) - offset)
                 for a, b in plain if a < offset + len(piece) and b > offset]
        ids.extend(service.tok.encode(piece, parse_special=True, plain=spans))
        if index + 1 < len(pieces):
            ids.append(-1)
        offset += len(piece) + len(marker)
    return ids


def expand_placeholders(ids: list[int], encoded: list[tuple[Path, int]]) -> tuple[list[int], list[tuple[Path, int]]]:
    out, records = [], []
    for token in ids:
        if token != -1:
            out.append(token)
            continue
        if len(records) == len(encoded):
            raise ValueError("the DeepSeek prompt and its images do not match")
        source, count = encoded[len(records)]
        skip = len(out) % 4
        if count <= skip:
            raise ValueError("the DeepSeek image block is too short")
        out.extend([-1] * (count - skip))
        records.append((source, skip))
    if len(records) != len(encoded):
        raise ValueError("the DeepSeek prompt and its images do not match")
    return out, records


def write_embeddings(path: Path, records: list[tuple[Path, int]]) -> None:
    """mtmd encodes with three leading PAD rows at offset zero. At prompt
    offset p, DeepSeek4V needs 3-(p mod 4), so strip p mod 4 rows."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0) | getattr(os, "O_SHORT_LIVED", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as output:
        for source, skip in records:
            raw = Path(source).read_bytes()
            if len(raw) < 20:
                raise ValueError("short DeepSeek image embedding record")
            magic, n, nx, ny, width = struct.unpack_from("<5i", raw)
            if (magic != 0x31455653 or n < 1 or nx != n or ny != 1 or width < 1 or
                    skip not in range(4) or n <= skip or len(raw) != 20 + n * width * 4):
                raise ValueError("invalid DeepSeek image embedding record")
            output.write(struct.pack("<5i", magic, n - skip, n - skip, 1, width))
            output.write(raw[20 + skip * width * 4:])
