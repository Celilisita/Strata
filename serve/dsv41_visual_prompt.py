"""DeepSeek-V4.1's built-in vision tower at the native boundary.

The engine owns the tower (pack's vision.* tensors, ``--vision on``): this builds what it reads - the
pictures as DSI1 RGB rows, both sides multiples of the patch size - and the prompt with one image
token per token the tower will report for that grid.
"""
from __future__ import annotations

import io
import struct
import uuid
from pathlib import Path

PATCH = 14
MERGE = 3
MAX_TOKENS = 1024


def _decode(source):
    from PIL import Image, ImageOps

    raw = source
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        text = str(source)
        if text.startswith("data:"):
            import base64

            _, _, payload = text.partition(",")
            raw = base64.b64decode(payload)
        elif text.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(text, timeout=120) as response:
                raw = response.read()
        else:
            raw = Path(text).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def _grid(n_h, n_w):
    # The tower's rows for a grid: n_h * n_w patch tokens plus a separator per row and the two edges
    # (ds4_vision.cu's cap is n_h * (n_w + 1) + 2, which is also what it emits).
    return n_h * (n_w + 1) + 2


def choose_grid(width, height):
    """The patch grid: the largest image-token count within the cap whose shape keeps the picture's aspect."""
    ratio = width / height if height else 1.0
    best = None
    for n_h in range(1, MAX_TOKENS):
        n_w = (MAX_TOKENS - 2) // n_h - 1
        if n_w < 1:
            break
        tokens = _grid(n_h, n_w)
        if tokens < 4:
            continue
        score = abs(__import__("math").log((n_w / n_h) / ratio))
        if best is None or score < best[0] - 1e-9 or (abs(score - best[0]) < 1e-9 and tokens > best[1]):
            best = (score, tokens, n_h, n_w)
    if best is None:
        raise ValueError("DeepSeek-V4.1 vision: no image grid fits the token cap")
    return best[2], best[3]


def _pixels(source):
    from PIL import Image

    image = _decode(source)
    n_h, n_w = choose_grid(image.width, image.height)
    width, height = n_w * MERGE * PATCH, n_h * MERGE * PATCH
    scaled = image.resize((width, height), Image.LANCZOS)
    return scaled.tobytes(), width, height, _grid(n_h, n_w)


def build_visual_prompt(service, messages, tools, kwargs, sources, vision, destination: Path, *, context: int):
    image_token = getattr(vision, "image_token", None)
    if not image_token:
        raise ValueError("DeepSeek-V4.1 vision: the config needs vision.image_token")
    marker = f"__STRATA_DSV41_IMAGE_{uuid.uuid4().hex}__"
    rendered_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered_messages.append(message)
            continue
        pieces = []
        for part in content:
            if part.get("type") == "image":
                pieces.append(marker)
            elif part.get("type") == "text":
                pieces.append(part.get("text", ""))
            else:
                raise ValueError("unsupported DeepSeek-V4.1 image message part")
        rendered_messages.append(dict(message, content="".join(pieces)))
    rendered = service.render_prompt(rendered_messages, tools, kwargs)
    pieces = rendered.split(marker)
    if len(pieces) - 1 != len(sources):
        raise ValueError("the DeepSeek-V4.1 prompt and its images do not match")
    ids, pictures = [], []
    for index, piece in enumerate(pieces):
        ids.extend(service.tok.encode(piece, parse_special=True))
        if index + 1 >= len(pieces):
            continue
        rgb, width, height, tokens = _pixels(sources[index])
        if len(ids) + tokens >= context:
            raise ValueError("DeepSeek-V4.1 visual prompt exceeds the context")
        pictures.append((len(ids), width, height, rgb))
        ids.extend([int(image_token)] * tokens)
    if len(ids) >= context:
        raise ValueError("DeepSeek-V4.1 visual prompt exceeds the context")
    with Path(destination).open("wb") as out:
        out.write(b"DSI1")
        out.write(struct.pack("<i", len(pictures)))
        for start, width, height, rgb in pictures:
            out.write(struct.pack("<qii", start, width, height))
            out.write(rgb)
    return ids
