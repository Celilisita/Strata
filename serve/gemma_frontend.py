"""Completed Gemma tokenizer/template binding; reads GGUF metadata only."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from dense_prepare import DenseFrontend, EOG_TEXT
from gguf_reader import GGUFFile


class Tokenizer:
    """Use the bound tokenizer's own normalizer, model, added tokens and decoder."""
    pre = "gemma4"

    def __init__(self, path, metadata):
        try:
            from tokenizers import Tokenizer as Backend
            from tokenizers.decoders import DecodeStream
        except ImportError as error:
            raise ValueError("Gemma needs tokenizers with native decoders.DecodeStream "
                             "(verified API: tokenizers 0.22.2); install that dependency") from error
        if not callable(getattr(DecodeStream, "step", None)):
            raise ValueError("Gemma needs tokenizers.decoders.DecodeStream.step(tokenizer, id)")
        self.decode_stream_type = DecodeStream
        # Check the constructor during config binding, before loading a model.
        try:
            DecodeStream(skip_special_tokens=False)
        except TypeError as error:
            raise ValueError("Gemma needs native DecodeStream(skip_special_tokens=False); "
                             "verified dependency API: tokenizers 0.22.2") from error
        self.backend = Backend.from_file(str(path))
        self.lock = threading.RLock()
        self.tokens = list(metadata["tokenizer.ggml.tokens"])
        self.ids = {text: token for token, text in enumerate(self.tokens)}
        actual = self.backend.get_vocab(with_added_tokens=True)
        if len(actual) != len(self.tokens) or any(actual.get(t) != i for i, t in enumerate(self.tokens)):
            raise ValueError("Gemma tokenizer.json vocabulary differs from the actual GGUF")
        self.special_ids = {k: int(v) for k, v in metadata.items()
                            if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        if any(not 0 <= v < len(self.tokens) for v in self.special_ids.values()):
            raise ValueError("Gemma special token ID outside vocabulary")
        if not hasattr(self.backend, "encode_special_tokens"):
            raise ValueError("Gemma frontend needs tokenizers with encode_special_tokens support")

    def encode(self, text, parse_special=False, plain=()):
        # The common server can mark literal spans whose control spellings must
        # remain ordinary text. Decode stays independent of this encode lock.
        cuts = sorted((int(a), int(b)) for a, b in plain)
        if any(a < 0 or b < a or b > len(text) or (i and a < cuts[i-1][1])
               for i, (a, b) in enumerate(cuts)):
            raise ValueError("Gemma literal span outside prompt")
        with self.lock:
            result, first = [], 0
            for a, b in cuts + [(len(text), len(text))]:
                self.backend.encode_special_tokens = not parse_special
                result.extend(self.backend.encode(text[first:a], add_special_tokens=False).ids)
                self.backend.encode_special_tokens = True
                result.extend(self.backend.encode(text[a:b], add_special_tokens=False).ids)
                first = b
            return result

    def decode(self, ids, errors="replace"):
        return self.backend.decode(list(ids), skip_special_tokens=False)

    def incremental_decoder(self):
        return GemmaIncrementalDecoder(self.backend, self.decode_stream_type)


class GemmaIncrementalDecoder:
    """Request-owned native decoder state; no full output history is re-decoded."""
    def __init__(self, backend, stream_type):
        self.backend = backend
        self.stream = stream_type(skip_special_tokens=False)
        self.waiting = False

    def push(self, token):
        chunk = self.stream.step(self.backend, int(token))
        # None also includes no-growth tokens. Conservatively defer optional
        # reasoning/force boundaries; hard output budgets and EOG stay separate.
        self.waiting = chunk is None
        return "" if chunk is None else chunk

    def pending(self):
        return self.waiting


class GemmaFrontend(DenseFrontend):
    def __init__(self, cfg):
        from dense_prepare import _json_filter, _raise
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        import datetime
        self.path = Path(cfg["model"])
        self.md = md = GGUFFile(self.path).metadata
        if md.get("general.architecture") != "gemma4":
            raise ValueError("completed Gemma frontend requires the Gemma4 GGUF")
        self.tokenizer = Tokenizer(cfg["tokenizer_json"], md)
        self.special_ids = self.tokenizer.special_ids
        self.template_key = "tokenizer.chat_template"
        source = md.get(self.template_key)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("actual Gemma GGUF has no chat template")
        self.template_hash = hashlib.sha256(source.encode()).hexdigest()
        self.template_chars = len(source)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _json_filter
        env.globals.update(raise_exception=_raise,
                           strftime_now=lambda fmt: datetime.datetime.now().strftime(fmt))
        self.template = env.from_string(source)
        self.template_tokens = {k[len("tokenizer.ggml."):-3]: self.tokenizer.tokens[v]
                                for k, v in self.special_ids.items()}
        stops = {v for k, v in self.special_ids.items()
                 if k.rsplit(".", 1)[-1] in ("eos_token_id", "eot_token_id", "eom_token_id")}
        stops.update(i for t, i in self.tokenizer.ids.items() if t in EOG_TEXT)
        self.stop_ids = sorted(stops)
        if not self.stop_ids or len(self.stop_ids) > 16:
            raise ValueError("Gemma needs one to sixteen actual EOG IDs")
        self.identity = dict(architecture="gemma4", template_sha256=self.template_hash,
            add_bos_token=bool(md.get("tokenizer.ggml.add_bos_token", False)),
            add_eos_token=bool(md.get("tokenizer.ggml.add_eos_token", False)))
