"""Opt-in dense tokenizer support, without replacing the shared tokenizer.

MiniCPM5 adds two ordered split passes and ignore_merges. Qwen35, special
tokens, plain spans (when supported by the base), GGUF loading and decoding
retain the installed shared Tokenizer's implementation.
"""
from __future__ import annotations

import regex

from ranked_bpe import piece_encoder
from strata_tokenizer import BYTE_TO_UNICODE, QWEN35_PATTERN, Tokenizer as BaseTokenizer


# LLAMA_VOCAB_PRE_TYPE_MINICPM5: digit groups are isolated before applying
# the second pattern. Combining marks deliberately remain outside its letters.
MINICPM5_PATTERNS = (
    r"\p{N}{1,3}",
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
)
PRE_PATTERNS = {"qwen35": (QWEN35_PATTERN,), "minicpm5": MINICPM5_PATTERNS}


class Tokenizer(BaseTokenizer):
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        if pre not in PRE_PATTERNS:
            raise ValueError(f"unsupported dense pre-tokenizer {pre!r}")
        # Both the historical base and the current main base support qwen35;
        # neither needs shared MiniCPM-specific branches for this adapter.
        super().__init__(tokens,merges,token_types,"qwen35",special_ids)
        if pre == "minicpm5":
            self.pre = pre
            self._patterns = tuple(regex.compile(pattern) for pattern in MINICPM5_PATTERNS)
            self._re = self._patterns[-1]
            self._encode_piece = piece_encoder(self.ranks,self.ids,BYTE_TO_UNICODE)

    def _encode_plain(self, text: str) -> list[int]:
        if self.pre != "minicpm5":
            return super()._encode_plain(text)
        pieces = [text]
        for pattern in self._patterns:
            split = []
            for piece in pieces:
                pos = 0
                for match in pattern.finditer(piece):
                    if match.start()>pos:
                        split.append(piece[pos:match.start()])
                    if match.end()>match.start():
                        split.append(match.group())
                    pos = match.end()
                if pos<len(piece):
                    split.append(piece[pos:])
            pieces = split
        out: list[int] = []
        for piece in pieces:
            mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
            token = self.ids.get(mapped)
            out.extend((token,) if token is not None else self._encode_piece(piece))
        return out
