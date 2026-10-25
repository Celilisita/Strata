"""Gemma thought channels at the existing frontend event boundary.

The shared parser reads `<think>...</think>` as text; Gemma4 writes its reasoning into
`<|channel>thought\\n...\\n<channel|>` blocks, and the decoder passes those control tokens
through as their spelled text (skip_special_tokens=False).  This wrapper keeps the shared
parser for everything outside a thought channel (with its own thinking turned off) and
delivers the inside of a channel as reasoning events.
"""
from serve.frontend import Event

OPEN = '<|channel>thought\n'
CLOSE = '<channel|>'


def hold_suffix(text, markers):
    return max((n for marker in markers for n in range(1, min(len(text), len(marker)) + 1)
                if text.endswith(marker[:n])), default=0)


class GemmaChannelOutputParser:
    """Reasoning events for the inside of a thought channel; the base parser for the rest."""

    def __init__(self, base):
        self.base, self.buffer, self.inside = base, '', False

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def state(self):
        return 'reasoning' if self.inside else self.base.state

    @property
    def buf(self):
        return self.buffer or self.base.buf

    def feed(self, text):
        self.buffer += text
        out = []
        while self.buffer:
            if not self.inside:
                at = self.buffer.find(OPEN)
                if at < 0:
                    safe = len(self.buffer) - hold_suffix(self.buffer, (OPEN,))
                    if safe:
                        out += self.base.feed(self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    break
                if at:
                    out += self.base.feed(self.buffer[:at])
                    self.buffer = self.buffer[at:]
                self.buffer = self.buffer[len(OPEN):]
                self.inside = True
                continue
            at = self.buffer.find(CLOSE)
            if at < 0:
                safe = len(self.buffer) - hold_suffix(self.buffer, (CLOSE,))
                if safe:
                    out.append(Event('reasoning', self.buffer[:safe]))
                    self.buffer = self.buffer[safe:]
                break
            if at:
                out.append(Event('reasoning', self.buffer[:at]))
            self.buffer = self.buffer[at + len(CLOSE):]
            self.inside = False
            continue
        return out

    def finish(self, reason=None):
        out = []
        if self.buffer:
            # a channel the reply ended inside: what it held stays reasoning text
            out += [Event('reasoning', self.buffer)] if self.inside else self.base.feed(self.buffer)
            self.buffer = ''
        self.inside = False
        return out + self.base.finish(reason)
