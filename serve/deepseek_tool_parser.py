"""DeepSeek DSML tool frames at the existing frontend event boundary."""
import html
import json
import re

ROOT = '<｜DSML｜tool_calls>'
ROOT_END = '</｜DSML｜tool_calls>'
INVOKE = '<｜DSML｜invoke'
INVOKE_END = '</｜DSML｜invoke>'
PARAM = '<｜DSML｜parameter'
PARAM_END = '</｜DSML｜parameter>'
MAX_CALL = 65536
ATTR = re.compile(r'([a-z_]+)\s*=\s*([\'"])(.*?)\2', re.S)


def normalize(buffer):
    """The tokenizer renders the marker with a space after it, and a reply may spell the root
    `<｜DSML｜calls>` (no `tool_`) as the model's own habit: both are the same frame here."""
    for prefix in ('<｜DSML｜', '</｜DSML｜'):
        buffer = buffer.replace(prefix + ' ', prefix)
    buffer = buffer.replace('<｜DSML｜calls>', ROOT)
    buffer = buffer.replace('</｜DSML｜calls>', ROOT_END)
    return buffer


def attributes(tag):
    return {m[1]: html.unescape(m[3]) for m in ATTR.finditer(tag)}


def hold_suffix(text, markers):
    return max((n for marker in markers for n in range(1, min(len(text), len(marker)) + 1)
                if text.endswith(marker[:n])), default=0)


def invocation(text):
    """A complete invoke -> (end, name, arguments); an incomplete frame -> None."""
    end = text.find('>')
    if end < 0:
        return None
    name = attributes(text[:end + 1]).get('name')
    if not name:
        raise ValueError('DSML invoke has no tool name')
    pos, args = end + 1, {}
    while True:
        pos += len(text[pos:]) - len(text[pos:].lstrip())
        rest = text[pos:]
        if rest.startswith(INVOKE_END):
            return pos + len(INVOKE_END), name, args
        if not rest or any(marker.startswith(rest) for marker in (PARAM, INVOKE_END)):
            return None
        if not rest.startswith(PARAM):
            raise ValueError('Unexpected text between DSML parameters')
        end = text.find('>', pos)
        if end < 0:
            return None
        attrs = attributes(text[pos:end + 1])
        if not attrs.get('name'):
            raise ValueError('DSML parameter has no name')
        start, search = end + 1, end + 1
        while True:
            close = text.find(PARAM_END, search)
            if close < 0:
                return None
            after = text[close + len(PARAM_END):].lstrip()
            if after.startswith((PARAM, INVOKE_END)):
                break
            if not after or any(marker.startswith(after) for marker in (PARAM, INVOKE_END)):
                return None
            search = close + len(PARAM_END)  # a literal closer inside a string
        value = text[start:close]
        args[attrs['name']] = value if attrs.get('string') == 'true' else json.loads(value)
        pos = close + len(PARAM_END)


class DsmlOutputParser:
    """Use the common parser for text/reasoning; preserve native typed DSML arguments.

    Each invoke is bounded to64K characters. Arguments are delivered when the invoke is
    complete, so arbitrary string contents never pass through a lossy XML rewrite.
    Truncated/cancelled frames never become successful tool_call events.
    """
    def __init__(self, base):
        self.base, self.buffer, self.inside = base, '', False

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def state(self):
        return 'call' if self.inside else self.base.state

    @property
    def buf(self):
        return self.buffer or self.base.buf

    def feed(self, text):
        from serve.frontend import Event, ToolCall
        self.buffer = normalize(self.buffer + text)
        out = []
        while self.buffer:
            if not self.inside:
                at = self.buffer.find(ROOT)
                if at < 0:
                    # A marker that has begun but is not complete yet waits, whatever spelling
                    # follows it (`calls`, `tool_calls`, a space): never release it as text.
                    started = max(self.buffer.rfind('<｜DSML｜'), self.buffer.rfind('</｜DSML｜'))
                    if started >= 0 and '>' not in self.buffer[started:]:
                        safe = started
                    else:
                        safe = len(self.buffer) - hold_suffix(self.buffer, (ROOT,))
                    if safe:
                        out += self.base.feed(self.buffer[:safe])
                        self.buffer = self.buffer[safe:]
                    break
                if at:
                    out += self.base.feed(self.buffer[:at])
                    self.buffer = self.buffer[at:]
                if self.base.state != 'content':
                    # The frames may open while the parser still sits in the reasoning (a template that
                    # rendered thinking off, or a reply that skips the marker): close it so they are read
                    # as calls instead of being swallowed as reasoning text.
                    out += self.base.feed('</think>')
                if self.base.state != 'content' or not self.base._ok_at(len(self.base.buf)):
                    out += self.base.feed(self.buffer[:1])
                    self.buffer = self.buffer[1:]
                    continue
                self.buffer = self.buffer[len(ROOT):]
                self.inside = True
                continue
            self.buffer = self.buffer.lstrip()
            if self.buffer.startswith('</｜DSML｜') and '>' not in self.buffer:
                break
            if self.buffer.startswith(ROOT_END):
                self.buffer = self.buffer[len(ROOT_END):]
                self.inside = False
                continue
            if not self.buffer or any(m.startswith(self.buffer) for m in (INVOKE, ROOT_END)):
                break
            if not self.buffer.startswith(INVOKE):
                raise ValueError('Expected a DSML invoke inside tool_calls')
            if INVOKE_END not in self.buffer:
                if len(self.buffer) > MAX_CALL:
                    raise ValueError('DSML invoke exceeds64K-character limit')
                break
            result = invocation(self.buffer)
            if result is None:
                if len(self.buffer) > MAX_CALL:
                    raise ValueError('DSML invoke exceeds64K-character limit')
                break
            end, name, args = result
            if end > MAX_CALL:
                raise ValueError('DSML invoke exceeds64K-character limit')
            if name not in self.base.schemas:
                raise ValueError('DSML names a tool that was not offered')
            call = ToolCall(name=name, arguments=args)
            if self.base.stream_tools:
                out += [Event('tool_start', call=call),
                        Event('tool_args', json.dumps(args, ensure_ascii=False), call=call)]
            out.append(Event('tool_call', call=call))
            self.buffer = self.buffer[end:]
        return out

    def finish(self, reason=None):
        out = []
        if self.buffer:
            # Preserve partial output as text, never fabricate missing arguments.
            pending = (ROOT if self.inside else '') + self.buffer
            out += self.base.feed(pending)
            self.buffer = ''
        self.inside = False
        return out + self.base.finish(reason)
