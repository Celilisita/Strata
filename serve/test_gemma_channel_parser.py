"""serve/test_gemma_channel_parser.py - Gemma4 thought channels at the shared event boundary.

Gemma4 writes its reasoning into `<|channel>thought\\n...\\n<channel|>` blocks; the decoder passes those control
tokens through as their spelled text, so the wrapper matches the spelled pair and keeps the shared parser (its
own thinking off) for everything outside a channel.  Everything must come out whole and stream-invariant: the
same events in the same order no matter how the deltas are cut.

    python -m unittest serve.test_gemma_channel_parser -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.deepseek_tools import output_parser  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402
from serve.gemma_channel_parser import GemmaChannelOutputParser  # noqa: E402

OPEN, CLOSE = "<|channel>thought\n", "<channel|>"
STREAM = ("hello\n\n" + OPEN + "the model ponders\n" + CLOSE + "\n\nthe answer.\n")
NO_CHANNEL = "just an answer\n"


class StubTemplate:
    think_dialect = "gemma_channel"


def parse(text, width):
    p = GemmaChannelOutputParser(OutputParser(thinking=False, tools=None, stream_tools=True))
    evs, rnd, i = [], random.Random(7), 0
    while i < len(text):
        k = width or rnd.randint(1, 9)
        evs += p.feed(text[i:i + k])
        i += k
    return evs + p.finish("stop")


def by_kind(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


WIDTHS = (1, 2, 3, 7, 13, 0, 100_000)


class Channels(unittest.TestCase):
    def test_the_channel_is_reasoning_and_the_rest_is_content(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                evs = parse(STREAM, width)
                self.assertEqual(by_kind(evs, "reasoning"), "the model ponders\n")
                self.assertEqual(by_kind(evs, "content"), "hello\n\n\n\nthe answer.\n")

    def test_no_channel_stays_content(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                self.assertEqual(by_kind(parse(NO_CHANNEL, width), "content"), NO_CHANNEL)

    def test_an_unclosed_channel_keeps_its_text_as_reasoning_at_finish(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                evs = parse("answer first\n" + OPEN + "pondered but cut", width)
                self.assertEqual(by_kind(evs, "reasoning"), "pondered but cut")
                self.assertEqual(by_kind(evs, "content"), "answer first\n")

    def test_the_state_reports_the_channel_for_the_budget(self):
        # run() counts reasoning tokens while parser.state is "reasoning" and checks a clean point on parser.buf
        p = GemmaChannelOutputParser(OutputParser(thinking=False))
        p.feed("x " + OPEN + "pondering")
        self.assertEqual(p.state, "reasoning")
        p.feed(CLOSE)
        self.assertEqual(p.state, "content")


class Registration(unittest.TestCase):
    def test_the_factory_wraps_a_gemma_template(self):
        p = output_parser(StubTemplate(), OutputParser, thinking=True, tools=None, stream_tools=True)
        self.assertIsInstance(p, GemmaChannelOutputParser)

    def test_an_ordinary_template_keeps_the_plain_parser(self):
        self.assertIsInstance(output_parser(object(), OutputParser, thinking=True), OutputParser)


if __name__ == "__main__":
    unittest.main(verbosity=2)
