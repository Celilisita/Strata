"""Pinned Gemma SM120 prompt assets with the shared native tokenizer decoder."""
from pathlib import Path
import hashlib
import json

from serve.frontend import ChatTemplate
from serve.gemma_frontend import Tokenizer as BoundTokenizer

EOS = frozenset({1, 50, 106})
TOKENIZER_SHA256 = '12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6'
TEMPLATE_SHA256 = '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_frontend(directory):
    directory = Path(directory)
    expected = {'tokenizer.json': TOKENIZER_SHA256, 'chat_template.jinja': TEMPLATE_SHA256}
    for name, digest in expected.items():
        if sha(directory/name) != digest:
            raise ValueError('Gemma original frontend identity changed: '+name)
    return expected


class GemmaTokenizer(BoundTokenizer):
    def __init__(self, directory, metadata):
        check_frontend(directory)
        super().__init__(Path(directory)/'tokenizer.json', metadata)
        if len(self.tokens) != 262144 or any(self.ids.get(k) != v for k, v in
                {'<bos>': 2, '<eos>': 1, '<|tool_response>': 50, '<turn|>': 106}.items()):
            raise ValueError('Gemma SM120 original vocabulary/control IDs mismatch')
        data = json.loads((Path(directory)/'tokenizer.json').read_text(encoding='utf-8'))
        self.control_tokens = [row['content'] for row in data['added_tokens'] if row['special']]


class GemmaTemplate(ChatTemplate):
    think_dialect = 'gemma_channel'       # reasoning arrives in `<|channel>thought ... <channel|>` blocks

    def __init__(self, directory):
        check_frontend(directory)
        super().__init__(Path(directory)/'chat_template.jinja')

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        kwargs.setdefault('bos_token', '<bos>')
        kwargs.setdefault('eos_token', '<eos>')
        return super().render(messages, tools=tools,
            add_generation_prompt=add_generation_prompt, **kwargs)


class GemmaFrontend:
    def __init__(self, cfg):
        from gguf_reader import GGUFFile
        self.path = Path(cfg['model'])
        self.md = GGUFFile(self.path).metadata
        if self.md.get('general.architecture') != 'gemma4':
            raise ValueError('Gemma SM120 frontend requires the actual Gemma4 GGUF')
        self.tokenizer = GemmaTokenizer(cfg['tokenizer'], self.md)
        self.stop_ids = sorted(EOS)
        self.identity = dict(architecture='gemma4', tokenizer_sha256=TOKENIZER_SHA256,
            template_sha256=TEMPLATE_SHA256, template_source='bound_chat_template.jinja',
            vocabulary_binding='exact_actual_GGUF', streaming_decode='native_DecodeStream')
