"""Explicit fixed-depth sampled MTP binding for the private Gemma service."""
from pathlib import Path
import hashlib
import re


def validate_mtp(cfg):
    depth = cfg.get('mtp')
    if type(depth) is not int or depth not in (0, 1, 2):
        raise ValueError('Gemma MTP depth must be the explicit integer0,1 or2')
    draft, binding = cfg.get('draft'), cfg.get('draft_asset_binding')
    if depth == 0:
        if draft or binding:
            raise ValueError('Gemma MTP0 must not load a draft asset')
        return depth
    if cfg.get('backend') != 'nvfp4' or not cfg.get('tiny_compensated', False):
        raise ValueError('Gemma sampled MTP requires the explicit native NVFP4/tiny backend')
    if cfg.get('max_batch', 2048) not in (2048, 8192):
        raise ValueError('Gemma sampled MTP requires the target scratch maximum_batch2048 or8192')
    if not isinstance(draft, str) or not isinstance(binding, dict):
        raise ValueError('Gemma sampled MTP requires draft and draft_asset_binding')
    path = Path(draft)
    if not path.is_file():
        raise ValueError('Gemma bound assistant GGUF is missing')
    stat = path.stat()
    if (binding.get('path') != str(path) or binding.get('bytes') != stat.st_size or
            binding.get('mtime_ns') != stat.st_mtime_ns):
        raise ValueError('Gemma assistant file tuple changed')
    if not re.fullmatch('[0-9a-f]{64}', binding.get('sha256', '')):
        raise ValueError('Gemma assistant requires its prior full SHA256 binding')
    count = binding.get('header_bytes')
    if type(count) is not int or not 1 <= count <= min(stat.st_size, 32 * 1024 * 1024):
        raise ValueError('Gemma assistant aligned header span is invalid')
    with path.open('rb') as stream:
        digest = hashlib.sha256(stream.read(count)).hexdigest()
    if digest != binding.get('header_sha256'):
        raise ValueError('Gemma assistant header identity changed')
    expected = {'general.architecture': 'gemma4-assistant',
        'gemma4-assistant.block_count': 4, 'gemma4-assistant.embedding_length': 1024,
        'gemma4-assistant.embedding_length_out': 5376,
        'gemma4-assistant.context_length': 131072}
    if any(binding.get('metadata', {}).get(k) != v for k, v in expected.items()):
        raise ValueError('Gemma assistant metadata differs from its Dense49 Q4_0 binder')
    if binding.get('tensor_count') != 49 or binding.get('quantization') != 'Q4_0':
        raise ValueError('Gemma sampled MTP requires its original Dense49 Q4_0 assistant')
    # Native bind_spec validates every tensor's role, shape and raw format. This
    # check reads the bound header only; it never hashes the entire draft again.
    return depth


def request_depth(cfg, request):
    """A loaded private engine has a fixed depth; switches require a new load."""
    depth = cfg['mtp']
    for name in ('mtp', 'strata_native_mtp'):
        value = (request or {}).get(name)
        if value is not None and (type(value) is not int or value != depth):
            raise ValueError('Gemma request MTP must match the loaded fixed-depth configuration')
    return depth


def validate_request_extent(cfg, prompt_tokens, maximum_new):
    if prompt_tokens + maximum_new > cfg['context']:
        raise ValueError('Gemma request exceeds the configured target logical context')
    # The native controller stops drafting at its actual131072 boundary and
    # uses sampled target T1 for the remaining target context; no draft wrap.
