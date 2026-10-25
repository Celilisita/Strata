"""Explicit Gemma weight formats; asset identities, never model computation."""
from pathlib import Path
import re

SOURCE_SHA = '71667f9e601a4b914a98425c59150b731f6e15d260d661dbd1f1ee07469fc7db'
SOURCE_BYTES = 18687062176
VARIANTS = {'nvfp4': 'gemma4_31b_nvfp4', 'original_quant': 'gemma4_31b_original_quant'}
SELECTED = ('blk.0.attn_q.weight', 'blk.5.attn_q.weight', 'blk.0.attn_k.weight',
    'blk.5.attn_k.weight', 'blk.0.attn_v.weight', 'blk.7.attn_v.weight',
    'blk.0.attn_output.weight', 'blk.5.attn_output.weight', 'blk.0.ffn_gate.weight',
    'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight', 'blk.7.ffn_down.weight', 'token_embd.weight')


def weight_backend(cfg):
    backend = cfg.get('backend')
    if backend is None and cfg.get('variant') == VARIANTS['nvfp4']:
        backend = 'nvfp4'  # Existing frozen NVFP4 configurations retain their explicit variant.
    if backend not in VARIANTS or cfg.get('variant') != VARIANTS[backend]:
        raise ValueError('Gemma requires a matching explicit weight backend and variant')
    if backend == 'original_quant':
        if cfg.get('conversion_receipt') or cfg.get('tiny_compensated', False):
            raise ValueError('Original GGML quant cannot use an NVFP4 receipt or tiny codec')
    return backend


def file_tuple(path):
    path = Path(path); s = path.stat()
    return dict(path=str(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns,
                device=s.st_dev, file_id=s.st_ino)


def validate_original_binding(cfg):
    if weight_backend(cfg) != 'original_quant':
        raise ValueError('Original asset validator requires original_quant')
    binding = cfg.get('asset_binding', {})
    actual = file_tuple(cfg['model'])
    if any(binding.get(key) != value for key, value in actual.items()):
        raise ValueError('Original Gemma file identity changed')
    if actual['bytes'] != SOURCE_BYTES or binding.get('sha256') != SOURCE_SHA:
        raise ValueError('Original Gemma must bind the exact publisher-matched source checkpoint')
    expected = dict(tensors=833, type_counts={'0':422, '12':355, '14':56},
                    body_tensors=410, body_type_counts={'12':355, '14':55},
                    tied_embedding_type=14, independent_output_weight=False)
    if binding.get('inventory') != expected:
        raise ValueError('Original Gemma requires raw Q4_K/Q6_K body and tied Q6_K embedding')
    selected = binding.get('selected_payloads', [])
    if len(selected) != len(SELECTED) or {x.get('name') for x in selected} != set(SELECTED):
        raise ValueError('Original Gemma selected payload inventory differs')
    for row in selected:
        if row.get('type') not in (12,14) or not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')):
            raise ValueError('Original payload requires raw Q4_K/Q6_K and actual SHA')
        shape = row.get('shape', [])
        if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError('Original payload matrix shape is invalid')
        elements = shape[0]*shape[1]
        expected_bytes = elements//256 * (144 if row['type']==12 else 210)
        if (elements%256 or row.get('bytes') != expected_bytes or type(row.get('offset')) is not int or
                row['offset'] < 0 or binding['header_bytes']+row['offset']+expected_bytes > actual['bytes']):
            raise ValueError('Original raw GGML payload extent is invalid')
    tied = next(x for x in selected if x['name']=='token_embd.weight')
    if tied['type'] != 14 or tied['shape'] != [5376,262144]:
        raise ValueError('Original tied embedding/head identity differs')
    return True
