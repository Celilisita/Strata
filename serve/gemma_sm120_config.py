"""Explicit isolated Gemma NVFP4 or original-quant profile; no format fallback."""
from pathlib import Path, PureWindowsPath
import hashlib
import math
import json
from serve.gemma_sm120_frontend import EOS, check_frontend, sha

from serve.gemma_original_contract import SOURCE_SHA, VARIANTS, weight_backend, validate_original_binding
from serve.gemma_sm120_mtp import validate_mtp


def validate_config(cfg):
    backend = weight_backend(cfg)
    timeout = cfg.get('response_timeout_s', 900)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Gemma response_timeout_s must be a finite positive number')
    expected = dict(family='gemma4_dense', variant=VARIANTS[backend],
                    slots=1, kv_policy='full')
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in expected.items()):
        raise ValueError('Gemma requires its explicit weight backend/family/variant, one slot and full Q8 KV')
    if type(cfg.get('context')) is not int or not 4 <= cfg['context'] <= 262144:
        raise ValueError('Gemma logical context must be4..262144; actual requests require GPU budget admission')
    batch = cfg.get('max_batch', 2048)
    if type(batch) is not int or not 1 <= batch <= 8192:
        raise ValueError('Gemma max_batch must be explicit1..8192; default2048')
    if type(cfg.get('reserve_mib', 4096)) is not int or cfg.get('reserve_mib', 4096) < 0:
        raise ValueError('Gemma reserve_mib must be a nonnegative integer')
    if cfg.get('vision') or cfg.get('dspark'):
        raise ValueError('Gemma vision/DSpark providers are not registered')
    if not isinstance(cfg.get('gpu_uuid'), str) or not cfg['gpu_uuid'].startswith('GPU-'):
        raise ValueError('Gemma requires an explicit GPU UUID')
    stops = cfg.get('stop_token_ids', [])
    if len(stops) != 3 or any(type(token) is not int for token in stops) or set(stops) != EOS:
        raise ValueError('Gemma requires original EOG IDs1,50,106')
    if type(cfg.get('cuda_graph', True)) is not bool:
        raise ValueError('Gemma cuda_graph must be a boolean')
    if type(cfg.get('original_input_reuse', False)) is not bool:
        raise ValueError('Gemma original_input_reuse must be an explicit boolean')
    if cfg.get('original_input_reuse', False) and backend != 'original_quant':
        raise ValueError('Gemma original input reuse requires original_quant')
    if type(cfg.get('pp_tc', False)) is not bool:
        raise ValueError('Gemma pp_tc must be an explicit boolean')
    for name in ('pp_global_keys1024', 'pp_tc_local_gqa2'):
        if type(cfg.get(name, False)) is not bool:
            raise ValueError('Gemma ' + name + ' must be an explicit boolean')
        if cfg.get(name, False) and (backend not in ('original_quant', 'nvfp4') or
                not cfg.get('pp_tc', False) or batch != 2048 or
                any(cfg.get(other, False) for other in ('pp_global_flash', 'pp_global_flash_f32pv',
                    'pp_whole_history', 'pp_cached_global', 'pp_segmented_flash'))):
            raise ValueError('Gemma ' + name + ' requires a native backend, PP_TC, batch2048 and other PP leaves off')
    if type(cfg.get('pp_rope_table', False)) is not bool:
        raise ValueError('Gemma pp_rope_table must be an explicit boolean')
    if cfg.get('pp_rope_table', False) and backend not in ('original_quant', 'nvfp4'):
        raise ValueError('Gemma PP RoPE table requires its explicit native weight backend')
    if type(cfg.get('t1_rope_table', False)) is not bool:
        raise ValueError('Gemma t1_rope_table must be an explicit boolean')
    if cfg.get('t1_rope_table', False) and (backend not in ('original_quant', 'nvfp4') or
            not cfg.get('pp_rope_table', False) or batch <= 3):
        raise ValueError('Gemma T1 RoPE table requires its native backend and an owned PP table')
    if type(cfg.get('t1_norm_cache_fusion', False)) is not bool:
        raise ValueError('Gemma t1_norm_cache_fusion must be an explicit boolean')
    if cfg.get('t1_norm_cache_fusion', False) and (backend not in ('original_quant', 'nvfp4') or
            not cfg.get('t1_rope_table', False)):
        raise ValueError('Gemma T1 norm/cache fusion requires explicit T1 owned-table mode')
    if type(cfg.get('tg_local270', False)) is not bool:
        raise ValueError('Gemma tg_local270 must be an explicit boolean')
    if cfg.get('tg_local270', False) and backend not in ('original_quant', 'nvfp4'):
        raise ValueError('Gemma TG local270 requires its explicit native weight backend')
    if type(cfg.get('pp_global_flash', False)) is not bool:
        raise ValueError('Gemma pp_global_flash must be an explicit boolean')
    if cfg.get('pp_global_flash', False) and backend != 'original_quant':
        raise ValueError('Gemma PP global flash requires original_quant')
    if type(cfg.get('pp_global_flash_f32pv', False)) is not bool:
        raise ValueError('Gemma pp_global_flash_f32pv must be an explicit boolean')
    if cfg.get('pp_global_flash_f32pv', False) and (backend != 'original_quant' or
            not cfg.get('pp_global_flash', False)):
        raise ValueError('Gemma F32-PV flash requires original_quant and explicit pp_global_flash')
    if type(cfg.get('original_q6_head', False)) is not bool:
        raise ValueError('Gemma original_q6_head must be an explicit boolean')
    if cfg.get('original_q6_head', False) and backend not in ('original_quant', 'nvfp4'):
        raise ValueError('Gemma tied Q6 head requires its explicit native weight backend')
    if type(cfg.get('pp_whole_history', False)) is not bool:
        raise ValueError('Gemma pp_whole_history must be an explicit boolean')
    if cfg.get('pp_whole_history', False) and (backend != 'original_quant' or
            not cfg.get('pp_tc', False) or cfg.get('pp_global_flash', False)):
        raise ValueError('Gemma whole-history PP requires original_quant+pp_tc and excludes pp_global_flash')
    if type(cfg.get('tg_global_gqa', False)) is not bool:
        raise ValueError('Gemma tg_global_gqa must be an explicit boolean')
    if cfg.get('tg_global_gqa', False) and (backend != 'original_quant' or not cfg.get('tg_global_merge', False)):
        raise ValueError('Gemma TG global GQA requires original_quant and qualified global merge')
    if type(cfg.get('tg_global_merge', False)) is not bool:
        raise ValueError('Gemma tg_global_merge must be an explicit boolean')
    if cfg.get('tg_global_merge', False) and backend not in ('original_quant', 'nvfp4'):
        raise ValueError('Gemma TG global merge requires its explicit native weight backend')
    if type(cfg.get('tiny_compensated', False)) is not bool:
        raise ValueError('Gemma tiny_compensated must be an explicit boolean')
    required = ['exe', 'model', 'tokenizer', 'temp_dir']
    if backend == 'nvfp4': required.append('conversion_receipt')
    for key in required:
        if not cfg.get(key) or not Path(cfg[key]).exists():
            raise ValueError('Missing Gemma asset/path: ' + key)
    if not Path(cfg['tokenizer']).is_dir() or not Path(cfg['temp_dir']).is_dir():
        raise ValueError('Gemma tokenizer/temp must be directories')
    if sha(cfg['exe']) != cfg.get('exe_sha256'):
        raise ValueError('Gemma executable identity differs from this candidate binding')
    temp = PureWindowsPath(str(Path(cfg['temp_dir']).resolve()))
    if temp.drive.upper() != 'Y:' or len(temp.parts) < 2 or temp.parts[1].lower() != 'temp':
        raise ValueError('Gemma task temp/cache must use Y:\\temp')
    if any(c.isspace() for c in cfg['temp_dir']):
        raise ValueError('Gemma native wire temp path must have no whitespace')
    binding = cfg.get('asset_binding', {})
    expected_metadata = {'general.architecture':'gemma4', 'gemma4.block_count':60,
        'gemma4.context_length':262144, 'gemma4.embedding_length':5376, 'tokenizer.ggml.model':'gemma4'}
    if any(binding.get('metadata', {}).get(key) != value for key, value in expected_metadata.items()):
        raise ValueError('Gemma model/frontend architecture binding differs')
    model = Path(cfg['model']); stat = model.stat()
    if (binding.get('path') != str(model) or binding.get('bytes') != stat.st_size or
            binding.get('mtime_ns') != stat.st_mtime_ns):
        raise ValueError('Gemma selected asset identity changed')
    count = binding.get('header_bytes')
    if type(count) is not int or not 1 <= count <= min(stat.st_size, 32*1024*1024):
        raise ValueError('Gemma bound GGUF header span is invalid')
    with model.open('rb') as stream:
        actual = hashlib.sha256(stream.read(count)).hexdigest()
    if actual != binding.get('header_sha256'):
        raise ValueError('Gemma selected header identity changed')
    if backend == 'original_quant':
        validate_original_binding(cfg)
    else:
        receipt = json.loads(Path(cfg['conversion_receipt']).read_text(encoding='utf-8-sig'))
        if (receipt.get('passed') is not True or receipt.get('sha256') != binding.get('sha256') or
                receipt.get('source', {}).get('sha256') != SOURCE_SHA or
                receipt.get('converted_tensors') != 410 or receipt.get('untouched_original_tensors') != 423):
            raise ValueError('Gemma requires the original-checkpoint410-body canonical conversion receipt')
    if binding.get('frontend') != check_frontend(cfg['tokenizer']):
        raise ValueError('Gemma original tokenizer/template binding changed')
    validate_mtp(cfg)
    return True
