"""One-time CPU file/identity binding for the isolated Gemma native candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('exe', 'model', 'receipt', 'tokenizer', 'temp-dir', 'codec-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--context', type=int, default=262144)
    parser.add_argument('--max-batch', type=int, default=2048)
    parser.add_argument('--reserve-mib', type=int, default=4096)
    parser.add_argument('--dll-dir', type=Path, action='append', default=[])
    parser.add_argument('--prior-config', type=Path)
    parser.add_argument('--tiny-compensated', action='store_true', help='Opt in to pure-linear T1..3 compensated NVFP4')
    parser.add_argument('--pp-tc', action='store_true', help='Opt in to Gemma key-outer FP16-input/F32-accum PP attention')
    inherited = ('pp-rope-table', 't1-rope-table', 't1-norm-cache-fusion',
        'tg-global-merge', 'tg-local270', 'original-q6-head',
        'pp-global-keys1024', 'pp-tc-local-gqa2')
    for name in inherited:
        parser.add_argument('--'+name, action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        raise ValueError('This explicit profile writer targets the Windows5090 candidate')
    for key in ('exe', 'model', 'receipt', 'tokenizer', 'temp_dir', 'codec_root'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    output = args.output.resolve()
    if output.drive.upper() != 'Y:' or output.parts[1].lower() != 'temp' or output.exists():
        raise ValueError('New candidate config must be an unused Y:\\temp path')
    receipt = json.loads(args.receipt.read_text(encoding='utf-8-sig'))
    before = args.model.stat()
    if str(args.model).casefold() != str(Path(receipt['output']).resolve()).casefold() or before.st_size != receipt['bytes']:
        raise ValueError('Gemma converted file differs from its publication receipt')
    prior = json.loads(args.prior_config.read_text(encoding='utf-8-sig')) if args.prior_config else {}
    old = prior.get('asset_binding', {})
    reusable = (old.get('path') == str(args.model) and old.get('bytes') == before.st_size and
                old.get('mtime_ns') == before.st_mtime_ns and old.get('sha256') == receipt.get('sha256'))
    digest = old['sha256'] if reusable else sha(args.model)
    if digest != receipt['sha256']:
        raise ValueError('Gemma converted full SHA differs from verified publication')
    sys.path.insert(0, str(args.codec_root))
    from gguf_reader import GGUFFile
    reader = GGUFFile(args.model)
    if len(reader.tensors) != 1243 or sum(t.type_id == 40 for t in reader.tensors) != 410:
        raise ValueError('Gemma canonical tensor inventory differs')
    with args.model.open('rb') as stream:
        header_sha = hashlib.sha256(stream.read(reader.data_start)).hexdigest()
    if (before.st_size, before.st_mtime_ns) != (args.model.stat().st_size, args.model.stat().st_mtime_ns):
        raise ValueError('Gemma asset changed during identity binding')
    frontend = {name: sha(args.tokenizer/name) for name in ('tokenizer.json', 'chat_template.jinja')}
    cfg = dict(family='gemma4_dense', backend='nvfp4', variant='gemma4_31b_nvfp4', model=str(args.model),
        exe=str(args.exe), exe_sha256=sha(args.exe), tokenizer=str(args.tokenizer),
        temp_dir=str(args.temp_dir), conversion_receipt=str(args.receipt), gpu_uuid=args.gpu_uuid,
        context=args.context, slots=1, mtp=0, kv_policy='full', max_batch=args.max_batch,
        reserve_mib=args.reserve_mib, cuda_graph=True, tiny_compensated=args.tiny_compensated, pp_tc=args.pp_tc,
        dspark=False, stop_token_ids=[1, 50, 106],
        qualification='UNQUALIFIED_CANDIDATE', default_max_new=2048,
        dll_dirs=[str(path.resolve(strict=True)) for path in args.dll_dir],
        sampling=dict(temperature=0.6, top_k=64, top_p=0.9, min_p=0.05, repetition_penalty=1.1, penalty_last_n=64),
        asset_binding=dict(path=str(args.model), bytes=before.st_size, mtime_ns=before.st_mtime_ns,
            sha256=digest, header_bytes=reader.data_start, header_sha256=header_sha, frontend=frontend,
            metadata={key: reader.metadata[key] for key in ('general.architecture', 'gemma4.block_count',
                'gemma4.context_length', 'gemma4.embedding_length', 'tokenizer.ggml.model')}))
    cfg.update({name.replace('-', '_'): getattr(args, name.replace('-', '_')) for name in inherited})
    cfg['original_input_reuse'] = False
    source_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source_root))
    from serve.gemma_sm120_config import validate_config
    validate_config(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(cfg, indent=2)+'\n')
    print(json.dumps(dict(config=str(output), sha256=sha(output), model_SHA_reused=reusable,
                          GPU=False, model_loads=0, qualification=cfg['qualification'])))


if __name__ == '__main__':
    main()
