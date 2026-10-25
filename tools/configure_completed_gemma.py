"""Bind completed Gemma source-built entry to the normal shared HTTP server.

Reads GGUF metadata and local tokenizer assets only. It never starts an engine,
loads tensors, downloads a tokenizer, rewrites a launcher, or edits a router.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gguf_reader import GGUFFile


def identity(path):
    p = Path(path).resolve(); stat = p.stat()
    return dict(path=str(p), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("exe", "model", "tokenizer-json", "gpu-uuid", "temp-dir", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--assistant")
    parser.add_argument("--dll-dir", action="append", type=Path, default=[])
    parser.add_argument("--mtp", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    for directory in args.dll_dir:
        if not directory.is_absolute() or not directory.is_dir():
            parser.error("--dll-dir must name an existing absolute directory")
    model, tokenizer = Path(args.model).resolve(), Path(args.tokenizer_json).resolve()
    md = GGUFFile(model).metadata
    source = md.get("tokenizer.chat_template")
    if md.get("general.architecture") != "gemma4" or not isinstance(source, str) or not source.strip():
        raise ValueError("bind an actual Gemma4 Dense GGUF with its own chat template")
    cfg = dict(family="gemma4_dense", variant="gemma4_31b_completed",
        model_name=md.get("general.name", "gemma4-completed"), exe=str(Path(args.exe).resolve()),
        model=str(model), tokenizer=str(tokenizer.parent), tokenizer_json=str(tokenizer),
        gpu_uuid=args.gpu_uuid, temp_dir=str(Path(args.temp_dir).resolve()), context=262144,
        slots=1, mtp=args.mtp, asset_binding=identity(model),
        frontend_binding=dict(tokenizer_sha256=hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                              template_sha256=hashlib.sha256(source.encode()).hexdigest()),
        sampling=dict(temperature=0.6, top_k=20, top_p=0.95, min_p=0),
        capabilities=["text", "sampling", "cancel", "reset"], repeat_stop_tokens=0,
        lazy_load=True, open_browser=False, history_experimental=True,
        dll_dirs=[str(path.resolve()) for path in args.dll_dir])
    if args.assistant:
        cfg.update(assistant=str(Path(args.assistant).resolve()), assistant_binding=identity(args.assistant))
    from serve.gemma_family import frontend_for_config
    frontend_for_config(cfg)
    output = Path(args.output).resolve()
    if not output.is_relative_to(Path(cfg["temp_dir"])):
        raise ValueError("write the new candidate config below its task temp-dir")
    if output.exists():
        raise ValueError("choose a new config output; existing configs are retained")
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
