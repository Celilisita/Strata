"""Bind the newly source-built V100 Qwen target to the shared HTTP server.

Only file/header identity is read. No engine, GPU, model execution or downloads.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile
from serve.v100_qwen_family import FAMILY, TARGET, VARIANTS, frontend_for_config, header_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    for key in ("exe", "model", "temp-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--dll-dir", action="append", type=Path, default=[])
    parser.add_argument("--mtp", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--kv-policy", choices=("full", "ram8k-reuse"), default="full")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--expected-model-sha256", help="optional already known full payload SHA; verified when supplied")
    args = parser.parse_args()
    for directory in args.dll_dir:
        if not directory.is_absolute() or not directory.is_dir():
            parser.error("--dll-dir must name an existing absolute directory")
    for key in ("exe", "model", "temp_dir", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.exists() or not args.output.is_relative_to(args.temp_dir):
        parser.error("write a new candidate config below its task temp-dir")
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    gguf = GGUFFile(args.model)
    source = gguf.metadata.get("tokenizer.chat_template")
    if not isinstance(source, str) or not source.strip():
        parser.error("the actual GGUF must carry its own chat template")
    stat = args.model.stat()
    binding = dict(path=str(args.model), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                   header_sha256=header_identity(args.model, gguf))
    if args.expected_model_sha256:
        digest = hashlib.sha256()
        with args.model.open("rb") as stream:
            for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(part)
        if digest.hexdigest() != args.expected_model_sha256.lower():
            parser.error("actual model payload differs from --expected-model-sha256")
        binding["sha256"] = digest.hexdigest()
    cfg = dict(family=FAMILY, backend="completed_v100", source_target=TARGET,
        variant=args.variant, exe=str(args.exe), model=str(args.model),
        model_name=args.variant.replace("_", "-"), gpu_uuid=args.gpu_uuid,
        temp_dir=str(args.temp_dir), context=262144, slots=1, mtp=args.mtp,
        kv_policy=args.kv_policy, max_output=65536, asset_binding=binding,
        frontend_binding=dict(template_sha256=hashlib.sha256(source.encode()).hexdigest()),
        dll_dirs=[str(path.resolve()) for path in args.dll_dir],
        sampling=dict(temperature=args.temperature, top_k=20, top_p=0.95, min_p=0),
        env={"STRATA_DENSE_SAMPLED_MTP": "0"}, native_policies=[],
        capabilities=["text", "sampling", "cancel", "reset"], repeat_stop_tokens=0,
        lazy_load=True, open_browser=False, response_timeout_s=900)
    from serve.native_sampling import parameters
    parameters(cfg["sampling"])
    frontend_for_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
