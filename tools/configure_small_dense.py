"""Write a shared-server small_dense config using static file/header identity only."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from dense_prepare import DenseFrontend
from serve.dense_family import VARIANTS, frontend_for_config


def binding(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    for name in ("exe", "model", "temp-dir", "output"):
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--vision-exe", type=pathlib.Path)
    parser.add_argument("--mmproj", type=pathlib.Path)
    parser.add_argument("--profile", choices=("off", "sm70", "sm75"), default="off",
                        help="explicit historical device profile; off is the unqualified-device default")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be in 0..2")
    for key in ("exe", "model", "temp_dir", "output", "vision_exe", "mmproj"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if not args.output.is_relative_to(args.temp_dir) or args.output.exists():
        parser.error("write a new candidate config below its temp-dir")
    if bool(args.vision_exe) != bool(args.mmproj):
        parser.error("vision-exe and mmproj must be specified together")
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    frontend = DenseFrontend(args.model)
    identity = binding(args.model)
    identity["template_sha256"] = frontend.identity["template_sha256"]
    cfg = {"family": "small_dense", "variant": args.variant, "backend": "cuda",
        "exe": str(args.exe), "model": str(args.model), "gpu_uuid": args.gpu_uuid,
        "temp_dir": str(args.temp_dir), "context": args.context or
            (131072 if args.variant == "minicpm5_2b" else 262144),
        "slots": 1, "budget_mib": 12000, "chunk": 512, "max_output": 65536,
        "model_name": args.variant.replace("_", "-"), "asset_binding": identity,
        "features": {}, "hardware_profile": args.profile,
        "sampling": {"temperature": args.temperature, "top_k": 0, "top_p": 1, "seed": 1},
        "api_monitor": False, "repeat_stop_tokens": 0}
    cfg["max_output"] = min(cfg["max_output"], cfg["context"] - 1)
    if args.profile != "off":
        qwen = args.variant == "qwen35_08b"
        cfg["features"] = {"pp_half_cache": 1, "device_greedy": 1, "projection_fusion": 1,
                           "compact_async": int(qwen), "norm_fusion": int(qwen)}
    if args.vision_exe:
        cfg["vision"] = {"exe": str(args.vision_exe), "mmproj": str(args.mmproj),
            "format": "dense_sdv1", "gpu": True, "asset_binding": binding(args.mmproj)}
    frontend_for_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(args.output), "variant": args.variant,
        "static_identity_only": True, "models_executed": 0, "profile": args.profile}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
