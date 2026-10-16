"""Inspect one explicit local model directory without loading weights or importing model code."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

from header_reader import gguf, safetensors

MAX_JSON = 16 * 1024 * 1024
MAX_FILES = 1024
CONFIG_FILES = ("config.json", "tokenizer_config.json", "generation_config.json",
                "preprocessor_config.json", "processor_config.json",
                "special_tokens_map.json", "model.safetensors.index.json")


def json_file(path):
    if path.stat().st_size > MAX_JSON:
        raise ValueError(f"metadata JSON exceeds {MAX_JSON} bytes: {path.name}")
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def stat_record(path, root):
    stat = path.stat()
    return {"path": path.relative_to(root).as_posix(), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def compact_metadata(metadata):
    result = {}
    for key, value in metadata.items():
        if isinstance(value, list) and len(value) > 128:
            encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
            result[key] = {"count": len(value), "decoded_value_sha256": hashlib.sha256(encoded).hexdigest(),
                           "first": value[:4], "last": value[-4:]}
        else:
            result[key] = value
    return result


def category(name):
    if name.startswith(("v.", "mm.")) or re.search(r"(?:^|\.)(vision|visual|vision_tower|image_encoder|aligner|mmproj)(?:\.|$)", name):
        return "vision"
    if name.startswith("mtp.") or re.search(r"(?:^|\.)(mtp|nextn|dspark)(?:\.|$)", name):
        return "draft"
    if re.search(r"\.ffn_(?:gate|up|down)_exps\.weight$", name) or ("experts" in name and "shared_experts" not in name):
        return "routed_expert"
    return "main_dense_or_shared"


def inspect(root, output):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model root must be a directory")
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside the read-only model directory")
    files = sorted(p for p in root.iterdir() if p.is_file() and not p.is_symlink())
    if len(files) > MAX_FILES:
        raise ValueError("file count exceeds bounded inventory limit")
    configs, issues = {}, []
    for name in CONFIG_FILES:
        path = root / name
        if path.is_file():
            value, digest = json_file(path)
            configs[name] = {"sha256": digest, "value": value}
    records, tensor_files = [], {}
    counters, storage = collections.Counter(), collections.Counter()
    categories = collections.Counter()
    samples, layer_ids, expert_ids = {}, set(), set()
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = output.with_name(output.stem + ".tensors.jsonl")
    with tensor_path.open("w", encoding="utf-8") as sink:
        for path in files:
            record = stat_record(path, root)
            if path.suffix.lower() in (".gguf", ".safetensors"):
                header = gguf(path) if path.suffix.lower() == ".gguf" else safetensors(path)
                tensors = header.pop("tensors")
                record.update(header)
                record["metadata"] = compact_metadata(record["metadata"])
                record["tensor_count"] = len(tensors)
                for tensor in tensors:
                    name = tensor["name"]
                    if name in tensor_files:
                        issues.append(f"duplicate tensor: {name}")
                    tensor_files[name] = path.name
                    counters[tensor["dtype"]] += 1
                    storage[tensor["dtype"]] += tensor.get("stored_bytes", tensor.get("extent_with_padding", 0))
                    categories[category(name)] += 1
                    layer_match = re.search(r"(?:layers|blk)\.(\d+)\.", name)
                    if layer_match:
                        layer_ids.add(int(layer_match[1]))
                    expert_match = re.search(r"experts\.(\d+)\.", name)
                    if expert_match:
                        expert_ids.add(int(expert_match[1]))
                    sample_key = re.sub(r"(?<=\.)\d+(?=\.)", "*", name)
                    samples.setdefault(sample_key, dict(tensor, file=path.name))
                    sink.write(json.dumps(dict(tensor, file=path.name), ensure_ascii=False) + "\n")
                if stat_record(path, root) != {key: record[key] for key in ("path", "bytes", "mtime_ns")}:
                    raise ValueError(f"file changed during inventory: {path.name}")
            records.append(record)
    index = configs.get("model.safetensors.index.json", {}).get("value", {}).get("weight_map", {})
    if index:
        for shard in sorted(set(index.values())):
            if not (root / shard).is_file():
                issues.append(f"missing indexed shard: {shard}")
        missing = set(index) - set(tensor_files)
        unexpected = set(tensor_files) - set(index)
        for name in sorted(missing)[:64]:
            issues.append(f"missing indexed tensor: {name}")
        for name in sorted(unexpected)[:64]:
            issues.append(f"unindexed tensor: {name}")
        mismatched = [name for name in index.keys() & tensor_files.keys() if index[name] != tensor_files[name]]
        for name in mismatched[:64]:
            issues.append(f"tensor shard differs from index: {name}")
        index_check = {"missing_tensors": len(missing), "unindexed_tensors": len(unexpected),
                       "wrong_shard": len(mismatched)}
        configs["model.safetensors.index.json"]["value"] = {
            "metadata": configs["model.safetensors.index.json"]["value"].get("metadata", {}),
            "tensor_count": len(index), "shards": sorted(set(index.values()))}
    else:
        index_check = None
    report = {"schema": "strata.deepseek_v4.inventory.v1", "root": str(root),
              "tensor_payload_bytes_read": 0, "full_weight_hashes_computed": False,
              "file_count": len(files), "files": records, "configs": configs,
              "tensor_count": sum(counters.values()), "dtype_counts": dict(counters),
              "storage_bytes_by_dtype": dict(storage), "category_counts": dict(categories),
              "layer_ids": sorted(layer_ids), "expert_ids": sorted(expert_ids),
              "tensor_representatives": list(samples.values()), "index_check": index_check,
              "issues": issues, "tensor_inventory": str(tensor_path),
              "machine": {"computer_name": os.environ.get("COMPUTERNAME"),
                          "TEMP": os.environ.get("TEMP"), "TMP": os.environ.get("TMP"), "cwd": os.getcwd()},
              "status": "identity_only_not_runtime_support"}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("root", "file_count", "tensor_count", "dtype_counts", "category_counts", "index_check", "issues")}, ensure_ascii=False))


def list_root(root):
    root = root.resolve(strict=True)
    entries = sorted(root.iterdir())
    if len(entries) > MAX_FILES:
        raise ValueError("root entry count exceeds limit")
    print(json.dumps({"root": str(root), "entries": [
        {"name": p.name, "directory": p.is_dir(), "bytes": p.stat().st_size if p.is_file() else None}
        for p in entries if not p.is_symlink()]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list_only:
        list_root(args.root)
    elif args.output:
        inspect(args.root, args.output)
    else:
        parser.error("--output is required unless --list-only is used")
