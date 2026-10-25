"""Receipt-derived cold head provenance and inexpensive file identity checks."""
from pathlib import Path


SOURCE_ENV = {
    "source_model": "STRATA_NVFP4_HEAD_SOURCE_MODEL",
    "bytes": "STRATA_NVFP4_HEAD_SOURCE_BYTES",
    "sha256": "STRATA_NVFP4_HEAD_SOURCE_MODEL_SHA256",
    "head_sha256": "STRATA_NVFP4_HEAD_SOURCE_HEAD_SHA256",
    "header_sha256": "STRATA_NVFP4_HEAD_SOURCE_HEADER_SHA256",
    "norm_sha256": "STRATA_NVFP4_HEAD_SOURCE_NORM_SHA256",
}


def uses_cold_head(cfg):
    if cfg.get("variant") == "qwen38_27b_uncensored_nvfp4":
        return True
    return (cfg.get("variant") == "qwen36_27b" and
            bool(cfg.get("head_asset_binding") or cfg.get("head_source_binding") or
                 any(key in cfg.get("env", {}) for key in
                     ("STRATA_NVFP4_HEAD_FILE", *SOURCE_ENV.values()))))


def _sha256(value):
    return (isinstance(value, str) and len(value) == 64 and value != "0" * 64 and
            all(c in "0123456789abcdef" for c in value))


def _asset(binding, path, label, require_file_id):
    if not isinstance(binding, dict) or not path or binding.get("path") != str(path):
        raise ValueError(label + " asset path differs from its binding")
    path = Path(path)
    if not path.is_file():
        raise ValueError(label + " asset is missing")
    stat = path.stat()
    expected = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if require_file_id:
        expected.update(device=stat.st_dev, file_id=stat.st_ino)
    if any(type(binding.get(key)) is not int or binding[key] != value
           for key, value in expected.items()):
        raise ValueError(label + " asset file identity changed")
    if require_file_id and not _sha256(binding.get("sha256")):
        raise ValueError(label + " asset needs its previously verified whole-file SHA256")


def validate_head_binding(cfg):
    head = cfg.get("head_asset_binding", {})
    qwen36 = cfg.get("variant") == "qwen36_27b"
    if not head.get("path"):
        raise ValueError("NVFP4 head profile requires a bound head asset")
    _asset(head, head["path"], "NVFP4 head", qwen36)
    environment = cfg.get("env", {})
    if environment.get("STRATA_NVFP4_HEAD_FILE") != head["path"]:
        raise ValueError("NVFP4 head path differs from its explicit asset binding")
    source = cfg.get("head_source_binding")
    if source is None:
        if qwen36 or any(key in environment for key in SOURCE_ENV.values()):
            raise ValueError("NVFP4 head needs all receipt-derived source identity fields")
        return
    if not isinstance(source, dict):
        raise ValueError("head_source_binding must be a complete receipt-derived mapping")
    name, size = source.get("source_model"), source.get("bytes")
    if (not isinstance(name, str) or not name or len(name.encode("utf-8")) > 512 or
            any(ord(c) < 32 or ord(c) == 127 or c in "/\\" for c in name) or
            type(size) is not int or not 0 < size <= 2**50 or
            any(not _sha256(source.get(key)) for key in
                ("sha256", "head_sha256", "header_sha256", "norm_sha256"))):
        raise ValueError("cold head source identity is incomplete or malformed")
    main = cfg.get("asset_binding", {})
    _asset(main, cfg.get("model"), "Dense source", qwen36)
    if source["bytes"] != main.get("bytes") or source["sha256"] != main.get("sha256"):
        raise ValueError("cold head source differs from the verified main model identity")
    if any(str(environment.get(variable)) != str(source[key])
           for key, variable in SOURCE_ENV.items()):
        raise ValueError("cold head source ENV differs from its receipt-derived binding")


def prepare_head_environment(cfg, effective):
    validate_head_binding(cfg)
    if effective.get("STRATA_NVFP4_HEAD_FILE") != cfg["head_asset_binding"]["path"]:
        raise ValueError("effective NVFP4 head differs from the bound asset")
    source = cfg.get("head_source_binding")
    for key, variable in SOURCE_ENV.items():
        if source is None:
            # The historical path must not inherit another model's source tuple.
            effective.pop(variable, None)
        elif str(effective.get(variable)) != str(source[key]):
            raise ValueError("effective cold head source ENV differs from its binding")
