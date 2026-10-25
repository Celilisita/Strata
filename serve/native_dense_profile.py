"""Dense asset and request-budget bindings for the public native family adapter."""
from __future__ import annotations


UNCENSORED_NVFP4 = "qwen38_27b_uncensored_nvfp4"
QWEN35_NVFP4 = "qwen35_27b_nvfp4"


def validate_dense_config(cfg):
    policy = cfg.get("context_policy", {})
    ram = cfg.get("kv_policy") == "ram8k-reuse"
    if policy.get("ram_index_enabled") != ram:
        raise ValueError("Dense RAM index policy differs from its native profile")
    if ram:
        if (policy.get("resident_tokens") not in (32768, 40960) or
                policy.get("selected_tokens") != 8192 or policy.get("keep_queries") != 3):
            raise ValueError("Dense context budget differs from an admitted native profile")
        if policy["resident_tokens"] == 40960:
            if str(cfg.get("env", {}).get("STRATA_DENSE_RAM_GPU_ROWS")) != "40960":
                raise ValueError("Dense 40960-row budget requires its explicit environment setting")
    unset = cfg.get("unset_env", [])
    if (not isinstance(unset, list) or
            any(not isinstance(key, str) or not key.startswith("STRATA_") for key in unset)):
        raise ValueError("native unset_env must list Strata environment keys")
    if set(unset).intersection(cfg.get("env", {})):
        raise ValueError("native environment both sets and unsets the same key")
    from serve.native_head_binding import uses_cold_head, validate_head_binding, SOURCE_ENV, _asset
    if cfg.get("variant") == QWEN35_NVFP4:
        if (cfg.get("head_asset_binding") or cfg.get("head_source_binding") or
                any(key in cfg.get("env", {}) for key in
                    ("STRATA_NVFP4_HEAD_FILE", *SOURCE_ENV.values()))):
            raise ValueError("Qwen3.5 NVFP4 requires its embedded head, without a cold-head binding")
        _asset(cfg.get("asset_binding", {}), cfg.get("model"), "Qwen3.5 derived NVFP4", True)
    if uses_cold_head(cfg):
        validate_head_binding(cfg)


def prepare_environment(cfg, environment):
    effective = dict(environment)
    if cfg.get("family") != "qwen35_dense":
        return effective
    if cfg.get("variant") == QWEN35_NVFP4:
        validate_dense_config(cfg)
    for key in cfg.get("unset_env", []):
        effective.pop(key, None)
    policy = cfg.get("context_policy", {})
    if cfg.get("kv_policy") == "ram8k-reuse":
        rows = str(effective.get("STRATA_DENSE_RAM_GPU_ROWS", "32768"))
        if rows != str(policy.get("resident_tokens")):
            raise ValueError("Dense effective RAM rows differ from the configuration binding")
    from serve.native_head_binding import uses_cold_head, prepare_head_environment
    if uses_cold_head(cfg):
        prepare_head_environment(cfg, effective)
        # Child-process scratch follows the explicitly configured task directory.
        effective.update(TEMP=cfg["temp_dir"], TMP=cfg["temp_dir"], TMPDIR=cfg["temp_dir"])
    elif cfg.get("variant") in ("qwen36_27b", QWEN35_NVFP4):
        # Embedded-head assets must not inherit another model's cold head.
        from serve.native_head_binding import SOURCE_ENV
        effective.pop("STRATA_NVFP4_HEAD_FILE", None)
        for variable in SOURCE_ENV.values():
            effective.pop(variable, None)
        if cfg.get("variant") == QWEN35_NVFP4:
            effective.update(TEMP=cfg["temp_dir"], TMP=cfg["temp_dir"], TMPDIR=cfg["temp_dir"])
    return effective
