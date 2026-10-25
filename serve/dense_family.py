"""Small dense family adapter for the shared Strata server and configuration."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

VARIANTS = {
    "minicpm5_2b": ("llama", "minicpm5", "ec2d5801640099e97d8d7e8003ad4d81f336e757811f03a26173dddf386602fd"),
    "qwen35_08b": ("qwen35", "qwen35", "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"),
}
FEATURES = ("pp_half_cache", "device_greedy", "projection_fusion", "compact_async", "norm_fusion")


def configured(cfg):
    return cfg.get("family") == "small_dense"


def frontend_for_config(cfg):
    """Read actual GGUF tokenizer/template metadata once; never load tensor weights."""
    from dense_prepare import DenseFrontend
    if cfg.get("variant") not in VARIANTS:
        raise ValueError("small_dense variant must be minicpm5_2b or qwen35_08b")
    for key in ("exe", "model", "temp_dir"):
        if not cfg.get(key) or not Path(cfg[key]).is_absolute() or not Path(cfg[key]).exists():
            raise ValueError("small_dense requires an existing absolute " + key)
    if not Path(cfg["temp_dir"]).is_dir():
        raise ValueError("small_dense temp_dir must be a directory")
    from serve.temp_paths import require_task_temp_dir
    require_task_temp_dir(cfg["temp_dir"], "small_dense")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", cfg.get("gpu_uuid", "")):
        raise ValueError("small_dense requires an explicit complete GPU UUID")
    if cfg.get("slots", 1) != 1 or cfg.get("mtp", 0) or cfg.get("dspark", False):
        raise ValueError("small_dense supports one native decoder slot without MTP/DSpark")
    path = Path(cfg["model"])
    binding = cfg.get("asset_binding", {})
    arch, pre, digest = VARIANTS[cfg["variant"]]
    if (binding.get("path") != str(path) or binding.get("bytes") != path.stat().st_size or
            binding.get("mtime_ns") != path.stat().st_mtime_ns or binding.get("sha256") != digest):
        raise ValueError("small_dense asset identity changed; regenerate its static config binding")
    cfg.setdefault("repeat_stop_tokens", 0)
    frontend = DenseFrontend(path)
    if frontend.identity["architecture"] != arch or frontend.identity["pre"] != pre:
        raise ValueError("small_dense actual GGUF architecture/tokenizer differs from selected variant")
    if binding.get("template_sha256") != frontend.identity["template_sha256"]:
        raise ValueError("small_dense actual GGUF template differs from its static binding")
    context = cfg.get("context")
    if type(context) is not int or not 256 <= context <= frontend.md[arch + ".context_length"] or context % 256:
        raise ValueError("small_dense context must be a trained-context multiple of 256")
    cfg.setdefault("max_output", min(65536, context - 1))
    # This engine has one physical PP graph size. Equal batch/ubatch requests
    # bind that actual count; unequal two-level scheduling is not implemented.
    if "batch" in cfg or "ubatch" in cfg:
        batch, ubatch = cfg.get("batch"), cfg.get("ubatch")
        if (type(batch) is not int or type(ubatch) is not int or
                not 1 <= batch <= 8192 or batch != ubatch):
            raise ValueError("small_dense requires equal batch/ubatch in 1..8192")
        if "chunk" in cfg and (type(cfg["chunk"]) is not int or cfg["chunk"] != ubatch):
            raise ValueError("small_dense chunk differs from actual batch/ubatch")
        cfg["chunk"] = ubatch
    for key, low, high in (("budget_mib", 1, 12000), ("chunk", 1, 8192), ("max_output", 1, context - 1)):
        value = cfg.get(key, {"budget_mib": 12000, "chunk": 512, "max_output": 65536}[key])
        if type(value) is not int or not low <= value <= high:
            raise ValueError("small_dense invalid " + key)
    features = cfg.get("features", {})
    if not isinstance(features, dict) or set(features) - set(FEATURES):
        raise ValueError("small_dense features contains unsupported keys")
    if any(type(value) is not int or value not in (0, 1) for value in features.values()):
        raise ValueError("small_dense feature values must be 0 or 1")
    if features.get("norm_fusion") and cfg["variant"] != "qwen35_08b":
        raise ValueError("normalization fusion is Qwen prefill only")
    if cfg.get("vision"):
        if cfg["variant"] != "qwen35_08b":
            raise ValueError("MiniCPM5-2B is text only")
        vision = cfg["vision"]
        for key in ("exe", "mmproj"):
            if not Path(vision.get(key, "")).is_absolute() or not Path(vision.get(key, "")).is_file():
                raise ValueError("small_dense visual asset is missing: " + key)
        projector = Path(vision["mmproj"])
        binding = vision.get("asset_binding", {})
        if (binding.get("path") != str(projector) or binding.get("bytes") != projector.stat().st_size or
                binding.get("mtime_ns") != projector.stat().st_mtime_ns or binding.get("sha256") !=
                "2b167b9d9a96bf18c31ece6de70aedca4034d936bd9aba21bd485b9a6dc78da5"):
            raise ValueError("small_dense projector identity changed; regenerate its static binding")
        if vision.get("format") != "dense_sdv1":
            raise ValueError("small_dense vision requires its SDV1/DVI1 bridge")
    return frontend


class DenseTemplate:
    """The same real GGUF template used by the accepted native dense frontend."""
    def __init__(self, frontend):
        self.frontend = frontend
        self.source = frontend.md[frontend.template_key]
        self.caps = {}
        self.tool_dialect = ""

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        return self.frontend.template.render(**dict(self.frontend.template_tokens, **kwargs),
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)


class DenseFamilyEngine:
    """Adapt the accepted RUN/PARAM/CANCEL process to the shared Engine protocol."""
    def __init__(self, cfg, frontend, environment, lazy=False):
        self.cfg, self.frontend = cfg, frontend
        self.max_context = cfg["context"]
        self.batch = self.concurrent_slots = 1
        self.model_path = cfg["model"]
        self.environment = dict(environment, TEMP=cfg["temp_dir"], TMP=cfg["temp_dir"],
            TMPDIR=cfg["temp_dir"], STRATA_TEMP_ROOT=cfg["temp_dir"],
            CUDA_VISIBLE_DEVICES=cfg["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.info = {"backend": "native-small-dense", "family": "small_dense",
            "variant": cfg["variant"], "gpu_uuid": cfg["gpu_uuid"], "request_slots": 1,
            "batch_slots": 1, "capabilities": ["text", "sampling", "cancel", "reset"]}
        self.request_state = threading.local()
        self._last, self.runner = {}, None
        self.progress = self.prompt_perf = self.request_phase = None
        self.unloaded = lazy
        from serve.dense_family_vision import DenseFamilyVision
        self.vision = DenseFamilyVision(self) if cfg.get("vision") else None
        if not lazy:
            if self.vision is not None:
                self.vision.restart()
            try:
                self.restart()
            except BaseException:
                if self.vision is not None:
                    self.vision.shutdown()
                raise

    @property
    def proc(self):
        return self.runner.worker.process if self.runner is not None else None

    @property
    def last(self):
        return getattr(self.request_state, "last", self._last)

    def request_last(self):
        return getattr(self.request_state, "last", {})

    def alive(self):
        return self.runner is not None and self.runner.alive()

    def exit_code(self):
        return self.proc.poll() if self.proc is not None else None

    def death_note(self):
        return "small dense native engine exited"

    def restart(self):
        from serve.dense_engine import DenseEngine
        self.close()
        cfg = self.cfg
        self.runner = DenseEngine(cfg["exe"], cfg["model"], self.max_context,
            cfg.get("budget_mib", 12000), cfg.get("chunk", 512), Path(cfg["temp_dir"]),
            self.environment, cfg.get("startup_timeout", 1800), cfg.get("request_timeout", 7200),
            task_dir=cfg["temp_dir"], **{key: cfg.get("features", {}).get(key, 0) for key in FEATURES})
        self.info["native_ready"] = self.runner.ready
        self.unloaded = False

    def close(self):
        if self.runner is not None:
            self.runner.close()
            self.runner = None

    def unload(self):
        self.close()
        self.unloaded = True

    def slots_view(self):
        return [{"slot": 0, "id": 0, "state": "busy" if self.runner and self.runner.active else "idle",
                 "held_tokens": 0}]

    def validate_request(self, req):
        for key in ("frequency_penalty", "presence_penalty", "min_p"):
            if req.get(key, 0) not in (None, 0):
                raise ValueError("small_dense sampler does not support " + key)
        if req.get("min_keep", 1) not in (None, 1):
            raise ValueError("small_dense sampler supports min_keep=1")
        if req.get("strata_prefix") is not None:
            raise ValueError("small_dense prefix resume is not supported")

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        from serve.dense_engine import sampling_of
        if embeddings:
            raise ValueError("small_dense uses prepared DVI1 visual payloads")
        values = {key: value for key, value in (sampling or {}).items() if value is not None}
        values.setdefault("temperature", 0.6)
        self.validate_request(values)
        if "repetition_penalty" in values:
            values["repeat_penalty"] = values["repetition_penalty"]
        prepared = dict(getattr(ids, "dense_prepared", {}))
        prepared.update(input_ids=list(ids), prompt_tokens=len(ids), max_new_tokens=max_new,
                        stop_ids=self.frontend.stop_ids)
        self.request_state.last = {}
        from serve.prefill_metrics import PrefillMeter
        self.prefill_meter = PrefillMeter(len(ids))
        self.progress = (0, len(ids))
        self.request_phase = "prefilling"
        runner, previous = self.runner, self.runner.last
        iterator = runner.generate(prepared, sampling_of(values, len(self.frontend.tokenizer.tokens), self.max_context), cancel)
        try:
            for item in iterator:
                if "prefill" in item:
                    self.prefill_meter.update(*item["prefill"])
                    self.progress = item["prefill"][:2]
                    yield None
                if "token" in item:
                    self.prefill_meter.complete()
                    self.request_phase = "generating"
                    yield item["token"]
        except Exception as error:
            if not runner.alive():
                from serve.server import EngineDied
                raise EngineDied("small dense engine exited during generation") from error
            raise
        finally:
            iterator.close()
            done = runner.last
            if done and done is not previous:
                self._last = {"prompt_ms": done["pp_ms"], "decode_ms": done["tg_ms"],
                    "generated": done["output_tokens"], "tg_steps": max(0, done["output_tokens"] - 1),
                    "prompt_tokens": done["input_tokens"], "reused": 0, "cache_slots": 0,
                    "drafts_offered": 0, "drafts_accepted": 0, "native_cost": done}
                self.request_state.last = self._last
            self.request_phase = "idle"
