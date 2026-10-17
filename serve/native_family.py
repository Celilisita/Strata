"""Config-bound adapters for native libraries in the public Strata executable."""
from __future__ import annotations

import os
import hashlib
import queue
import subprocess
import threading
from pathlib import Path


class NativeFamilyEngine:
    def __init__(self, cfg, tokenizer, environment):
        from serve.native_engine import NativeFamilyRunner
        self.cfg = cfg
        self.max_context = int(cfg["context"])
        self.batch = int(cfg.get("slots", 1))
        self.concurrent_slots = self.batch
        self.family = cfg["family"]
        self.stops = {tokenizer.ids[t] for t in ("<|im_end|>", "<|endoftext|>", "<|fim_suffix|>")
                      if t in tokenizer.ids}
        if not self.stops:
            raise ValueError("native tokenizer has no configured stop token")
        self.stops.update(int(t) for t in cfg.get("stop_token_ids", []))
        self.environment = dict(environment)
        gpu = cfg["gpu_uuid"]
        identity = subprocess.run(["nvidia-smi", "--id=" + gpu,
                                   "--query-gpu=uuid,compute_cap", "--format=csv,noheader"],
                                  capture_output=True, text=True, check=True).stdout.strip().split(",")
        if len(identity) != 2 or identity[0].strip() != gpu or float(identity[1]) < 7:
            raise ValueError("native device identity/capability mismatch")
        self.environment.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
        directories = cfg.get("dll_dirs", [])
        if directories:
            self.environment["PATH"] = os.pathsep.join(directories + [self.environment.get("PATH", "")])
        self.info = {"backend": "native-" + self.family, "gpu_uuid": gpu,
                     "request_slots": self.batch, "batch_slots": self.batch,
                     "dspark": bool(cfg.get("dspark", False)), "mtp": int(cfg.get("mtp", 0)),
                     "kv_policy": cfg.get("kv_policy", "on-demand"),
                     "capabilities": cfg["capabilities"]}
        self._last = {}; self.request_state = threading.local()
        self.progress = self.prompt_perf = self.request_phase = None
        self.frontend_facts = {}; self.runner = None
        self.runner_type = NativeFamilyRunner
        self.restart()

    @property
    def last(self):
        return getattr(self.request_state, "last", self._last)

    @last.setter
    def last(self, value):
        self._last = value
        self.request_state.last = value

    @property
    def proc(self):
        return None if self.runner is None else self.runner.proc

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self):
        return None if self.proc is None else self.proc.poll()

    def death_note(self):
        return "native family engine exited"

    def request_last(self):
        return getattr(self.request_state, "last", {})

    def restart(self):
        self.close()
        cfg = self.cfg
        self.runner = self.runner_type(
            Path(cfg["exe"]), Path(cfg["model"]),
            Path(cfg["draft"]) if cfg.get("draft") else None,
            Path(cfg["gguf_python"]) if cfg.get("gguf_python") else None,
            self.max_context, Path(cfg["temp_dir"]), int(cfg.get("block_size", 4)),
            self.batch, bool(cfg.get("dspark", False)), family=self.family,
            variant=cfg["variant"], mtp=int(cfg.get("mtp", 0)),
            kv_policy=cfg.get("kv_policy", "ram8k-reuse"), environment=self.environment)

    def close(self):
        if self.runner is not None:
            self.runner.close(); self.runner = None

    def slots_view(self):
        if self.runner is not None:
            return self.runner.slot_status.view()
        return [{"slot": slot, "id": slot, "state": "idle", "held_tokens": 0}
                for slot in range(self.batch)]

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError("native families use model-specific prepared visual spans")
        from serve.native_sampling import parameters
        selection = {k: v for k, v in (sampling or {}).items() if v is not None}
        parameters(selection)
        if self.info["dspark"] and (float(selection.get("temperature", 0) or 0) > 0 or
                any(float(selection.get(k, default)) != default for k, default in
                    (("frequency_penalty", 0), ("presence_penalty", 0), ("repetition_penalty", 1)))):
            raise ValueError("DSpark supports greedy generation without penalties")
        events = queue.Queue(); stop = threading.Event(); finished = {}
        self.request_phase = "prefilling"
        self.request_state.last = {}

        class Cancelled:
            def is_set(self):
                return stop.is_set() or cancel.is_set()

        def work():
            try:
                _, stats = self.runner.generate(ids, max_new, self.stops,
                    on_token=lambda token: events.put(("token", token)),
                    cancel=Cancelled(), sampling=selection)
                finished["last"] = {"prompt_ms": stats["pp_ms"],
                    "decode_ms": stats["total_ms"] - stats["pp_ms"],
                    "generated": stats["generated"], "tg_steps": max(0, stats["generated"] - 1),
                    "reused": 0, "cache_slots": 0, "prompt_tokens": len(ids)}
                self.last = finished["last"]
                events.put(("done", None))
            except BaseException as error:
                events.put(("error", error))
        worker = threading.Thread(target=work, daemon=True); worker.start()
        try:
            while not cancel.is_set():
                try:
                    kind, value = events.get(timeout=0.5)
                except queue.Empty:
                    yield None; continue
                if kind == "token":
                    self.request_phase = "generating"; yield value
                elif kind == "done":
                    return
                else:
                    raise value
        finally:
            stop.set(); worker.join(timeout=10)
            self.request_state.last = finished.get("last", {})


def validate_config(cfg):
    family = cfg.get("family")
    if family not in ("qwen35_moe", "qwen35_dense"):
        return False
    variants = ("minicpm_v47", "qwen36_35b") if family == "qwen35_moe" else ("qwen38_27b",)
    if cfg.get("variant") not in variants or not cfg.get("gpu_uuid"):
        raise ValueError("native config must bind family, variant and device identity")
    required = ("exe", "model", "tokenizer", "temp_dir")
    for key in required:
        if not cfg.get(key) or not Path(cfg[key]).exists():
            raise ValueError("missing native asset/path: " + key)
    if not Path(cfg["temp_dir"]).is_dir() or not Path(cfg["tokenizer"]).is_dir():
        raise ValueError("native tokenizer/temp paths must be directories")
    binding = cfg.get("asset_binding", {})
    model = Path(cfg["model"])
    if binding.get("path") != str(model) or binding.get("bytes") != model.stat().st_size or binding.get("mtime_ns") != model.stat().st_mtime_ns:
        raise ValueError("native model asset identity changed; regenerate the candidate binding")
    expected = "qwen35moe" if family == "qwen35_moe" else "qwen35"
    if binding.get("metadata", {}).get("general.architecture") != expected:
        raise ValueError("native metadata architecture differs from selected family")
    if not binding.get("frontend"):
        raise ValueError("native frontend identity is missing")
    for name, digest in binding["frontend"].items():
        if Path(name).name != name or hashlib.sha256((Path(cfg["tokenizer"]) / name).read_bytes()).hexdigest() != digest:
            raise ValueError("native tokenizer/template identity changed")
    if any(c.isspace() for c in cfg["temp_dir"]):
        raise ValueError("native wire payload directory must have no whitespace")
    if cfg.get("vision"):
        for key in ("exe", "model", "mmproj"):
            if not Path(cfg["vision"][key]).is_file():
                raise ValueError("native visual asset is missing: " + key)
    if cfg.get("dspark"):
        if cfg.get("variant") != "minicpm_v47" or cfg.get("slots", 1) != 1:
            raise ValueError("DSpark requires MiniCPM single slot")
        for key in ("draft", "gguf_python"):
            if not cfg.get(key) or not Path(cfg[key]).exists():
                raise ValueError("DSpark requires " + key)
    elif family == "qwen35_dense" and cfg.get("slots", 1) != 1:
        raise ValueError("Dense currently supports one slot")
    if family == "qwen35_dense":
        policy = cfg.get("context_policy", {})
        ram = cfg.get("kv_policy") == "ram8k-reuse"
        if policy.get("ram_index_enabled") != ram or (ram and
                any(policy.get(k) != v for k, v in (("resident_tokens", 32768), ("selected_tokens", 8192), ("keep_queries", 3)))):
            raise ValueError("Dense context policy differs from its admitted native profile")
    if cfg.get("vision"):
        expected_format = "minicpm_canvas" if cfg["variant"] == "minicpm_v47" else "qwen"
        if cfg["vision"].get("format", "qwen") != expected_format:
            raise ValueError("native visual format differs from selected model variant")
        if os.path.normcase(cfg["vision"]["model"]) != os.path.normcase(cfg["model"]):
            raise ValueError("native projector and decoder assets differ")
    return True
