"""Completed Gemma adapter for the shared HTTP engine and request lifecycle."""
from __future__ import annotations

import hashlib
import math
import os
import queue
import re
import threading
from pathlib import Path

from serve.native_family import NativeFamilyEngine


def configured(cfg):
    return cfg.get("family") == "gemma4_dense"


def configured_assistant(cfg):
    """Resolve optional greedy MTP before assistant validation or launch."""
    requested = cfg.get("mtp", 0)
    # Other raw sampling fields keep the shared server's normalization rules.
    sampling = {k: v for k, v in (cfg.get("sampling") or {}).items() if v is not None}
    sampling.setdefault("temperature", 0.6)
    sampled = float(sampling["temperature"]) > 0 or any(
        float(sampling.get(key, default)) != default for key, default in
        (("frequency_penalty", 0), ("presence_penalty", 0), ("repetition_penalty", 1)))
    return requested, 0 if sampled else requested, "configured_sampling" if sampled else None


def frontend_for_config(cfg):
    from serve.gemma_frontend import GemmaFrontend
    if cfg.get("variant") != "gemma4_31b_completed":
        raise ValueError("completed Gemma requires its explicit source profile")
    for key in ("exe", "model", "tokenizer_json", "temp_dir"):
        if not cfg.get(key) or not Path(cfg[key]).is_absolute() or not Path(cfg[key]).exists():
            raise ValueError("completed Gemma requires an existing absolute " + key)
    if not Path(cfg["temp_dir"]).is_dir():
        raise ValueError("Gemma temp_dir must be a directory")
    from serve.temp_paths import require_task_temp_dir
    require_task_temp_dir(cfg["temp_dir"], "Gemma")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", cfg.get("gpu_uuid", "")):
        raise ValueError("completed Gemma requires an explicit GPU UUID")
    if cfg.get("context") != 262144 or cfg.get("slots", 1) != 1 or cfg.get("mtp", 0) not in (0, 1):
        raise ValueError("completed Gemma uses one slot, 256K logical context and optional MTP1")
    if cfg.get("dspark") or cfg.get("vision"):
        raise ValueError("this completed Gemma module is a text Dense provider")
    model = Path(cfg["model"]); stat = model.stat(); binding = cfg.get("asset_binding", {})
    if (binding.get("path") != str(model) or binding.get("bytes") != stat.st_size or
            binding.get("mtime_ns") != stat.st_mtime_ns):
        raise ValueError("completed Gemma model identity changed")
    path = Path(cfg["tokenizer_json"])
    if cfg.get("frontend_binding", {}).get("tokenizer_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Gemma tokenizer.json identity changed")
    frontend = GemmaFrontend(cfg)
    if cfg["frontend_binding"].get("template_sha256") != frontend.template_hash:
        raise ValueError("actual Gemma GGUF template identity changed")
    if cfg.get("assistant") and configured_assistant(cfg)[2] is None:
        path = Path(cfg["assistant"]); bound = cfg.get("assistant_binding", {})
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Gemma assistant is missing")
        stat = path.stat()
        if (bound.get("path") != str(path) or bound.get("bytes") != stat.st_size or
                bound.get("mtime_ns") != stat.st_mtime_ns):
            raise ValueError("Gemma assistant identity changed")
    directories = cfg.get("dll_dirs", [])
    if not isinstance(directories, list) or (os.name == "nt" and not directories):
        raise ValueError("this source-built CUDA engine requires explicit dll_dirs")
    for directory in directories:
        path = Path(directory)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("dll_dirs must contain existing absolute directories")
    cfg.setdefault("max_output", 65536)
    cfg.setdefault("repeat_stop_tokens", 0)
    timeout = cfg.setdefault("response_timeout_s", 900)
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Gemma response_timeout_s must be a finite positive number")
    return frontend


class GemmaFamilyEngine(NativeFamilyEngine):
    def __init__(self, cfg, frontend, environment, lazy=False):
        self.cfg, self.frontend, self.family = cfg, frontend, "gemma4_dense"
        self.max_context, self.batch, self.concurrent_slots = cfg["context"], 1, 1
        self.stops = set(frontend.stop_ids)
        self.environment = dict(environment, TEMP=cfg["temp_dir"], TMP=cfg["temp_dir"],
            TMPDIR=cfg["temp_dir"], CUDA_VISIBLE_DEVICES=cfg["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if cfg.get("dll_dirs"):
            self.environment["PATH"] = os.pathsep.join(cfg["dll_dirs"] + [self.environment.get("PATH", "")])
        requested, launch, reason = configured_assistant(cfg)
        self.info = dict(backend="native-gemma4-completed", family=self.family,
            variant=cfg["variant"], gpu_uuid=cfg["gpu_uuid"], request_slots=1, batch_slots=1,
            mtp_requested=requested, mtp_launch=launch, mtp_effective=0,
            assistant_requested=bool(cfg.get("assistant")),
            assistant_launch_enabled=bool(cfg.get("assistant")) and bool(launch),
            assistant_disabled_reason=reason, capabilities=["text", "sampling", "cancel", "reset"])
        self._last, self.runner = {}, None
        self._gate = threading.RLock()
        self._draining = None
        self.request_state = threading.local()
        self.progress = self.prompt_perf = self.request_phase = None
        self.frontend_facts = {}; self.unloaded = lazy
        if not lazy:
            self.restart()

    def restart(self):
        from serve.gemma_engine import GemmaRunner
        with self._gate:
            self.close()
            self.runner = GemmaRunner(self.cfg, self.stops, self.environment)
            self.info.update(mtp_launch=self.runner.mtp_launch, mtp_effective=0,
                             assistant_disabled_reason=self.runner.assistant_disabled_reason)
            self.unloaded = False

    def _require_idle(self):
        if self._draining is not None and not self._draining.is_set():
            from serve.server import EngineStuck
            raise EngineStuck("Gemma request is still draining; restart/unload is blocked")

    def close(self):
        with self._gate:
            self._require_idle()
            super().close()

    def unload(self):
        with self._gate:
            self.close(); self.unloaded = True

    def validate_request(self, req):
        if req.get("strata_prefix") is not None:
            raise ValueError("completed Gemma prefix resume is not supported")

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError("completed Gemma accepts text requests")
        from serve.native_sampling import parameters
        selection = {k: v for k, v in (sampling or {}).items() if v is not None}
        selection.setdefault("temperature", 0.6)
        parameters(selection)
        # A single owner holds this gate until Runner.generate's RELEASE finally
        # finishes. Cancelled HTTP iteration never makes a running slot idle.
        with self._gate:
            self._require_idle()
            if self.runner is None:
                self.restart()
            runner = self.runner
            events, stop, done, finished = queue.Queue(), threading.Event(), threading.Event(), {}
            self._draining = done
            self.request_phase = "prefilling"
            self.request_state.last = {}

            class Cancelled:
                def is_set(self):
                    return stop.is_set() or cancel.is_set()

            def work():
                try:
                    _, stats = runner.generate(ids, max_new, self.stops,
                        on_token=lambda token: events.put(("token", token)),
                        cancel=Cancelled(), sampling=selection)
                    native = stats.get("native_info", {})
                    reason = runner.assistant_disabled_reason or (
                        "request_sampling" if int(native.get("mtp_sampling_fallback", 0)) else None)
                    finished["last"] = dict(prompt_ms=stats["pp_ms"],
                        decode_ms=stats["total_ms"]-stats["pp_ms"], generated=stats["generated"],
                        tg_steps=max(0, stats["generated"]-1), prompt_tokens=len(ids), reused=0,
                        cache_slots=0, drafts_offered=stats["checked_proposals"],
                        drafts_accepted=stats["accepted_proposals"],
                        speculative_enabled=stats.get("native_speculative", False),
                        native_info=native, mtp_requested=runner.mtp_requested,
                        mtp_launch=runner.mtp_launch, mtp_effective=int(native.get("actual_mtp", 0)),
                        assistant_disabled_reason=reason)
                    self.info.update(mtp_effective=int(native.get("actual_mtp", 0)),
                                     assistant_disabled_reason=reason)
                    events.put(("done", None))
                except BaseException as error:
                    events.put(("error", error))
                finally:
                    done.set()

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
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
                        if not self.alive():
                            from serve.server import EngineDied
                            raise EngineDied("Gemma engine exited during generation") from value
                        raise value
            finally:
                stop.set()
                if not done.wait(float(self.cfg["response_timeout_s"])):
                    from serve.server import EngineDied
                    runner.abort()
                    # Killing the process releases the pending IPC read. A rare
                    # Python-side stall stays marked draining and blocks reuse.
                    if done.wait(10):
                        worker.join()
                    raise EngineDied("Gemma cancel/RELEASE exceeded response_timeout_s; engine ended")
                worker.join()
                self._draining = None
                self.last = finished.get("last", {})
                self.request_phase = "idle"
