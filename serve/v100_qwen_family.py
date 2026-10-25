"""Explicit completed-V100 Qwen backend; independent of the final223 loader."""
from __future__ import annotations
import hashlib
import math
import os
import queue
import re
import threading
from pathlib import Path
from serve.native_family import NativeFamilyEngine

FAMILY = "qwen35_dense_v100"
TARGET = "strata-qwen-v100-native"
VARIANTS = {"qwen35_27b": "raw_nvfp4_mixed", "qwen36_27b": "q4k_q6k", "qwen38_27b": "q4k_q6k"}
POLICIES = {"MATRIXPOLICY", "MTPPOLICY", "PPPOLICY", "SHAREDPREPARE",
            "HEADPOLICY", "SERVICEABORT", "ATTENTIONPOLICY"}
ENV = {"STRATA_DENSE_SAMPLED_MTP", "STRATA_DENSE_REUSE_EAGER_PP",
       "STRATA_NVFP4_PACKED_DECODE", "STRATA_NVFP4_LIBRARY_DECODE"}


def configured(cfg):
    return cfg.get("family") == FAMILY


def header_identity(model, gguf):
    with model.open("rb") as stream:
        return hashlib.sha256(stream.read(gguf.data_start)).hexdigest()


def frontend_for_config(cfg):
    from gguf_reader import GGUFFile
    from dense_prepare import DenseFrontend
    if (not configured(cfg) or cfg.get("backend") != "completed_v100" or
            cfg.get("variant") not in VARIANTS or cfg.get("source_target") != TARGET):
        raise ValueError("V100 Qwen requires its explicit backend, source target and variant")
    for key in ("exe", "model", "temp_dir"):
        path = Path(cfg.get(key, ""))
        if not path.is_absolute() or not path.exists():
            raise ValueError("V100 Qwen requires an existing absolute " + key)
    exe, model, temp = (Path(cfg[k]) for k in ("exe", "model", "temp_dir"))
    if not exe.is_file() or exe.stem != TARGET or not model.is_file() or not temp.is_dir():
        raise ValueError("V100 Qwen requires its newly built dedicated target and model/temp paths")
    from serve.temp_paths import require_task_temp_dir
    require_task_temp_dir(cfg["temp_dir"], "V100 Qwen")
    if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", cfg.get("gpu_uuid", "")):
        raise ValueError("V100 Qwen requires an explicit GPU UUID")
    if (any(type(cfg.get(key, default)) is not int for key, default in
            (("context", 262144), ("slots", 1), ("mtp", 0))) or
            cfg.get("context") != 262144 or cfg.get("slots", 1) != 1 or
            cfg.get("mtp", 0) not in (0, 1, 2) or cfg.get("kv_policy", "full") not in ("full", "ram8k-reuse")):
        raise ValueError("V100 Qwen requires the frozen 256K/single-slot/full-or-RAM profile")
    if cfg.get("dspark") or cfg.get("vision"):
        raise ValueError("this completed V100 Qwen factory exposes its frozen text service")
    stat = model.stat(); binding = cfg.get("asset_binding", {})
    if (binding.get("path") != str(model) or binding.get("bytes") != stat.st_size or
            binding.get("mtime_ns") != stat.st_mtime_ns):
        raise ValueError("V100 Qwen model identity changed")
    gguf = GGUFFile(model)
    if binding.get("header_sha256") != header_identity(model, gguf):
        raise ValueError("V100 Qwen actual GGUF header identity changed")
    if gguf.metadata.get("general.architecture") != "qwen35" or gguf.metadata.get("qwen35.context_length") != 262144:
        raise ValueError("V100 Qwen requires its actual Dense qwen35 256K metadata")
    tensors = {t.name: t for t in gguf.tensors}
    if cfg["variant"] == "qwen35_27b":
        if not any(t.type_id == 40 for t in gguf.tensors) or any("input_scale" in t.name for t in gguf.tensors):
            raise ValueError("Qwen3.5 V100 requires the frozen raw NVFP4 mixed inventory without input_scale")
    elif "output.weight" not in tensors or tensors["output.weight"].type_id != 14:
        raise ValueError("V100 original Q4K/Q6K Dense requires its bound Q6_K output head")
    frontend = DenseFrontend(model)
    if cfg.get("frontend_binding", {}).get("template_sha256") != frontend.template_hash:
        raise ValueError("V100 Qwen actual GGUF template identity changed")
    for key, value in cfg.get("env", {}).items():
        if key not in ENV or str(value) not in ("0", "1"):
            raise ValueError("unregistered V100 Qwen environment override: " + key)
    for policy in cfg.get("native_policies", []):
        if (not isinstance(policy, dict) or set(policy) != {"command", "value"} or
                policy["command"] not in POLICIES or
                not re.fullmatch(r"[A-Za-z0-9_]+", str(policy["value"]))):
            raise ValueError("V100 Qwen policy must be one registered idle command and value")
    directories = cfg.get("dll_dirs", [])
    if not isinstance(directories, list) or (os.name == "nt" and not directories):
        raise ValueError("this source-built CUDA engine requires explicit dll_dirs")
    for directory in directories:
        path = Path(directory)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("dll_dirs must contain existing absolute directories")
    timeout = cfg.setdefault("response_timeout_s", 900)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("V100 Qwen response_timeout_s must be finite and positive")
    maximum = cfg.setdefault("max_output", 65536)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum < cfg["context"]:
        raise ValueError("V100 Qwen max_output must fit its actual context")
    cfg.setdefault("kv_policy", "full"); cfg.setdefault("repeat_stop_tokens", 0)
    return frontend


class V100QwenEngine(NativeFamilyEngine):
    def __init__(self, cfg, frontend, environment, lazy=False):
        self.cfg, self.frontend, self.family = cfg, frontend, FAMILY
        self.max_context, self.batch, self.concurrent_slots = cfg["context"], 1, 1
        self.stops = set(frontend.stop_ids)
        self.environment = {k: v for k, v in environment.items()
            if not k.startswith(("STRATA_DENSE_", "STRATA_NVFP4_"))}
        self.environment.update({key: str(value) for key, value in cfg.get("env", {}).items()})
        self.environment.setdefault("STRATA_DENSE_SAMPLED_MTP", "0")
        self.environment.update(TEMP=cfg["temp_dir"], TMP=cfg["temp_dir"], TMPDIR=cfg["temp_dir"],
            CUDA_VISIBLE_DEVICES=cfg["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if cfg.get("dll_dirs"):
            self.environment["PATH"] = os.pathsep.join(cfg["dll_dirs"]+[self.environment.get("PATH", "")])
        self.info = dict(backend="completed_v100", family=FAMILY, source_target=TARGET,
            variant=cfg["variant"], gpu_uuid=cfg["gpu_uuid"], request_slots=1, batch_slots=1,
            mtp_requested=cfg.get("mtp", 0), capabilities=["text", "sampling", "cancel", "reset"])
        self._last, self.runner = {}, None
        self._gate = threading.RLock(); self._draining = None
        self.request_state = threading.local()
        self.progress = self.prompt_perf = self.request_phase = None
        self.frontend_facts = {}; self.unloaded = lazy
        if not lazy:
            self.restart()

    def restart(self):
        from serve.v100_qwen_engine import V100QwenRunner
        with self._gate:
            self.close()
            self.runner = V100QwenRunner(self.cfg, self.stops, self.environment)
            self.unloaded = False

    def _require_idle(self):
        if self._draining is not None and not self._draining.is_set():
            from serve.server import EngineStuck
            raise EngineStuck("V100 Qwen request is still draining; restart/unload is blocked")

    def close(self):
        with self._gate:
            self._require_idle()
            super().close()

    def unload(self):
        with self._gate:
            self.close(); self.unloaded = True

    def validate_request(self, req):
        if req.get("strata_prefix") is not None:
            raise ValueError("completed V100 Qwen prefix resume is not supported")

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        if embeddings:
            raise ValueError("completed V100 Qwen accepts text requests")
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
                    finished["last"] = dict(prompt_ms=stats["pp_ms"],
                        decode_ms=stats["total_ms"]-stats["pp_ms"], generated=stats["generated"],
                        tg_steps=max(0, stats["generated"]-1), prompt_tokens=len(ids), reused=0,
                        cache_slots=0, drafts_offered=stats["checked_proposals"],
                        drafts_accepted=stats["accepted_proposals"],
                        speculative_enabled=stats.get("native_speculative", False),
                        native_info=stats.get("native_info", {}))
                    native = stats.get("native_info", {})
                    self.info.update(mtp_effective=int(native.get("effective_mtp", native.get("actual_mtp", 0))),
                                     sampling=bool(int(native.get("sampling", 0))))
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
                            raise EngineDied("V100 Qwen engine exited during generation") from value
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
                    raise EngineDied("V100 Qwen cancel/RELEASE exceeded response_timeout_s; engine ended")
                worker.join()
                self._draining = None
                self.last = finished.get("last", {})
                self.request_phase = "idle"
