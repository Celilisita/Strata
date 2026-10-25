"""Resident single-slot boundary and model-template frontend for native dense models."""
from __future__ import annotations

import codecs
import inspect
import math
import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dense_bench import EngineProcess, TITAN_UUID, parse_json
from dense_prepare import DenseFrontend
from strata_tokenizer import UNICODE_TO_BYTE
from serve.frontend import OutputParser, openai_to_messages

_OUTPUT_FINISH_HAS_REASON = "reason" in inspect.signature(OutputParser.finish).parameters


class RequestCancelled(RuntimeError):
    pass


class DenseEngine:
    """One owned process; PARAM/RUN/DONE are serialized for the entire request."""
    def __init__(self, executable, model, context, budget_mib, chunk, log_dir, environment,
                 startup_timeout=1800, request_timeout=7200, pp_half_cache=0, task_dir=None,
                 device_greedy=0, projection_fusion=0, compact_async=0, norm_fusion=0):
        self.max_context, self.request_timeout = context, request_timeout
        self.task_dir = pathlib.Path(task_dir or log_dir).resolve()
        self.lock, self.closed = threading.Lock(), threading.Event()
        self.active, self.last = None, {}
        command = [str(executable), "--model", str(model), "--context", str(context),
                   "--budget-mib", str(budget_mib), "--chunk", str(chunk)]
        if self.task_dir != pathlib.Path(log_dir).resolve():
            command += ["--task-dir", str(self.task_dir)]
        if pp_half_cache:
            command += ["--pp-half-cache", str(pp_half_cache)]
        if device_greedy:
            command += ["--device-greedy", str(device_greedy)]
        if projection_fusion:
            command += ["--projection-fusion", str(projection_fusion)]
        if compact_async:
            command += ["--compact-async", str(compact_async)]
        if norm_fusion:
            command += ["--norm-fusion", str(norm_fusion)]
        environment = dict(environment, STRATA_DENSE_PP_PROGRESS="1")
        self.worker = EngineProcess(command, pathlib.Path(log_dir), environment)
        try:
            _, line = self.worker.next(self.worker.started + startup_timeout)
            if not line.startswith("READY "):
                raise RuntimeError(f"expected dense READY, got {line[:256]}")
            self.ready = parse_json(line[6:])
        except BaseException:
            self.worker.close(False)
            raise
        self.capabilities = set(self.ready.get("capabilities", []))
        self.used_sampling = False

    def alive(self):
        return not self.closed.is_set() and self.worker.process.poll() is None

    def snapshot(self):
        return {"status": "ok" if self.alive() else "unavailable", "busy": self.active is not None,
                "gpu_uuid": TITAN_UUID, "context": self.max_context, "ready": self.ready,
                "sampling": "sampling" in self.capabilities or self.used_sampling,
                "async_cancel": "cancel" in self.capabilities}

    def _next(self, deadline, cancel=None):
        sent_cancel = False
        while time.perf_counter() < deadline:
            if cancel is not None and cancel.is_set() and not sent_cancel and "cancel" in self.capabilities:
                self.worker.send("CANCEL")
                sent_cancel = True
            try:
                return self.worker.next(min(deadline, time.perf_counter() + 0.1))
            except TimeoutError:
                continue
        raise TimeoutError("dense request deadline exceeded")

    def _params(self, sampling, deadline):
        greedy = (sampling["temperature"] == 0 and sampling["repeat_penalty"] == 1)
        if greedy and not self.used_sampling:
            return
        self.worker.send("PARAM {temperature:.9g} {top_k} {top_p:.9g} {repeat_penalty:.9g} "
                         "{penalty_last_n} {seed}".format(**sampling))
        _, answer = self._next(deadline)
        if answer != "PARAM":
            if answer.startswith("ERROR"):
                raise ValueError("this dense engine build does not support requested sampling: " + answer[:256])
            raise RuntimeError(f"invalid PARAM acknowledgement: {answer[:256]}")
        self.used_sampling = True

    def generate(self, prepared, sampling, cancel):
        while not self.lock.acquire(timeout=0.1):
            if cancel.is_set() or not self.alive():
                return
        completed, started = False, False
        output_count, deadline = 0, time.perf_counter() + self.request_timeout
        self.active = cancel
        try:
            if not self.alive():
                raise RuntimeError("owned dense engine is unavailable")
            if cancel.is_set():
                return
            self._params(sampling, deadline)
            # A future visual preparation callback may return a before_run
            # binder. It executes under this same slot lock before RUN reset.
            binder = prepared.get("before_run")
            if binder is not None:
                binder(self.worker)
            stops, maximum = prepared["stop_ids"], prepared["max_new_tokens"]
            if prepared.get("dvi_path"):
                if "visual" not in self.capabilities:
                    raise ValueError("this dense engine/model does not support visual RUNV requests")
                dvi = pathlib.Path(prepared["dvi_path"]).resolve()
                if not dvi.is_relative_to(self.task_dir) or not dvi.is_file() or any(ch in str(dvi) for ch in "\r\n"):
                    raise ValueError("DVI artifact must be a file inside the configured task directory")
                line = "RUNV {} {} {} {}".format(maximum, len(stops), " ".join(map(str, stops)), dvi)
            else:
                ids = prepared["input_ids"]
                line = "RUN {} {} {} {} {}".format(maximum, len(ids), len(stops),
                    " ".join(map(str, ids)), " ".join(map(str, stops)))
            self.worker.send(line)
            started = True
            cancel_sent = False
            while True:
                # Send CANCEL once for this RUN, then continue draining through
                # DONE. Old binaries without a capability drain normally.
                if cancel.is_set() and not cancel_sent and "cancel" in self.capabilities:
                    self.worker.send("CANCEL")
                    cancel_sent = True
                try:
                    _, line = self._next(min(deadline, time.perf_counter() + 0.2))
                except TimeoutError:
                    if time.perf_counter() < deadline:
                        continue
                    raise
                if line.startswith("PP "):
                    fields = line.split()
                    if len(fields) != 5:
                        raise RuntimeError("invalid dense prefill progress line")
                    yield {"prefill": (int(fields[1]), int(fields[2]), float(fields[3]))}
                elif line.startswith("TOKEN "):
                    token = int(line[6:])
                    output_count += 1
                    if output_count > maximum:
                        raise RuntimeError("dense engine exceeded the requested output limit")
                    if not cancel.is_set():
                        yield {"token": token}
                elif line.startswith("DONE "):
                    done = parse_json(line[5:])
                    if done.get("output_tokens") != output_count:
                        raise RuntimeError("dense DONE disagrees with observed output token count")
                    if done.get("finish_reason") not in ("stop", "length", "cancelled"):
                        raise RuntimeError("dense engine returned an unknown finish reason")
                    planned = prepared.get("prompt_tokens", len(prepared.get("input_ids", [])))
                    if done.get("planned_input_tokens", planned) != planned or \
                            (done["finish_reason"] != "cancelled" and done.get("input_tokens") != planned):
                        raise RuntimeError("dense DONE disagrees with the complete prepared input workload")
                    self.last, completed = done, True
                    yield {"done": done}
                    break
                elif line.startswith("ERROR"):
                    completed = True  # engine catches errors and resets before this line
                    raise RuntimeError(line)
                else:
                    raise RuntimeError(f"unexpected dense output: {line[:256]}")
        finally:
            if started and not completed:
                cancel.set()
                try:
                    if "cancel" in self.capabilities:
                        self.worker.send("CANCEL")
                    while time.perf_counter() < deadline:
                        _, line = self._next(deadline)
                        if line.startswith("PP "):
                            continue  # drain cancelled request progress before DONE
                        if line.startswith("TOKEN "):
                            output_count += 1
                        elif line.startswith("DONE "):
                            done = parse_json(line[5:])
                            if done.get("output_tokens") != output_count:
                                raise RuntimeError("cancel-drain DONE output count differs from actual emitted tokens")
                            self.last = done
                            completed = True
                            break
                        elif line.startswith("ERROR"):
                            completed = True
                            break
                except Exception:
                    pass
                if not completed:
                    self.closed.set()
                    self.worker.close(False)
            self.active = None
            self.lock.release()

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        if self.active is not None:
            self.active.set()
        acquired = self.lock.acquire(timeout=15)
        try:
            self.worker.close(acquired)
        finally:
            if acquired:
                self.lock.release()


class ModelOutput:
    """Choose reasoning from an actual prompt/opening tag, then use the shared parser."""
    def __init__(self, prompt, tools):
        self.thinking = prompt.rfind("<think>") > prompt.rfind("</think>")
        self.tools, self.parser, self.pending = tools, None, ""

    def feed(self, text, final=False):
        if self.parser is not None:
            return self.parser.feed(text)
        self.pending += text
        probe = self.pending.lstrip()
        opening = "<think>"
        if not final and (not probe or opening.startswith(probe)):
            return []
        if probe.startswith(opening):
            self.thinking = True
            self.pending = probe[len(opening):]
        self.parser = OutputParser(thinking=self.thinking, tools=self.tools)
        pending, self.pending = self.pending, ""
        return self.parser.feed(pending)

    def finish(self, reason=None):
        events = self.feed("", final=True)
        tail = self.parser.finish(reason=reason) if _OUTPUT_FINISH_HAS_REASON else self.parser.finish()
        return events + tail


def sampling_of(request, vocabulary, context):
    defaults = {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repeat_penalty": 1.0,
                "penalty_last_n": 64, "seed": 1}
    aliases = dict(request)
    if "repeat_last_n" in aliases and "penalty_last_n" not in aliases:
        aliases["penalty_last_n"] = aliases["repeat_last_n"]
    result = {key: aliases.get(key, default) for key, default in defaults.items()}
    for key in ("temperature", "top_p", "repeat_penalty"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= result["temperature"] <= 2 or not 0 < result["top_p"] <= 1 or result["repeat_penalty"] <= 0:
        raise ValueError("temperature must be 0..2, top_p in (0,1], repeat_penalty positive")
    for key, maximum in (("top_k", vocabulary), ("penalty_last_n", context), ("seed", 4294967295)):
        if type(result[key]) is not int or not 0 <= result[key] <= maximum:
            raise ValueError(f"{key} must be an integer in 0..{maximum}")
    for key in ("presence_penalty", "frequency_penalty"):
        if request.get(key, 0) != 0:
            raise ValueError(f"{key} is not configured by this dense sampler")
    return result


class DenseService:
    def __init__(self, engine, frontend: DenseFrontend, model_name, max_output=65536, visual_prepare=None):
        self.engine, self.frontend, self.model_name = engine, frontend, model_name
        self.max_output, self.visual_prepare = max_output, visual_prepare
        self.request_lock, self.phase, self.last_request = threading.Lock(), "idle", None

    def acquire(self, cancel):
        while not self.request_lock.acquire(timeout=0.1):
            if cancel.is_set():
                raise RequestCancelled("cancelled while waiting for the model slot")
            if not self.engine.alive():
                raise RuntimeError("owned dense engine is unavailable")
        self.phase = "preparing"

    def release(self):
        self.phase = "idle"
        self.request_lock.release()

    def snapshot(self):
        vision = self.visual_prepare.snapshot() if self.visual_prepare is not None else None
        return dict(self.engine.snapshot(), busy=self.request_lock.locked(), phase=self.phase,
                    vision=vision, images=vision is not None and vision["status"] == "ok",
                    last_request=self.last_request)

    def prepare(self, request, cancel=None, request_started=None):
        begin = time.perf_counter()
        cancel = cancel or threading.Event()
        if cancel.is_set():
            raise RequestCancelled("cancelled before request preparation")
        if request.get("model", self.model_name) != self.model_name:
            raise ValueError("requested model is not loaded in this service")
        if type(request.get("n", 1)) is not int or request.get("n", 1) != 1:
            raise ValueError("this single-slot service supports n=1")
        raw_messages = request.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a nonempty list")
        has_images = any(isinstance(m, dict) and isinstance(m.get("content"), list) and any(
            isinstance(part, dict) and part.get("type") in ("image", "input_image", "image_url")
            for part in m["content"]) for m in raw_messages)
        if has_images and self.visual_prepare is None:
            raise ValueError("images are not configured for this dense service")
        messages, tools, kwargs = openai_to_messages(request)
        template_kwargs = request.get("chat_template_kwargs", {})
        if not isinstance(template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be an object")
        if "enable_thinking" in template_kwargs and type(template_kwargs["enable_thinking"]) is not bool:
            raise ValueError("chat_template_kwargs.enable_thinking must be boolean")
        if "enable_thinking" in request:
            if type(request["enable_thinking"]) is not bool:
                raise ValueError("enable_thinking must be boolean")
            if "enable_thinking" in template_kwargs and request["enable_thinking"] != template_kwargs["enable_thinking"]:
                raise ValueError("enable_thinking conflicts with chat_template_kwargs.enable_thinking")
        kwargs.update(template_kwargs)
        if "enable_thinking" in request:
            kwargs["enable_thinking"] = request["enable_thinking"]
        limits = [request[key] for key in ("max_completion_tokens", "max_tokens") if key in request]
        if any(type(value) is not int or value < 1 for value in limits) or len(set(limits)) > 1:
            raise ValueError("output limits must be positive integers and agree when both are supplied")
        maximum = limits[0] if limits else None
        if maximum is not None and maximum > self.max_output:
            raise ValueError(f"requested output exceeds service limit {self.max_output}")
        sampling = sampling_of(request, len(self.frontend.tokenizer.tokens), self.engine.max_context)
        if has_images:
            self.phase = "image_encoding"
            prepared = self.visual_prepare(self.frontend, messages, tools, kwargs, maximum,
                                           self.engine.max_context, cancel=cancel)
            try:
                if prepared.get("metadata", {}).get("requires_visual_expansion", True) or not prepared.get("dvi_path"):
                    raise ValueError("visual callback did not prepare complete image inputs")
                count, limit = prepared.get("prompt_tokens"), prepared.get("max_new_tokens")
                if type(count) is not int or type(limit) is not int or count < 1 or limit < 1 or \
                        limit > self.max_output or count+limit > self.engine.max_context:
                    raise ValueError("visual callback input/output workload exceeds the configured context/limits")
            except BaseException:
                if prepared.get("cleanup"):
                    prepared["cleanup"]()
                raise
        else:
            prepared = self.frontend.prepare(messages, tools=tools, template_kwargs=kwargs,
                                              max_new=1, context=self.engine.max_context)
            room = self.engine.max_context - len(prepared["input_ids"])
            maximum = maximum if maximum is not None else min(self.max_output, room)
            if maximum > room:
                raise ValueError("complete prompt plus output limit exceeds configured context")
            prepared["max_new_tokens"] = maximum
        prepared.update(_api_request_start=request_started or begin,
                        _api_prepare_ms=(time.perf_counter()-begin)*1000)
        prompt = prepared["rendered_prompt"]
        if "enable_thinking" in kwargs:
            thinking_mode = "explicit_enabled" if kwargs["enable_thinking"] else "explicit_disabled"
        elif prompt.rfind("<think>") >= 0 and prompt.rfind("</think>") > prompt.rfind("<think>") and \
                not prompt[prompt.rfind("</think>")+8:].strip():
            thinking_mode = "model_default_non_thinking"
        elif prompt.rfind("<think>") > prompt.rfind("</think>"):
            thinking_mode = "model_default_thinking"
        else:
            thinking_mode = "model_default_unspecified"
        prepared["metadata"]["thinking_mode"] = thinking_mode
        return prepared, tools, sampling

    def record_done(self, done, prepared, first):
        result = dict(done)
        started = prepared.get("_api_request_start", time.perf_counter())
        result.update(api_ttft_ms=(first-started)*1000 if first is not None else None,
                      api_pipeline_ms=(time.perf_counter()-started)*1000,
                      api_prepare_ms=prepared.get("_api_prepare_ms"), vision_cost=prepared.get("vision_cost"),
                      thinking_mode=prepared["metadata"].get("thinking_mode"),
                      api_timing_scope="handler entry through raw decoder first token / DONE")
        self.last_request = result
        return result

    def events(self, prepared, tools, sampling, cancel):
        tokenizer = self.frontend.tokenizer
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        parser = ModelOutput(prepared["rendered_prompt"], tools)
        iterator = self.engine.generate(prepared, sampling, cancel)
        first = None
        previous_done, recorded = self.engine.last, False
        self.phase = "decoding"
        try:
            for item in iterator:
                if "token" in item:
                    if first is None:
                        first = time.perf_counter()
                    token = item["token"]
                    if not 0 <= token < len(tokenizer.tokens):
                        raise RuntimeError("engine emitted a token outside the actual vocabulary")
                    if token in prepared["stop_ids"]:
                        continue
                    raw = bytes(UNICODE_TO_BYTE[ch] for ch in tokenizer.tokens[token])
                    for event in parser.feed(decoder.decode(raw)):
                        yield event
                else:
                    for event in parser.feed(decoder.decode(b"", final=True)) + parser.finish(item["done"].get("finish_reason")):
                        yield event
                    done = self.record_done(item["done"], prepared, first)
                    recorded = True
                    yield {"done": done}
        finally:
            iterator.close()  # cancellation drains DONE before releasing the slot
            if not recorded and self.engine.last is not previous_done:
                self.record_done(self.engine.last, prepared, first)
            if prepared.get("cleanup"):
                try:
                    prepared["cleanup"]()
                except OSError as error:
                    if self.last_request is not None:
                        self.last_request["cleanup_error"] = str(error)
