"""Run independent dense-engine cases in one process and save actual work/costs.

The transport is the engine's native stdin/stdout protocol, not HTTP. EOS and
per-case output limits are preserved; quality is left pending manual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import queue
import subprocess
import threading
import time

from dense_prepare import DenseFrontend

# Resolve explicit instance configuration before any CUDA backend loads.
# The compatibility alias is used by the accepted protocol helpers.
TITAN_UUID = os.environ.get("STRATA_DENSE_GPU_UUID", "")
if TITAN_UUID:
    _uuid_parts = TITAN_UUID[4:].split("-")
    if (not TITAN_UUID.startswith("GPU-") or list(map(len, _uuid_parts)) != [8, 4, 4, 4, 12] or
            any(c not in "0123456789abcdefABCDEF" for part in _uuid_parts for c in part)):
        raise ValueError("STRATA_DENSE_GPU_UUID must name one complete GPU UUID")
# The Windows service writes task artifacts only below an explicitly configured
# root. A missing root does not fail module import; service startup checks it.
_task_root = os.environ.get("STRATA_DENSE_TASK_ROOT")
WINDOWS_TASK = pathlib.Path(_task_root) if _task_root else None
if _task_root and not (pathlib.Path(_task_root).is_absolute() or
                       pathlib.PureWindowsPath(_task_root).is_absolute()):
    raise ValueError("STRATA_DENSE_TASK_ROOT must be an absolute task directory")


def runtime_environment():
    environment = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if TITAN_UUID:
        environment["CUDA_VISIBLE_DEVICES"] = TITAN_UUID
    return environment


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def file_identity(path):
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def parse_json(text):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant {value}")
    result = json.loads(text, parse_constant=invalid)
    if not isinstance(result, dict) or not finite(result):
        raise ValueError("engine metadata must be a finite JSON object")
    return result


class BoundedLog:
    def __init__(self, path, limit=32 * 1024 * 1024):
        self.file = path.open("wb")
        self.limit = limit
        self.written = 0
        self.truncated = False
        self.lock = threading.Lock()

    def write(self, text):
        raw = text.encode("utf-8", errors="replace")
        with self.lock:
            room = max(0, self.limit - self.written)
            if room:
                self.file.write(raw[:room])
                self.written += min(room, len(raw))
                self.file.flush()
            if len(raw) > room and not self.truncated:
                self.file.write(b"\n[log byte limit reached; further detail omitted]\n")
                self.file.flush()
                self.truncated = True

    def close(self):
        with self.lock:
            self.file.close()


class EngineProcess:
    def __init__(self, command, cwd, environment):
        self.stdout_log = BoundedLog(cwd / "engine.stdout.log")
        self.stderr_log = BoundedLog(cwd / "engine.stderr.log")
        self.lines = queue.Queue(maxsize=16384)
        self.reader_error = None
        self.started = time.perf_counter()
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=cwd, env=environment, text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except BaseException:
            self.stdout_log.close()
            self.stderr_log.close()
            raise
        self.readers = [threading.Thread(target=self._pump, args=(self.process.stdout, True), daemon=True),
                        threading.Thread(target=self._pump, args=(self.process.stderr, False), daemon=True)]
        for thread in self.readers:
            thread.start()

    def _pump(self, stream, protocol):
        log = self.stdout_log if protocol else self.stderr_log
        try:
            while True:
                line = stream.readline(65537)
                arrived = time.perf_counter()
                if not line:
                    break
                log.write(line)
                if protocol:
                    if len(line) > 65536:
                        raise RuntimeError("engine stdout line exceeds 65536 characters")
                    self.lines.put((arrived, line.rstrip("\r\n")), timeout=5)
        except Exception as error:
            self.reader_error = f"engine log reader failed: {error}"
        finally:
            stream.close()
            if protocol:
                try:
                    self.lines.put((time.perf_counter(), None), timeout=1)
                except queue.Full:
                    pass

    def next(self, deadline):
        while True:
            if self.reader_error:
                raise RuntimeError(self.reader_error)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("engine protocol deadline exceeded")
            try:
                stamp, line = self.lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError(f"engine stdout closed, exit={self.process.poll()}; see stderr log")
            if line:
                return stamp, line

    def send(self, line):
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited before request, exit={self.process.returncode}")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def close(self, graceful):
        if self.process.poll() is None and graceful:
            try:
                self.send("QUIT")
                self.process.wait(timeout=15)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdin.close()
        for thread in self.readers:
            thread.join(timeout=2)
        self.stdout_log.close()
        self.stderr_log.close()
        return self.process.returncode


def load_cases(path, frontend, context, names):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    cases = data.get("cases", [data]) if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases JSON must contain a nonempty case list or one request object")
    selected, seen = [], set()
    for index, case in enumerate(cases):
        name = str(case.get("name") or f"case_{index + 1}")
        seen.add(name)
        if names and name not in names:
            continue
        ids, stops = case.get("input_ids"), case.get("stop_ids")
        for field, values in (("input_ids", ids), ("stop_ids", stops)):
            if not isinstance(values, list) or not values or any(
                    type(token) is not int or not 0 <= token < len(frontend.tokenizer.tokens) for token in values):
                raise ValueError(f"{name}: {field} must contain valid vocabulary IDs")
        if not set(frontend.stop_ids).issubset(stops) or len(stops) > 32:
            raise ValueError(f"{name}: stop_ids must retain all actual EOG tokens and contain at most 32 IDs")
        limit = case.get("max_new_tokens")
        if type(limit) is not int or limit < 1 or len(ids) + limit > context:
            raise ValueError(f"{name}: full prompt and unchanged output limit exceed --context")
        if case.get("prompt_tokens", len(ids)) != len(ids):
            raise ValueError(f"{name}: prompt_tokens disagrees with the actual input IDs")
        if case.get("model") and pathlib.Path(case["model"]).resolve() != frontend.path.resolve():
            raise ValueError(f"{name}: prepared model identity differs from --model")
        metadata = case.get("metadata", {})
        if metadata.get("requires_visual_expansion"):
            raise ValueError(f"{name}: image placeholders still require actual visual expansion")
        if metadata.get("template_sha256", frontend.template_hash) != frontend.template_hash:
            raise ValueError(f"{name}: prepared template identity differs from actual GGUF")
        selected.append(dict(case, name=name))
    if names - seen:
        raise ValueError(f"unknown --case-names: {sorted(names - seen)}")
    if not selected:
        raise ValueError("case filter selected no requests")
    return selected


def finish_case(record, done, frontend, stops, limit, first_stamp, end_stamp, begin):
    ids = record["output_ids"]
    if done.get("input_tokens") != record["input_tokens"] or done.get("output_tokens") != len(ids):
        raise ValueError("DONE token counts disagree with the actual transmitted/received work")
    reason = done.get("finish_reason")
    if reason == "stop":
        if not ids or ids[-1] not in stops:
            raise ValueError("DONE says stop without a terminal stop token")
    elif reason != "length" or len(ids) != limit:
        raise ValueError("DONE has an invalid finish reason or output length")
    if any(token in stops for token in ids[:-1]):
        raise ValueError("engine continued after an EOG token")
    content_ids = ids[:-1] if ids and ids[-1] in stops else ids
    try:
        text = frontend.tokenizer.decode(content_ids, errors="strict")
        complete = True
    except UnicodeDecodeError:
        text, complete = frontend.tokenizer.decode(content_ids), False
    pp_ms, tg_ms = float(done["pp_ms"]), float(done["tg_ms"])
    if pp_ms < 0 or tg_ms < 0:
        raise ValueError("negative engine duration")
    record.update(status="complete", output_tokens=len(ids), content_output_tokens=len(content_ids),
                  text=text, text_utf8_complete=complete, finish_reason=reason, engine_cost=done,
                  host_ttft_ms=(first_stamp - begin) * 1000 if first_stamp is not None else None,
                  host_request_ms=(end_stamp - begin) * 1000,
                  engine_pp_tokens_s=record["input_tokens"] * 1000 / pp_ms if pp_ms else None,
                  engine_tg_tokens_s=max(0, len(ids) - 1) * 1000 / tg_ms if tg_ms else None,
                  tg_decode_steps=max(0, len(ids) - 1), finite=True,
                  finite_scope="finite engine costs and valid token IDs; engine rejects nonfinite logits",
                  quality_status="pending_manual_review")


def run_case(engine, case, frontend, timeout, records, save):
    ids, stops, limit = case["input_ids"], case["stop_ids"], case["max_new_tokens"]
    record = {"name": case["name"], "status": "running", "input_tokens": len(ids),
              "input_ids_sha256": hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest(),
              "max_new_tokens": limit, "stop_ids": stops, "messages": case.get("messages"),
              "workload": case.get("workload"), "output_ids": [], "quality_status": "pending_manual_review"}
    records.append(record)
    save()
    line = "RUN {} {} {} {} {}".format(limit, len(ids), len(stops),
                                         " ".join(map(str, ids)), " ".join(map(str, stops)))
    begin, first = time.perf_counter(), None
    try:
        engine.send(line)
        while True:
            stamp, answer = engine.next(begin + timeout)
            if answer.startswith("TOKEN "):
                token = int(answer[6:])
                if not 0 <= token < len(frontend.tokenizer.tokens) or len(record["output_ids"]) >= limit:
                    raise ValueError("engine emitted invalid/excess token IDs")
                record["output_ids"].append(token)
                if first is None:
                    first = stamp
            elif answer.startswith("DONE "):
                finish_case(record, parse_json(answer[5:]), frontend, stops, limit, first, stamp, begin)
                save()
                return record
            elif answer.startswith("ERROR"):
                raise RuntimeError(answer)
            else:
                raise ValueError(f"unexpected engine protocol line: {answer[:256]}")
    except BaseException as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}", finite=False,
                      host_request_ms=(time.perf_counter() - begin) * 1000,
                      output_tokens=len(record["output_ids"]))
        record["text"] = frontend.tokenizer.decode(record["output_ids"])
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("engine", "model", "cases", "output", "log-dir"):
        parser.add_argument("--" + option, type=pathlib.Path, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--budget-mib", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--pp-half-cache", type=int, choices=(0,1), default=0)
    parser.add_argument("--device-greedy", type=int, choices=(0,1), default=0)
    parser.add_argument("--projection-fusion", type=int, choices=(0,1), default=0)
    parser.add_argument("--compact-async", type=int, choices=(0,1), default=0)
    parser.add_argument("--norm-fusion", type=int, choices=(0,1), default=0)
    parser.add_argument("--case-names", help="comma-separated exact case names; output limits are retained")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=7200)
    args = parser.parse_args()
    for key in ("engine", "model", "cases", "output", "log_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.context < 1 or not 1 <= args.budget_mib <= 12000 or not 1 <= args.chunk <= 1024:
        raise ValueError("invalid context, budget (1..12000 MiB), or chunk (1..1024)")
    if args.startup_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if os.name == "nt":
        if WINDOWS_TASK is None:
            raise ValueError("STRATA_DENSE_TASK_ROOT is required on Windows")
        root = WINDOWS_TASK.resolve()
        for path in (args.log_dir, args.output):
            if not path.is_relative_to(root):
                raise ValueError(f"task output/log path must be inside {root}: {path}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.log_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    environment = runtime_environment()
    environment.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    prep_begin = time.perf_counter()
    frontend = DenseFrontend(args.model)
    cases = load_cases(args.cases, frontend, args.context,
                       set(filter(None, (args.case_names or "").split(","))))
    command = [str(args.engine), "--model", str(args.model), "--context", str(args.context),
               "--budget-mib", str(args.budget_mib), "--chunk", str(args.chunk),
               "--pp-half-cache", str(args.pp_half_cache)]
    if args.device_greedy:
        command += ["--device-greedy", str(args.device_greedy)]
    if args.projection_fusion:
        command += ["--projection-fusion", str(args.projection_fusion)]
    if args.compact_async:
        command += ["--compact-async", str(args.compact_async)]
    if args.norm_fusion:
        command += ["--norm-fusion", str(args.norm_fusion)]
    result = {"status": "starting", "transport": "native_engine_stdio", "http_measured": False,
              "gpu_uuid": TITAN_UUID, "command": command, "engine": file_identity(args.engine),
              "model": file_identity(args.model), "cases_file": file_identity(args.cases),
              "frontend": frontend.identity, "context": args.context, "budget_mib": args.budget_mib,
              "chunk": args.chunk, "device_greedy": args.device_greedy,
              "projection_fusion": args.projection_fusion,
              "compact_async": args.compact_async, "norm_fusion": args.norm_fusion,
              "frontend_ms": (time.perf_counter() - prep_begin) * 1000,
              "model_load_count": 0, "case_names": [case["name"] for case in cases], "cases": []}
    save = lambda: atomic_json(args.output, result)
    save()
    engine, exit_status = None, 1
    try:
        engine = EngineProcess(command, args.log_dir, environment)
        stamp, answer = engine.next(engine.started + args.startup_timeout)
        if not answer.startswith("READY "):
            raise ValueError(f"expected READY, received {answer[:256]}")
        result.update(ready=parse_json(answer[6:]), host_process_ready_ms=(stamp-engine.started)*1000,
                      model_load_count=1, status="running")
        save()
        print(json.dumps({"status": "ready", "host_process_ready_ms": result["host_process_ready_ms"]}), flush=True)
        for case in cases:
            record = run_case(engine, case, frontend, args.request_timeout, result["cases"], save)
            print(json.dumps({k: record[k] for k in ("name", "input_tokens", "output_tokens", "finish_reason",
                              "engine_pp_tokens_s", "engine_tg_tokens_s", "host_ttft_ms", "host_request_ms")}), flush=True)
        result["status"], exit_status = "complete", 0
    except BaseException as error:
        result.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                      error=f"{type(error).__name__}: {error}")
    finally:
        if engine is not None:
            result["engine_exit_code"] = engine.close(exit_status == 0)
            result["logs"] = {"stdout": str(args.log_dir / "engine.stdout.log"),
                              "stderr": str(args.log_dir / "engine.stderr.log"),
                              "stdout_truncated": engine.stdout_log.truncated,
                              "stderr_truncated": engine.stderr_log.truncated}
            if exit_status == 0 and result["engine_exit_code"] != 0:
                result.update(status="error", error="engine exited unsuccessfully after QUIT")
                exit_status = 1
        result["summary"] = {"completed_cases": sum(c["status"] == "complete" for c in result["cases"]),
                             "planned_cases": len(cases), "finite": all(
                                 c.get("finite", False) for c in result["cases"]) and exit_status == 0,
                             "quality_status": "pending_manual_review"}
        save()
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "summary": result["summary"]}), flush=True)
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
