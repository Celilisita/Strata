#!/usr/bin/env python3
"""Manual text API for one resident native Strata dense engine and its GGUF template."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from dense_bench import BoundedLog, TITAN_UUID, WINDOWS_TASK, parse_json, runtime_environment
from dense_prepare import DenseFrontend
from serve.dense_engine import DenseEngine, DenseService, RequestCancelled
from serve.dense_vision import ResidentVision, VisualPrepare


class DenseHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


def make_handler(service, access_log):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, message, *args):
            access_log.write(f"{self.log_date_time_string()} {self.client_address[0]} {message % args}\n")

        def json_response(self, status, data):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/health":
                state = service.snapshot()
                state.update(model=service.model_name, max_output_tokens=service.max_output)
                self.json_response(200 if state["status"] == "ok" else 503, state)
            elif self.path == "/v1/models":
                self.json_response(200, {"object": "list", "data": [{"id": service.model_name,
                    "object": "model", "created": 0, "owned_by": "strata"}]})
            else:
                self.json_response(404, {"error": {"message": "unknown endpoint", "type": "invalid_request_error"}})

        def watch_disconnect(self, cancel, finished):
            while not finished.wait(0.1):
                try:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and not self.connection.recv(1, socket.MSG_PEEK):
                        cancel.set()
                        return
                except OSError:
                    cancel.set()
                    return

        def send_sse(self, data):
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, allow_nan=False)
            self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
            self.wfile.flush()

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.json_response(404, {"error": {"message": "unknown endpoint", "type": "invalid_request_error"}})
                return
            request_begin = time.perf_counter()
            cancel, finished, iterator, prepared = threading.Event(), threading.Event(), None, None
            slot_acquired, events_started = False, False
            stream_started = False
            self.close_connection = True
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("request body must be 1 byte..16 MiB")
                request = parse_json(self.rfile.read(length).decode("utf-8-sig"))
                if not isinstance(request.get("stream", False), bool):
                    raise ValueError("stream must be boolean")
                if request.get("stream_options") is not None and not isinstance(request["stream_options"], dict):
                    raise ValueError("stream_options must be an object")
                watcher = threading.Thread(target=self.watch_disconnect, args=(cancel, finished), daemon=True)
                watcher.start()
                service.acquire(cancel)
                slot_acquired = True
                prepared, tools, sampling = service.prepare(request, cancel, request_begin)
                if not service.engine.alive():
                    raise RuntimeError("owned dense engine is unavailable")
                completion_id, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
                streaming = request.get("stream", False)
                if streaming:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    stream_started = True
                iterator = service.events(prepared, tools, sampling, cancel)
                content, reasoning, calls, done, sent_role = [], [], [], None, False

                def chunk(delta, finish=None):
                    nonlocal sent_role
                    if not sent_role:
                        delta = dict(role="assistant", **delta)
                        sent_role = True
                    return {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                            "model": service.model_name, "choices": [{"index": 0, "delta": delta,
                            "finish_reason": finish}]}

                events_started = True
                for event in iterator:
                    if isinstance(event, dict):
                        done = event["done"]
                        continue
                    if event.kind == "reasoning":
                        reasoning.append(event.text)
                        delta = {"reasoning_content": event.text}
                    elif event.kind == "content":
                        content.append(event.text)
                        delta = {"content": event.text}
                    elif event.kind == "tool_call":
                        call = {"id": event.call.id, "type": "function", "function": {
                            "name": event.call.name, "arguments": json.dumps(event.call.arguments, ensure_ascii=False)}}
                        calls.append(call)
                        delta = {"tool_calls": [dict(index=len(calls)-1, **call)]}
                    else:
                        continue
                    if streaming and not cancel.is_set():
                        self.send_sse(chunk(delta))
                if cancel.is_set():
                    return
                if done is None:
                    raise RuntimeError("dense request ended without DONE")
                finish_reason = done["finish_reason"]
                if finish_reason == "cancelled":
                    finish_reason = "stop"
                if calls and finish_reason == "stop":
                    finish_reason = "tool_calls"
                prompt_tokens = prepared["prompt_tokens"]
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": done["output_tokens"],
                         "total_tokens": prompt_tokens+done["output_tokens"]}
                if streaming:
                    self.send_sse(chunk({}, finish_reason))
                    if (request.get("stream_options") or {}).get("include_usage"):
                        self.send_sse({"id": completion_id, "object": "chat.completion.chunk", "created": created,
                            "model": service.model_name, "choices": [], "usage": usage})
                    self.send_sse("[DONE]")
                else:
                    message = {"role": "assistant", "content": "".join(content)}
                    if reasoning:
                        message["reasoning_content"] = "".join(reasoning)
                    if calls:
                        message["tool_calls"] = calls
                    self.json_response(200, {"id": completion_id, "object": "chat.completion", "created": created,
                        "model": service.model_name, "choices": [{"index": 0, "message": message,
                        "finish_reason": finish_reason}], "usage": usage})
            except RequestCancelled:
                cancel.set()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                cancel.set()
            except Exception as error:
                cancel.set()
                body = {"error": {"message": str(error), "type": "invalid_request_error" if
                                  isinstance(error, (ValueError, KeyError, TypeError)) else "server_error"}}
                try:
                    if stream_started:
                        self.send_sse(body)
                        self.send_sse("[DONE]")
                    else:
                        self.json_response(400 if body["error"]["type"] == "invalid_request_error" else 503, body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
            finally:
                finished.set()
                try:
                    if iterator is not None:
                        iterator.close()  # drains DONE before reusing the slot or releasing DVI storage
                    if prepared is not None and not events_started and prepared.get("cleanup"):
                        prepared["cleanup"]()
                finally:
                    if slot_acquired:
                        service.release()
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--gguf", dest="model", type=pathlib.Path, required=True)
    parser.add_argument("--engine", type=pathlib.Path, required=True)
    parser.add_argument("--log-dir", type=pathlib.Path, required=True)
    parser.add_argument("--task-dir", type=pathlib.Path, help="allowed root for real visual DVI artifacts; default log-dir")
    parser.add_argument("--vision-engine", type=pathlib.Path, help="resident strata-dense-vision executable")
    parser.add_argument("--mmproj", type=pathlib.Path, help="matching Qwen3.5-0.8B projector GGUF")
    parser.add_argument("--vision-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vision-threads", type=int)
    parser.add_argument("--vision-max-tokens", type=int, help="explicit per-image limit; default model preprocessing")
    parser.add_argument("--vision-keep-artifacts", action="store_true", help="keep task-owned inputs/SDV/DVI for diagnosis")
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--budget-mib", type=int, default=12000)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--pp-half-cache", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device-greedy", type=int, choices=(0, 1), default=0)
    parser.add_argument("--projection-fusion", type=int, choices=(0, 1), default=0)
    parser.add_argument("--compact-async", type=int, choices=(0, 1), default=0)
    parser.add_argument("--norm-fusion", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-output", type=int, default=65536)
    parser.add_argument("--model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=7200)
    args = parser.parse_args()
    for key in ("model", "engine", "log_dir"):
        setattr(args, key, getattr(args, key).resolve())
    for key in ("vision_engine", "mmproj"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if bool(args.vision_engine) != bool(args.mmproj):
        parser.error("vision-engine and mmproj must be configured together")
    if any(value is not None and not 1 <= value <= 1000000 for value in (args.vision_threads, args.vision_max_tokens)):
        parser.error("vision threads/token limit must be in 1..1000000 when explicitly set")
    args.task_dir = (args.task_dir or args.log_dir).resolve()
    if os.name == "nt":
        if WINDOWS_TASK is None:
            parser.error("STRATA_DENSE_TASK_ROOT is required on Windows")
        if any(not path.is_relative_to(WINDOWS_TASK.resolve()) for path in (args.log_dir, args.task_dir)):
            parser.error(f"log-dir/task-dir must be inside {WINDOWS_TASK}")
    if args.context < 2 or not 1 <= args.budget_mib <= 12000 or not 1 <= args.chunk <= 1024:
        parser.error("invalid context, total budget (1..12000 MiB), or chunk (1..1024)")
    if args.max_output < 1 or not 1 <= args.port <= 65535 or min(args.startup_timeout, args.request_timeout) <= 0:
        parser.error("output limit, port and timeouts must be positive")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.log_dir / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    os.chdir(args.log_dir)
    environment = runtime_environment()
    frontend = DenseFrontend(args.model)
    if args.vision_engine and (frontend.md.get("general.architecture") != "qwen35" or
                              frontend.md.get("qwen35.embedding_length") != 1024):
        parser.error("configured vision requires the real qwen35 model with hidden width 1024")
    engine, encoder, visual = None, None, None
    access_log = BoundedLog(args.log_dir / "http.access.log")
    # This is the real service bind, performed before model load. An occupied
    # port fails directly; no process discovery or port probing is performed.
    server = None
    try:
        server = DenseHTTPServer((args.host, args.port), make_handler(None, access_log))
        engine = DenseEngine(args.engine, args.model, args.context, args.budget_mib, args.chunk,
            args.log_dir, environment, args.startup_timeout, args.request_timeout, args.pp_half_cache,
            args.task_dir, device_greedy=args.device_greedy, projection_fusion=args.projection_fusion,
            compact_async=args.compact_async, norm_fusion=args.norm_fusion)
        if args.vision_engine:
            if "visual" not in engine.capabilities:
                raise ValueError("this engine/model does not declare native visual input support")
            encoder = ResidentVision(args.vision_engine, args.model, args.mmproj, args.task_dir,
                args.log_dir/"vision", environment, args.context, args.vision_gpu, args.vision_threads,
                args.vision_max_tokens, args.startup_timeout, args.request_timeout)
            visual = VisualPrepare(encoder, args.max_output, args.vision_keep_artifacts)
        service = DenseService(engine, frontend, args.model_name or args.model.stem, args.max_output, visual)
        server.RequestHandlerClass = make_handler(service, access_log)
        print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "model": service.model_name,
                          "gpu_uuid": TITAN_UUID, "context": args.context,
                          "device_greedy": args.device_greedy,
                          "projection_fusion": args.projection_fusion,
                          "compact_async": args.compact_async, "norm_fusion": args.norm_fusion,
                          "max_output_tokens": args.max_output, "images": visual is not None,
                          "vision": visual.snapshot() if visual else None,
                          "engine_capabilities": sorted(engine.capabilities)}), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if engine is not None:
            engine.close()
        if visual is not None:
            visual.close()
        elif encoder is not None:
            encoder.close()
        access_log.close()


if __name__ == "__main__":
    main()
