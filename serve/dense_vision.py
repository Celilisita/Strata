"""Resident real mtmd ENC/SDV1 bridge and task-owned image request preparation."""
from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import pathlib
import shutil
import struct
import threading
import time
import urllib.parse
import uuid

from serve.dense_engine import RequestCancelled
from dense_bench import BoundedLog, EngineProcess, file_identity
from dense_visual_prepare import SDV_MAGIC, prepare_frontend


class ResidentVision:
    """One encoder load. ENC has no cancellation command, so cancellation drains its OK/ERR."""
    def __init__(self, executable, model, projector, task_dir, log_dir, environment, context,
                 gpu=True, threads=None, max_tokens=None, startup_timeout=1800, request_timeout=7200):
        self.task_dir, logs = pathlib.Path(task_dir).resolve(), pathlib.Path(log_dir).resolve()
        self.task_dir.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        self.gpu, self.request_timeout = gpu, request_timeout
        self.lock, self.closed = threading.Lock(), threading.Event()
        self.active, self.error, self.requests = None, None, 0
        command = [str(executable), "--model", str(model), "--mmproj", str(projector),
                   "--output-dir", str(self.task_dir), "--max-request-tokens", str(context)]
        if gpu:
            command += ["--gpu"]
        for key, value in (("--threads", threads), ("--max-tokens", max_tokens)):
            if value is not None:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                command += [key, str(value)]
        self.identity = {"encoder": file_identity(pathlib.Path(executable)),
                         "projector": file_identity(pathlib.Path(projector)), "gpu": gpu,
                         "text_metadata_only": True, "max_image_tokens": max_tokens}
        self.worker = EngineProcess(command, logs, environment)
        try:
            stamp, answer = self.worker.next(self.worker.started + startup_timeout)
            if answer != "READY 1024 SDV1":
                raise RuntimeError(f"unexpected resident image encoder greeting: {answer[:256]}")
            self.identity["host_process_ready_ms"] = (stamp-self.worker.started)*1000
        except BaseException as error:
            self.worker.close(False)
            raise RuntimeError(f"vision startup failed (exit={self.worker.process.returncode}): {error}") from error

    def alive(self):
        return not self.closed.is_set() and self.worker.process.poll() is None

    def snapshot(self):
        return dict(self.identity, status="ok" if self.alive() else "unavailable",
                    busy=self.active is not None, encode_requests=self.requests, error=self.error)

    def encode(self, output, images, cancel):
        while not self.lock.acquire(timeout=0.1):
            if cancel.is_set():
                raise RequestCancelled("cancelled before image encoding")
        try:
            if cancel.is_set():
                raise RequestCancelled("cancelled before image encoding")
            if not self.alive():
                raise RuntimeError("owned image encoder is unavailable")
            output = pathlib.Path(output).resolve()
            paths = [output, *(pathlib.Path(path).resolve() for path in images)]
            if not 1 <= len(images) <= 32 or any(not path.is_relative_to(self.task_dir) or
                    any(ch in str(path) for ch in "\t\r\n") for path in paths):
                raise ValueError("encoder paths must be task-owned and request must contain 1..32 images")
            self.active = cancel
            self.requests += 1
            begin = time.perf_counter()
            self.worker.send("\t".join(["ENC", *(str(path) for path in paths)]))
            deadline = begin + self.request_timeout
            while time.perf_counter() < deadline:
                try:
                    stamp, answer = self.worker.next(min(deadline, time.perf_counter()+0.1))
                except TimeoutError:
                    continue
                if answer.startswith("ERR "):
                    raise ValueError("image encoder rejected the request: " + answer[4:])
                fields = answer.split()
                if len(fields) != 6 or fields[0] != "OK":
                    raise RuntimeError(f"invalid image encoder response: {answer[:256]}")
                count, records, image_tokens, text_tokens = map(int, fields[1:5])
                elapsed = float(fields[5])
                if count != len(images) or records < count or image_tokens < count or text_tokens < 0 or \
                        not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("image encoder returned inconsistent counts/cost")
                with output.open("rb") as stream:
                    header = struct.unpack("<8I", stream.read(32))
                expected = (SDV_MAGIC, 1, 1024, count, records, text_tokens, image_tokens, int(self.gpu))
                if header != expected:
                    raise RuntimeError("real SDV1 artifact does not match its encoder acknowledgement")
                return {"images": count, "records": records, "image_tokens": image_tokens,
                        "encoder_text_tokens": text_tokens, "encoder_ms": elapsed,
                        "encoder_host_ms": (stamp-begin)*1000, "encoder_request_index": self.requests,
                        "first_encoder_request": self.requests == 1, "gpu": self.gpu}
            raise TimeoutError("image encoder request deadline exceeded")
        except (RequestCancelled, ValueError):
            raise
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
            self.closed.set()
            self.worker.close(False)
            raise
        finally:
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


def image_kind(raw):
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if raw.startswith(b"BM"):
        return "image/bmp", ".bmp"
    raise ValueError("supported still-image formats are PNG, JPEG and BMP")


class VisualPrepare:
    def __init__(self, encoder, max_output=65536, keep_artifacts=False):
        self.encoder, self.max_output, self.keep_artifacts = encoder, max_output, keep_artifacts
        self.task_dir = encoder.task_dir
        self.requests_dir = self.task_dir / "vision-requests"
        self.requests_dir.mkdir(exist_ok=True)
        self.log = BoundedLog(self.task_dir / "vision.requests.log")

    def snapshot(self):
        return dict(self.encoder.snapshot(), formats=["png", "jpeg", "bmp"], max_images=32,
                    input_sources=["base64_data_url", "task_local_file"], keep_artifacts=self.keep_artifacts)

    def read_image(self, source):
        maximum = 32 * 1024 * 1024
        if not isinstance(source, str) or not source:
            raise ValueError("image source must be a nonempty data URL or task-local path")
        declared = None
        if source.startswith("data:"):
            header, comma, payload = source.partition(",")
            if not comma or not header.endswith(";base64") or len(payload) > (maximum+2)//3*4:
                raise ValueError("image data URL must contain at most 32 MiB of base64 image data")
            declared = header[5:-7].lower()
            if declared not in ("image/png", "image/jpeg", "image/bmp"):
                raise ValueError("image data URLs must declare image/png, image/jpeg or image/bmp")
            try:
                raw = base64.b64decode(payload, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError("invalid image base64 payload") from error
            source_kind = "base64_data_url"
        else:
            if source.startswith("file:"):
                url = urllib.parse.urlsplit(source)
                if url.netloc not in ("", "localhost"):
                    raise ValueError("remote file URLs are not supported")
                source = urllib.parse.unquote(url.path)
                if len(source) > 3 and source[0] == "/" and source[2] == ":":
                    source = source[1:]
            elif "://" in source:
                raise ValueError("network image URLs are not configured; use a data URL or task-local file")
            path = pathlib.Path(source)
            if not path.is_absolute():
                raise ValueError("local image paths must be absolute")
            path = path.resolve()
            if not path.is_relative_to(self.task_dir) or not path.is_file() or path.stat().st_size > maximum:
                raise ValueError("local image must be a file of at most 32 MiB inside task-dir")
            raw, source_kind = path.read_bytes(), "task_local_file"
        if not raw or len(raw) > maximum:
            raise ValueError("image payload must be nonempty and at most 32 MiB")
        media_type, extension = image_kind(raw)
        if declared is not None and media_type != declared:
            raise ValueError("image data URL media type does not match its actual file header")
        return raw, media_type, extension, source_kind

    def cleanup(self, folder):
        if not self.keep_artifacts and folder.is_relative_to(self.requests_dir):
            shutil.rmtree(folder, ignore_errors=False)

    def __call__(self, frontend, messages, tools, kwargs, maximum, context, cancel=None):
        cancel = cancel or threading.Event()
        begin = time.perf_counter()
        folder = self.requests_dir / uuid.uuid4().hex
        folder.mkdir()
        request_id = folder.name
        try:
            normalized, image_paths, inputs, total_bytes = copy.deepcopy(messages), [], [], 0
            for message in normalized:
                if not isinstance(message.get("content"), list):
                    continue
                for part in message["content"]:
                    if part.get("type") != "image":
                        continue
                    if cancel.is_set():
                        raise RequestCancelled("cancelled while preparing image inputs")
                    raw, media_type, extension, source_kind = self.read_image(part.get("source"))
                    total_bytes += len(raw)
                    if len(image_paths) >= 32 or total_bytes > 64 * 1024 * 1024:
                        raise ValueError("request supports at most 32 images and 64 MiB total image bytes")
                    target = folder / f"image-{len(image_paths)+1:02d}{extension}"
                    with target.open("xb") as stream:
                        stream.write(raw)
                    image_paths.append(target)
                    inputs.append({"path": str(target), "bytes": len(raw), "media_type": media_type,
                                   "source_kind": source_kind, "sha256": hashlib.sha256(raw).hexdigest()})
                    part.clear()
                    part.update(type="image", source=str(target))
            if not image_paths:
                raise ValueError("visual request contains no supported user image parts")
            messages_path = folder / "messages.json"
            messages_path.write_text(json.dumps({"messages": normalized, "tools": tools,
                "template_kwargs": kwargs}, ensure_ascii=False) + "\n", encoding="utf-8")
            staged = time.perf_counter()
            sdv = folder / "encoded.sdv"
            cost = self.encoder.encode(sdv, image_paths, cancel)
            if cancel.is_set():
                raise RequestCancelled("cancelled after draining image encoding")
            prep_start = time.perf_counter()
            summary = prepare_frontend(frontend, messages_path, sdv, folder/"input.dvi", maximum,
                                       context, max_output=self.max_output)
            prepared = json.loads(pathlib.Path(summary["sidecar"]).read_text(encoding="utf-8"))
            cost.update(request_id=request_id, image_input_bytes=total_bytes,
                        image_stage_ms=(staged-begin)*1000,
                        template_sdv_dvi_ms=(time.perf_counter()-prep_start)*1000,
                        visual_prepare_ms=(time.perf_counter()-begin)*1000)
            prepared.update(dvi_path=summary["output"], vision_cost=cost,
                            cleanup=lambda: self.cleanup(folder))
            manifest = {"request_id": request_id, "inputs": inputs, "cost": cost,
                        "dvi": summary, "retained": self.keep_artifacts}
            (folder/"request.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            self.log.write(json.dumps({"request_id": request_id, "cost": cost}, ensure_ascii=False)+"\n")
            if cancel.is_set():
                raise RequestCancelled("cancelled after visual input preparation")
            return prepared
        except BaseException:
            self.cleanup(folder)
            raise

    def close(self):
        self.encoder.close()
        self.log.close()
