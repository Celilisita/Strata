"""MiniCPM 4.7 DSpark proposals with Strata target verification."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


from serve.native_sampling import parameters as sampling_parameters
from serve.native_slots import NativeSlots


class NativeFamilyRunner:
    def __init__(self, executable: Path, target: Path, draft: Path,
                 gguf_python: Path, context: int, temp_dir: Path,
                 block_size: int = 15, slots: int = 1, dspark: bool = False, *,
                 family="qwen35_moe", variant="minicpm_v47", mtp=0,
                 kv_policy="ram8k-reuse", environment=None):
        if family == "qwen35_moe" and (environment or os.environ).get("STRATA_QWEN35_MTP") == "1":
            raise ValueError("MiniCPM 4.7 uses DSpark; disable MTP")
        if not temp_dir.is_dir():
            raise ValueError("DSpark temporary directory must exist")
        if not 1 <= block_size <= 15:
            raise ValueError("DSpark block size must be 1..15")
        if slots not in (1, 2) or (dspark and slots != 1):
            raise ValueError("DSpark draft mode supports one slot; target-only mode supports two")
        self.family = family
        self.target = target
        self.context = context
        self.block_size = block_size
        self.dspark = dspark
        self.slots = slots
        self.slot_status = NativeSlots(slots)
        self.available = queue.Queue(maxsize=slots)
        self.commands = queue.Queue()
        for slot in range(slots):
            self.available.put(slot)
        self.temp_dir = temp_dir
        self.aux_path = temp_dir / "minicpm-dspark-active-aux.f32"
        self.draft = None
        if dspark:
            from serve.minicpm_dspark import MiniCPMDSpark
            self.draft = MiniCPMDSpark(draft, target, gguf_python)
        env = dict(os.environ if environment is None else environment)
        env["STRATA_QWEN35_MTP"] = "0"
        env["STRATA_QWEN35_SAMPLE"] = "0"
        args = [str(executable), "--family", family, "--variant", variant,
                "--model", str(target), "--context", str(context), "--slots", str(slots)]
        if family == "qwen35_moe":
            args += ["--aux", str(self.aux_path), "--dspark", "1" if dspark else "0"]
        else:
            args += ["--mtp", str(mtp), "--kv-policy", kv_policy]
        self.proc = subprocess.Popen(args, cwd=temp_dir, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=sys.stderr, text=True, encoding="utf-8", bufsize=1)
        self.lines = queue.Queue(maxsize=128)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        if self._line().strip() != f"READY slots={slots}":
            self.close()
            raise RuntimeError("Strata DSpark target did not start")
        self.io_thread = threading.Thread(target=self._serve_commands, daemon=True)
        self.io_thread.start()

    def _read_stdout(self):
        try:
            for line in self.proc.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def _line(self):
        try:
            line = self.lines.get(timeout=900)
        except queue.Empty as error:
            raise RuntimeError("native engine response timed out") from error
        if line is None:
            raise RuntimeError("native engine output closed")
        return line

    def close(self):
        if getattr(self, "proc", None) is not None:
            if self.proc.poll() is None:
                if getattr(self, "io_thread", None) is not None:
                    self.commands.put(None)
                    self.io_thread.join(timeout=5)
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            self.proc = None

    def _serve_commands(self):
        while True:
            item = self.commands.get()
            if item is None:
                return
            line, reply = item
            try:
                self.proc.stdin.write(line + "\n")
                self.proc.stdin.flush()
                reply.put(self._line().strip())
            except BaseException as error:
                reply.put(error)

    def _command(self, line: str, expected: str) -> dict[str, str] | None:
        reply = queue.Queue(maxsize=1)
        self.commands.put((line, reply))
        response = reply.get()
        if isinstance(response, BaseException):
            raise response
        if response == "BUSY" and expected == "PREFILL":
            return None
        if response.startswith("ERROR "):
            raise ValueError(response[6:])
        if not response.startswith(expected + " "):
            raise RuntimeError(f"Strata DSpark target {expected}: {response or 'exited'}")
        return dict(field.split("=", 1) for field in response.split()[1:])

    def _aux(self, count: int):
        import numpy as np
        import torch
        values = np.fromfile(self.aux_path, dtype=np.float32)
        if values.size != count * 5 * 2048 or not np.isfinite(values).all():
            raise RuntimeError("DSpark target auxiliary state length or values invalid")
        return torch.from_numpy(values.reshape(count, 5, 2048))

    def generate(self, prompt_ids: list[int], max_new: int,
                 stops: set[int], on_token: Callable[[int], None] | None = None,
                 cancel=None, sampling=None) -> tuple[list[int], dict]:
        slot = self.available.get()
        self.slot_status.begin(slot, len(prompt_ids), max_new)
        def report(token):
            self.slot_status.token(slot)
            if on_token:
                on_token(token)
        try:
            return self._generate(slot, prompt_ids, max_new, stops, report, cancel, sampling)
        finally:
            visual_path = getattr(prompt_ids, "visual_path", None)
            if visual_path:
                visual_path.unlink(missing_ok=True)
            failed = sys.exc_info()[0] is not None
            try:
                self._command(f"RELEASE {slot}", "RELEASED")
            except Exception:
                if not failed:
                    raise
            finally:
                self.slot_status.release(slot)
                self.available.put(slot)

    def _generate(self, slot: int, prompt_ids: list[int], max_new: int,
                  stops: set[int], on_token, cancel, sampling) -> tuple[list[int], dict]:
        if (len(prompt_ids) < 2 or max_new < 1 or
            len(prompt_ids) + max_new > self.context):
            raise ValueError("DSpark prompt or generation exceeds context")
        if self.draft is not None:
            self.draft.reset()
        else:
            options = dict(sampling or {})
            options.pop("experimental_speed_projection", None)
            options.setdefault("temperature", 0)
            parameters = sampling_parameters(options).split(" ", 1)[1]
            self._command(f"CONFIG {slot} {parameters}", "CONFIGURED")
        started = time.perf_counter()
        visual_path = getattr(prompt_ids, "visual_path", None)
        try:
            command = (f"PREPAREV {slot} {max_new} {visual_path}" if visual_path else
                       f"PREPARE {slot} {max_new} " + ",".join(map(str, prompt_ids)))
            while True:
                if cancel is not None and cancel.is_set():
                    return [], {"prompt_tokens": len(prompt_ids), "generated": 0,
                                "pp_ms": 0, "total_ms": 0, "windows": 0,
                                "accepted_proposals": 0, "checked_proposals": 0,
                                "draft_ms": 0, "verify_ms": 0}
                fields = self._command(command, "PREFILL")
                if fields is not None:
                    break
                time.sleep(0.1)
        finally:
            if visual_path:
                visual_path.unlink(missing_ok=True)
        prompt_size = len(prompt_ids)
        if int(fields["tokens"]) != prompt_size:
            raise RuntimeError("Strata target prompt length changed")
        if self.draft is not None:
            self.draft.append_context(self._aux(prompt_size), 0)
        anchor = int(fields["anchor"])
        prefill_ms = 1000 * (time.perf_counter() - started)
        generated: list[int] = []
        emitted = 0
        if self.draft is None:
            generated.append(anchor)
            if on_token and not (cancel is not None and cancel.is_set()):
                on_token(anchor)
            while len(generated) < max_new and generated[-1] not in stops:
                if cancel is not None and cancel.is_set():
                    break
                command = f"STEP {slot} {generated[-1]}"
                if self.family == "qwen35_dense":
                    command += f" {max_new - len(generated)}"
                fields = self._command(command, "STEPPED")
                picks = ([int(v) for v in fields["ids"].split(",")] if "ids" in fields
                         else [int(fields["next"])])
                if not picks or len(picks) > max_new - len(generated):
                    raise RuntimeError("native generation output extent changed")
                for token in picks:
                    if cancel is not None and cancel.is_set():
                        break
                    generated.append(token)
                    if on_token:
                        on_token(token)
                    if token in stops:
                        break
            elapsed_ms = 1000 * (time.perf_counter() - started)
            return generated, {"prompt_tokens": prompt_size, "generated": len(generated),
                "pp_ms": round(prefill_ms, 3), "total_ms": round(elapsed_ms, 3),
                "draft_ms": 0.0, "verify_ms": 0.0, "windows": 0,
                "checked_proposals": 0, "accepted_proposals": 0}
        if on_token and not (cancel is not None and cancel.is_set()):
            on_token(anchor)
            emitted = 1
        windows = accepted_proposals = checked = 0
        draft_ms = verify_ms = 0.0
        while len(generated) < max_new:
            if cancel is not None and cancel.is_set():
                break
            remaining = max_new - len(generated)
            if remaining == 1 or anchor in stops:
                generated.append(anchor)
                if on_token and len(generated) > emitted:
                    on_token(anchor)
                    emitted += 1
                break
            n_proposals = min(self.block_size, remaining - 1)
            t0 = time.perf_counter()
            proposals, _ = self.draft.draft(anchor, self.draft.context_length,
                                            n_proposals)
            draft_ms += 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            fields = self._command(f"VERIFY {slot} {anchor} " + ",".join(map(str, proposals)),
                                   "VERIFIED")
            verified_ids = [int(value) for value in fields["ids"].split(",")]
            if (not verified_ids or verified_ids[0] != anchor or
                len(verified_ids) > remaining or
                int(fields["verified"]) != n_proposals + 1):
                raise RuntimeError("Strata DSpark verification contract violated")
            self.draft.append_context(self._aux(len(verified_ids)),
                                      self.draft.context_length)
            verify_ms += 1000 * (time.perf_counter() - t0)
            windows += 1
            checked += n_proposals
            accepted_proposals += len(verified_ids) - 1
            anchor = int(fields["next"])
            for token in verified_ids:
                generated.append(token)
                if on_token and len(generated) > emitted:
                    on_token(token)
                    emitted += 1
                if token in stops:
                    break
            if generated[-1] in stops:
                break
        elapsed_ms = 1000 * (time.perf_counter() - started)
        stats = {"prompt_tokens": prompt_size, "generated": len(generated),
                 "pp_ms": round(prefill_ms, 3), "total_ms": round(elapsed_ms, 3),
                 "draft_ms": round(draft_ms, 3), "verify_ms": round(verify_ms, 3),
                 "windows": windows, "checked_proposals": checked,
                 "accepted_proposals": accepted_proposals,
                 "acceptance": round(accepted_proposals / checked, 4) if checked else 0.0,
                 "tg_tok_s": round(1000 * max(0, len(generated) - 1) /
                                   max(1e-6, elapsed_ms - prefill_ms), 3)}
        return generated, stats

