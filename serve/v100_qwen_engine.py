"""Bound IPC for the newly source-built completed V100 Qwen library entry."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time

from serve.native_engine import NativeFamilyRunner
from serve.native_sampling import parameters
from serve.native_slots import NativeSlots


class V100QwenRunner(NativeFamilyRunner):
    def __init__(self, cfg, stops, environment):
        self.cfg, self.family = cfg, "qwen35_dense_v100"
        self._aborted = False
        self.context, self.slots = cfg["context"], 1
        self.temp_dir, self.draft = cfg["temp_dir"], None
        self.slot_status = NativeSlots(1)
        self.available = queue.Queue(maxsize=1); self.available.put(0)
        self.commands, self.lines = queue.Queue(), queue.Queue(maxsize=128)
        # This is the new dedicated source-built executable. Its private CLI
        # retains the frozen internal family spelling; it cannot enter final223.
        args = [cfg["exe"], "--family", "qwen35_dense", "--variant", cfg["variant"],
                "--model", cfg["model"], "--context", str(self.context), "--slots", "1",
                "--mtp", str(cfg.get("mtp", 0)), "--kv-policy", cfg.get("kv_policy", "full")]
        self.proc = subprocess.Popen(args, cwd=self.temp_dir, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        try:
            if self._line().strip() != "READY slots=1":
                raise RuntimeError("completed V100 Qwen entry did not become ready")
        except BaseException:
            self.abort()
            self.close()
            raise
        self.io_thread = threading.Thread(target=self._serve_commands, daemon=True)
        self.io_thread.start()

    def _line(self):
        try:
            line = self.lines.get(timeout=float(self.cfg["response_timeout_s"]))
        except queue.Empty as error:
            # A timed-out reply cannot remain in this IPC stream for the next
            # command/request to consume under a different expected response.
            self.abort()
            raise RuntimeError("V100 Qwen response exceeded response_timeout_s") from error
        if line is None:
            raise RuntimeError("V100 Qwen engine output closed")
        return line

    def _command(self, line, expected):
        if self._aborted:
            raise RuntimeError("V100 Qwen engine was ended while draining its request")
        try:
            return super()._command(line, expected)
        except RuntimeError:
            self.abort()
            raise

    def abort(self):
        self._aborted = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.commands.put(None)

    def close(self):
        try:
            super().close()
        except (OSError, subprocess.TimeoutExpired):
            self.abort()
            raise
        finally:
            self.commands.put(None)
            thread = getattr(self, "io_thread", None)
            if thread is not None:
                thread.join(timeout=5)

    def _generate(self, slot, prompt_ids, max_new, stops, on_token, cancel, sampling):
        if not prompt_ids or max_new < 1 or len(prompt_ids) + max_new > self.context:
            raise ValueError("V100 Qwen complete prompt plus output exceeds context")
        encoded = parameters(sampling).split(" ", 1)[1]
        self._command(f"CONFIG 0 {encoded}", "CONFIGURED")
        # RELEASE clears these policies; only explicitly bound config choices
        # are re-applied while idle, before every PREPARE.
        for policy in self.cfg.get("native_policies", []):
            command = policy["command"]
            self._command(f"{command} 0 {policy['value']}", "POLICY" if command == "PPPOLICY" else command)
        start = time.perf_counter()
        if cancel is not None and cancel.is_set():
            return [], self._stats(start, 0, len(prompt_ids), 0, {})
        fields = self._command(f"PREPARE 0 {max_new} " + ",".join(map(str, prompt_ids)), "PREFILL")
        if int(fields["tokens"]) != len(prompt_ids):
            raise RuntimeError("V100 Qwen complete prefill extent changed")
        prefill_ms = 1000 * (time.perf_counter() - start)
        generated = []
        picks = [int(fields["anchor"])]
        while True:
            for token in picks:
                if cancel is not None and cancel.is_set():
                    break
                if len(generated) >= max_new:
                    raise RuntimeError("V100 Qwen step exceeded remaining output budget")
                generated.append(token)
                if on_token:
                    on_token(token)
                if token in stops:
                    break
            if (not generated or generated[-1] in stops or len(generated) == max_new or
                    (cancel is not None and cancel.is_set())):
                break
            fields = self._command(f"STEP 0 {generated[-1]} {max_new-len(generated)}", "STEPPED")
            picks = [int(v) for v in fields["ids"].split(",")]
            if not picks or int(fields["next"]) != picks[-1]:
                raise RuntimeError("V100 Qwen pending-token response changed")
        info = self._command("INFO 0", "INFO")
        return generated, self._stats(start, prefill_ms, len(prompt_ids), len(generated), info)

    def _stats(self, start, pp_ms, prompt, count, info):
        return dict(prompt_tokens=prompt, generated=count, pp_ms=pp_ms,
            total_ms=1000*(time.perf_counter()-start), windows=int(info.get("rounds", 0)),
            accepted_proposals=int(info.get("accepted", 0)),
            checked_proposals=int(info.get("proposed", 0)),
            native_speculative=bool(int(info.get("actual_mtp", 0))),
            dspark_requested=False, dspark_enabled=False, native_info=info)
