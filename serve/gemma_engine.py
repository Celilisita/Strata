"""Bound IPC for the newly source-built completed Gemma library entry."""
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


class GemmaRunner(NativeFamilyRunner):
    def __init__(self, cfg, stops, environment):
        self.cfg, self.family = cfg, "gemma4_dense"
        from serve.gemma_family import configured_assistant
        self.mtp_requested, self.mtp_launch, self.assistant_disabled_reason = configured_assistant(cfg)
        self._aborted = False
        self.context, self.slots = cfg["context"], 1
        self.temp_dir, self.draft = cfg["temp_dir"], None
        self.slot_status = NativeSlots(1)
        self.available = queue.Queue(maxsize=1); self.available.put(0)
        self.commands, self.lines = queue.Queue(), queue.Queue(maxsize=128)
        args = [cfg["exe"], "--model", cfg["model"], "--context", str(self.context),
                "--batch", "8192", "--mtp", str(self.mtp_launch),
                "--eog", ",".join(map(str, sorted(stops)))]
        if cfg.get("assistant") and self.mtp_launch:
            args += ["--assistant", cfg["assistant"]]
        for key, flag in (("pp_hh349", "--pp-hh349"), ("tg_gqa371", "--tg-gqa371"),
                          ("draft_head399", "--draft-head399"), ("ffn_full_k", "--ffn-full-k"),
                          ("swa_tc403", "--swa-tc403"), ("history_experimental", "--history-experimental")):
            if key in cfg:
                args += [flag, str(int(cfg[key]))]
        self.proc = subprocess.Popen(args, cwd=self.temp_dir, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            text=True, encoding="utf-8", bufsize=1)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        try:
            if not self._line().strip().startswith("READY slots=1 family=gemma4_dense "):
                raise RuntimeError("completed Gemma entry did not become ready")
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
            raise RuntimeError("Gemma response exceeded response_timeout_s") from error
        if line is None:
            raise RuntimeError("Gemma engine output closed")
        return line

    def _command(self, line, expected):
        if self._aborted:
            raise RuntimeError("Gemma engine was ended while draining its request")
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
            raise ValueError("Gemma complete prompt plus output exceeds context")
        encoded = parameters(sampling).split(" ", 1)[1]
        self._command(f"CONFIG 0 {encoded}", "CONFIGURED")
        start = time.perf_counter()
        if cancel is not None and cancel.is_set():
            return [], self._stats(start, 0, len(prompt_ids), 0, {})
        fields = self._command(f"PREPARE 0 {max_new} " + ",".join(map(str, prompt_ids)), "PREFILL")
        if int(fields["tokens"]) != len(prompt_ids):
            raise RuntimeError("Gemma complete prefill extent changed")
        prefill_ms = 1000 * (time.perf_counter() - start)
        generated = []
        picks = [int(fields["anchor"])]
        while True:
            for token in picks:
                if cancel is not None and cancel.is_set():
                    break
                if len(generated) >= max_new:
                    raise RuntimeError("Gemma step exceeded remaining output budget")
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
                raise RuntimeError("Gemma pending-token response changed")
        info = self._command("INFO 0", "INFO")
        return generated, self._stats(start, prefill_ms, len(prompt_ids), len(generated), info)

    def _stats(self, start, pp_ms, prompt, count, info):
        return dict(prompt_tokens=prompt, generated=count, pp_ms=pp_ms,
            total_ms=1000*(time.perf_counter()-start), windows=int(info.get("rounds", 0)),
            accepted_proposals=int(info.get("accepted", 0)),
            checked_proposals=max(0, int(info.get("rounds", 0))-int(info.get("base_T1_tail_rounds", 0))),
            native_speculative=bool(int(info.get("actual_mtp", 0))),
            dspark_requested=False, dspark_enabled=False, native_info=info)
