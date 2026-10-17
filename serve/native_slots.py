"""Per-slot native request telemetry for the common metrics and slots APIs."""
import threading
import time


class NativeSlots:
    def __init__(self, count):
        self._live = [None] * count
        self._lock = threading.Lock()

    def begin(self, slot, prompt_tokens, max_tokens):
        with self._lock:
            self._live[slot] = {"state": "reading", "prompt_tokens": prompt_tokens,
                                "max_tokens": max_tokens, "generated": 0,
                                "started": time.monotonic(), "first_token": None}

    def token(self, slot):
        now = time.monotonic()
        with self._lock:
            row = self._live[slot]
            if row is not None:
                row["state"] = "generating"
                row["generated"] += 1
                if row["first_token"] is None:
                    row["first_token"] = now

    def release(self, slot):
        with self._lock:
            self._live[slot] = None

    def view(self):
        now = time.monotonic()
        with self._lock:
            rows = [None if row is None else dict(row) for row in self._live]
        result = []
        for slot, row in enumerate(rows):
            item = {"slot": slot, "id": slot, "state": "idle", "held_tokens": 0}
            if row is not None:
                first = row.pop("first_token")
                item.update(row, elapsed_s=round(now - row["started"], 3),
                            tok_s=round(row["generated"] / max(1e-6, now - first), 1) if first else None)
                item.pop("started", None)
            result.append(item)
        return result
