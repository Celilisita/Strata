"""Shared validated sampling serialization for native families."""
import math
import secrets

def parameters(sampling):
    s = sampling or {}
    if s.get("experimental_speed_projection"):
        raise ValueError("native Qwen35 does not enable experimental projection")
    values = [s.get("temperature", 0.6), s.get("top_k", 20), s.get("top_p", 0.95),
              s.get("min_p", 0), s.get("seed"), s.get("repetition_penalty", 1),
              s.get("frequency_penalty", 0), s.get("presence_penalty", 0),
              s.get("penalty_last_n", 64), s.get("min_keep", 1)]
    if values[4] is None or values[4] == -1:
        values[4] = secrets.randbits(64)
    for i in (1, 4, 8, 9):
        if isinstance(values[i], bool) or not isinstance(values[i], int):
            raise ValueError("top_k, seed and penalty window must be integers")
    for i in (0, 2, 3, 5, 6, 7):
        if not isinstance(values[i], (float, int)) or not math.isfinite(values[i]):
            raise ValueError("sampling values must be finite")
    if values[1] < 0 or not 0 <= values[4] < 2**64 or values[8] < 0 or values[9] < 1:
        raise ValueError("sampling integer outside supported range")
    if values[0] < 0 or not 0 <= values[2] <= 1 or not 0 <= values[3] <= 1 or values[5] <= 0:
        raise ValueError("sampling probability or repetition penalty outside supported range")
    return "PARAM " + " ".join(map(str, values))
