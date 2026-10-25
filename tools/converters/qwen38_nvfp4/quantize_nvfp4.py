"""Bounded NumPy Q4_K/Q6_K -> GGUF NVFP4 conversion math.

The original tensor is the source of truth. Scalar global factors remain
separate from E4M3 block16 factors and E2M1 values.
"""
import numpy as np

J = np.arange(256)
FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
MID4 = (FP4[:-1] + FP4[1:]) * .5
I = np.arange(127)
FP8 = np.where(I < 8, I / 512., (1 + (I % 8) / 8.) * np.exp2(I // 8 - 7)).astype(np.float32)

def dequantize(raw, type_id):
    """Decode original GGUF blocks as float32; no runtime-layout assumptions."""
    size = {12: 144, 14: 210}[type_id]
    block = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size)
    if type_id == 12:
        delta = block[:, :2].copy().view('<f2').astype(np.float32)
        minimum = block[:, 2:4].copy().view('<f2').astype(np.float32)
        meta = block[:, 4:16].astype(np.uint16)
        scales = np.concatenate((meta[:, :4] & 63,
            (meta[:, 8:12] & 15) | ((meta[:, :4] >> 6) << 4)), axis=1)
        mins = np.concatenate((meta[:, 4:8] & 63,
            (meta[:, 8:12] >> 4) | ((meta[:, 4:8] >> 6) << 4)), axis=1)
        packed = block[:, 16:].reshape(-1, 4, 32)
        codes = np.stack((packed & 15, packed >> 4), axis=2).reshape(-1, 8, 32)
        output = (delta * scales)[:, :, None] * codes - (minimum * mins)[:, :, None]
    else:
        low_index = (J // 128) * 64 + ((J % 128 // 32) & 1) * 32 + J % 32
        low = (block[:, low_index] >> ((J % 128 // 64) * 4)) & 15
        high = (block[:, 128 + (J // 128) * 32 + J % 32] >> ((J % 128 // 32) * 2)) & 3
        codes = (low | (high << 4)).astype(np.int16) - 32
        scales = block[:, 192:208].view(np.int8)[:, J // 16].astype(np.float32)
        delta = block[:, 208:210].copy().view('<f2').astype(np.float32)
        output = (delta * scales) * codes
    output = output.reshape(-1, 256)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite original quantized payload')
    return output

def nearest_fp8(values):
    high = np.clip(np.searchsorted(FP8, values), 1, 126)
    low = high - 1
    a, b = values - FP8[low], FP8[high] - values
    result = np.where((a < b) | ((a == b) & ((low & 1) == 0)), low, high)
    return np.where(values <= 0, 0, result).astype(np.uint8)

def codes_fp4(values):
    magnitude = np.abs(values)
    index = np.searchsorted(MID4, magnitude, side='left').astype(np.uint8)
    safe = np.minimum(index, 6)
    tie = (index < 7) & (magnitude == MID4[safe]) & ((index & 1) != 0)
    index += tie.astype(np.uint8)
    return index | (np.signbit(values).astype(np.uint8) << 3)

def quantize(values, global_scale):
    groups = values.reshape(-1, 16)
    maximum = np.max(np.abs(groups), axis=1)
    center = nearest_fp8(maximum / (6 * global_scale))
    best_error = np.full(len(groups), np.inf, dtype=np.float64)
    best_codes = np.empty(groups.shape, dtype=np.uint8)
    best_scale = np.empty(len(groups), dtype=np.uint8)
    best_values = np.empty_like(groups)
    # Three adjacent scale encodings minimize actual block16 weight error.
    for shift in (0, -1, 1):
        scale_code = np.clip(center.astype(np.int16) + shift, 0, 126).astype(np.uint8)
        scale = FP8[scale_code] * global_scale
        inverse = np.divide(1., scale, out=np.zeros_like(scale), where=scale > 0)
        code = codes_fp4(groups * inverse[:, None])
        decoded = FP4[code & 7] * np.where(code & 8, -1., 1.).astype(np.float32) * scale[:, None]
        error = np.sum((decoded.astype(np.float64) - groups) ** 2, axis=1)
        take = error < best_error
        best_error[take] = error[take]
        best_codes[take] = code[take]
        best_scale[take] = scale_code[take]
        best_values[take] = decoded[take]
    out = np.empty((len(groups) // 4, 36), dtype=np.uint8)
    out[:, :4] = best_scale.reshape(-1, 4)
    packed = best_codes[:, :8] | (best_codes[:, 8:] << 4)
    out[:, 4:] = packed.reshape(-1, 32)
    return out.tobytes(), dict(error2=float(best_error.sum()),
        norm2=float(np.sum(groups.astype(np.float64) ** 2)),
        max_error=float(np.max(np.abs(best_values - groups))),
        zero_scales=int(np.count_nonzero(best_scale == 0)), elements=int(values.size))
