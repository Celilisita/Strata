"""Versioned per-expert ABI; no whole tensor materialization or implicit build."""
from pathlib import Path
import ctypes
import math
import os
import numpy as np


class Result(ctypes.Structure):
    _fields_ = [('global_scale', ctypes.c_float), ('maximum_error', ctypes.c_float),
                ('error2', ctypes.c_double), ('norm2', ctypes.c_double),
                ('zero_scales', ctypes.c_uint64), ('elements', ctypes.c_uint64)]


class GpuConverter:
    def __init__(self, library):
        library = Path(library).resolve(strict=True)
        self.dll_directories = []
        if os.name == 'nt':
            self.dll_directories.append(os.add_dll_directory(str(library.parent)))
            cuda = os.environ.get('CUDA_PATH')
            if cuda:
                for suffix in ('bin/x64', 'bin'):
                    path = Path(cuda) / suffix
                    if path.is_dir():
                        self.dll_directories.append(os.add_dll_directory(str(path)))
        self.library = ctypes.CDLL(str(library))
        self.library.moe_nvfp4_abi_v1.restype = ctypes.c_int
        if self.library.moe_nvfp4_abi_v1() != 1 or ctypes.sizeof(Result) != 40:
            raise RuntimeError('Wrong MoE NVFP4 converter ABI')
        self.library.moe_nvfp4_error_v1.restype = ctypes.c_char_p
        self.convert_expert = self.library.moe_nvfp4_convert_expert_v1
        self.convert_expert.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.POINTER(Result)]
        self.convert_expert.restype = ctypes.c_int

    def convert_raw(self, raw, elements, type_id):
        if type_id not in (12, 13, 14) or elements < 256 or elements > 1048576 or elements % 256:
            raise ValueError('Per-expert conversion admits Q4/Q5/Q6_K,256..1048576 aligned elements')
        expected = elements // 256 * {12: 144, 13: 176, 14: 210}[type_id]
        if len(raw) != expected:
            raise ValueError('Truncated expert conversion payload')
        original = np.frombuffer(raw, dtype=np.uint8)
        packed = np.empty(elements // 64 * 36, dtype=np.uint8)
        result = Result()
        if self.convert_expert(original.ctypes.data, elements, type_id, packed.ctypes.data, ctypes.byref(result)):
            raise RuntimeError(self.library.moe_nvfp4_error_v1().decode('utf-8', errors='strict'))
        if result.elements != elements or not all(math.isfinite(v) and v >= 0 for v in
                (result.global_scale, result.maximum_error, result.error2, result.norm2)) or result.global_scale <= 0:
            raise RuntimeError('Invalid GPU expert conversion result')
        return packed.tobytes(), float(result.global_scale), dict(error2=result.error2, norm2=result.norm2,
            max_error=float(result.maximum_error), zero_scales=int(result.zero_scales), elements=int(result.elements))
