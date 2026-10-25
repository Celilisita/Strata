"""Original GPU conversion ABI; an explicit prebuilt library, never an implicit build."""
from pathlib import Path
import ctypes
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
            cuda_root = os.environ.get('CUDA_PATH')
            if cuda_root:
                for suffix in ('bin/x64', 'bin'):
                    directory = Path(cuda_root) / suffix
                    if directory.is_dir():
                        self.dll_directories.append(os.add_dll_directory(str(directory)))
        self.library = ctypes.CDLL(str(library))
        self.library.sfp4_error.restype = ctypes.c_char_p
        self.library.sfp4_convert.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_void_p, ctypes.POINTER(Result)]
        self.library.sfp4_convert.restype = ctypes.c_int

    def convert(self, source, tensor, data_start):
        if tensor.elements > 200_000_000:
            raise ValueError('Conversion tensor exceeds admitted 200M-element bound')
        with Path(source).open('rb') as stream:
            stream.seek(data_start + tensor.offset)
            raw = stream.read(tensor.expected_bytes())
        if len(raw) != tensor.expected_bytes():
            raise ValueError('Truncated original GPU conversion source')
        original = np.frombuffer(raw, dtype=np.uint8)
        output = np.empty(tensor.elements // 64 * 36, dtype=np.uint8)
        result = Result()
        status = self.library.sfp4_convert(original.ctypes.data, tensor.elements, tensor.type_id,
            output.ctypes.data, ctypes.byref(result))
        if status:
            raise RuntimeError(self.library.sfp4_error().decode('utf-8'))
        return output.tobytes(), float(result.global_scale), dict(error2=result.error2,
            norm2=result.norm2, max_error=float(result.maximum_error),
            zero_scales=int(result.zero_scales), elements=int(result.elements))
