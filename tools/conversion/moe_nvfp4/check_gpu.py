"""Four real per-expert codec checks; no whole-file SHA, runtime or model load."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from gguf_reader import GGUFFile
from gpu_conversion import GpuConverter
from moe_conversion import BLOCK_BYTES, SUFFIX, geometry
from quantize_nvfp4 import FP8, codes_fp4, decode_nvfp4, dequantize, nearest_fp8, quantize


def require(ok, message):
    if not ok:
        raise ValueError(message)


def select(reader, role, dtype):
    shape, _ = geometry(reader, (40, 256, 2048, 512))
    expected = [512, 2048, 256] if role == 'down' else [2048, 512, 256]
    for tensor in reader.tensors:
        match = SUFFIX.fullmatch(tensor.name)
        if match and int(match[1]) < shape[0] and match[2] == role and tensor.type_id == dtype and tensor.shape == expected:
            return tensor
    raise ValueError('Actual ' + role + ' expert tensor type' + str(dtype) + ' missing')


def one(reader, tensor, backend, family):
    initial = reader.path.stat()
    expert = 0
    elements = tensor.shape[0] * tensor.shape[1]
    size = elements // 256 * BLOCK_BYTES[tensor.type_id]
    require(reader.data_start <= initial.st_size and tensor.offset <= initial.st_size-reader.data_start and
            tensor.expected_bytes() <= initial.st_size-reader.data_start-tensor.offset, 'Real expert file extent')
    with reader.path.open('rb') as stream:
        stream.seek(reader.data_start + tensor.offset + expert * size)
        raw = stream.read(size)
    require(len(raw) == size, 'Truncated real expert')
    values = dequantize(raw, tensor.type_id).reshape(-1)
    maximum = float(np.max(np.abs(values)))
    global_scale = np.float32(maximum / 2688 if maximum else 1.)
    cpu, cpu_stats = quantize(values, global_scale)
    packed, gpu_scale, gpu_stats = backend.convert_raw(raw, elements, tensor.type_id)
    require(np.float32(gpu_scale).tobytes() == global_scale.tobytes(), 'Independent per-expert F32 global differs')
    block = np.frombuffer(packed, dtype=np.uint8).reshape(-1, 36)
    require(np.all(block[:, :4] <= 126), 'Invalid E4M3 block scale')
    sf = block[:, :4].reshape(-1)
    groups = values.reshape(-1, 16)
    center = nearest_fp8(np.max(np.abs(groups), axis=1) / (6 * global_scale))
    allowed = np.stack([np.clip(center.astype(np.int16)+shift,0,126) for shift in (0,-1,1)])
    require(np.all(np.any(allowed == sf[None, :], axis=0)), 'GPU chose scale outside the declared three neighbors')
    scales = FP8[sf] * global_scale
    inverse = np.divide(1., scales, out=np.zeros_like(scales), where=scales > 0)
    actual_bytes = block[:, 4:].reshape(-1, 8)
    actual_codes = np.concatenate((actual_bytes & 15, actual_bytes >> 4), axis=1)
    require(np.array_equal(actual_codes, codes_fp4(groups * inverse[:, None])), 'GPU E2M1 RNE/nibble layout mismatch')
    a = decode_nvfp4(cpu, global_scale).reshape(-1, 16)
    b = decode_nvfp4(packed, global_scale).reshape(-1, 16)
    require(np.isfinite(b).all(), 'Nonfinite GPU reconstructed weight')
    source = groups.astype(np.float64)
    ae = np.sum((a.astype(np.float64)-source)**2, axis=1)
    be = np.sum((b.astype(np.float64)-source)**2, axis=1)
    norm = np.sum(source*source, axis=1)
    # CPU/GPU FP64 SSE summation may choose another tied scale, but every
    # block must remain at the CPU three-neighbor minimum within roundoff.
    tolerance = 128 * np.finfo(np.float64).eps * (norm + ae) + 1e-300
    require(np.all(be <= ae + tolerance), 'GPU block16 scale is not the same three-neighbor SSE minimum')
    error2, norm2 = float(be.sum()), float(norm.sum())
    require(math.isclose(gpu_stats['error2'], error2, rel_tol=1e-8, abs_tol=1e-18*norm2+1e-300) and
            math.isclose(gpu_stats['norm2'], norm2, rel_tol=1e-8, abs_tol=1e-300), 'GPU statistics mismatch decoded payload')
    require(gpu_stats['zero_scales'] == int(np.count_nonzero(sf == 0)), 'GPU scale count mismatch')
    final = reader.path.stat()
    require((initial.st_size,initial.st_mtime_ns)==(final.st_size,final.st_mtime_ns), 'GGUF changed during real expert check')
    return dict(family=family, tensor=tensor.name, type=tensor.type_id, shape=tensor.shape, expert=expert,
        source_path=str(reader.path), source_bytes=initial.st_size, source_mtime_ns=initial.st_mtime_ns,
        expert_source_bytes=size, expert_elements=elements, expert_nvfp4_bytes=len(packed), global_scale=float(global_scale),
        source_expert_sha256=hashlib.sha256(raw).hexdigest(), nvfp4_expert_sha256=hashlib.sha256(packed).hexdigest(),
        cpu_payload_sha256=hashlib.sha256(cpu).hexdigest(), cpu_gpu_payload_equal=cpu==packed,
        per_expert_global_exact=True, full_payload_RNE_layout=True, all_block16_SSE_minimum=True,
        finite=True, original_quantized_relative_l2=math.sqrt(error2/(norm2+1e-300)),
        cpu_relative_l2=math.sqrt(cpu_stats['error2']/(cpu_stats['norm2']+1e-300)),
        statistics=gpu_stats, mathematical_passed=True, model_quality_qualified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mini', type=Path, required=True)
    parser.add_argument('--qwen', type=Path, required=True)
    parser.add_argument('--gpu-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    mini, qwen = GGUFFile(args.mini.resolve(strict=True)), GGUFFile(args.qwen.resolve(strict=True))
    backend = GpuConverter(args.gpu_library)
    results = []
    for reader, family, role, dtype in [(mini,'Mini','gate',12),(mini,'Mini','down',13),
                                       (mini,'Mini','down',12),(qwen,'Qwen','down',14)]:
        results.append(one(reader,select(reader,role,dtype),backend,family))
    result = dict(status='real-expert-NVFP4-codec-passed', passed=True, cases=results, source_full_hashes=0,
                  full_model_loads=0, runtime_qualified=False, quantization_algorithm='perexpert-global/block16-threeSF/E2M1-RNE')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],passed=True,cases=len(results),output=str(args.output),full_model_loads=0)),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,error=str(error))),file=sys.stderr)
        raise SystemExit(1)
