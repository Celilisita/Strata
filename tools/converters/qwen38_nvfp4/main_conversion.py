"""Preserve the frozen main-layer selection and NVFP4 quantization mathematics."""
import hashlib
import math
import re
import struct
import os
from pathlib import Path
import numpy as np
from gguf_reader import GGUFFile
from quantize_nvfp4 import dequantize, quantize
from conversion_io import (ALGORITHM, Transaction, align, header, identity,
                           implementation_identity, reuse, source_chunks, string_records,
                           validate_output, validate_reader, verify_source)


def metadata_records(source, reader):
    records = []
    extras = {'strata.source.file': Path(source).name,
              'strata.source.quantization': 'Q4_K_M',
              'strata.compute_quantization': 'NVFP4',
              'strata.conversion': 'Original decoded Q4_K/Q6_K; block16 E4M3 scale MSE search; E2M1 RNE'}
    if any(key in reader.metadata for key in extras):
        raise ValueError('Source already has conversion metadata; use the original Q4_K_M GGUF')
    with Path(source).open('rb') as stream:
        stream.seek(16)
        count, = struct.unpack('<Q', stream.read(8))
        for unused in range(count):
            start = stream.tell()
            name = reader._str(stream)
            reader._value(stream)
            end = stream.tell()
            if name != 'general.file_type':
                stream.seek(start)
                records.append(stream.read(end - start))
    return records + string_records(extras)


def plan_entries(reader):
    block_count = int(reader.metadata['qwen35.block_count'])
    main_layers = block_count - int(reader.metadata.get('qwen35.nextn_predict_layers', 0))
    if main_layers <= 0:
        raise ValueError('Invalid main-layer count')
    entries, offset = [], 0
    original_names = {tensor.name for tensor in reader.tensors}
    for tensor in reader.tensors:
        match = re.match(r'blk\.(\d+)\.', tensor.name)
        replace = bool(match and int(match[1]) < main_layers and len(tensor.shape) == 2
                       and tensor.type_id in (12, 14) and tensor.shape[1] >= 256)
        size = tensor.elements // 64 * 36 if replace else tensor.expected_bytes()
        offset = align(offset, reader.alignment)
        entries.append(dict(tensor=tensor, name=tensor.name, shape=tensor.shape,
                            type=40 if replace else tensor.type_id, offset=offset,
                            size=size, replace=replace))
        offset += size
        if replace:
            if not tensor.name.endswith('.weight'):
                raise ValueError('Selected matrix lacks the expected weight name: ' + tensor.name)
            scale_name = tensor.name[:-7] + '.scale'
            if scale_name in original_names:
                raise ValueError('Generated scale tensor would replace source bytes: ' + scale_name)
            offset = align(offset, reader.alignment)
            entries.append(dict(tensor=tensor, name=scale_name, shape=[1], type=0,
                                offset=offset, size=4, scalar=True))
            offset += 4
    if not any(entry.get('replace') for entry in entries):
        raise ValueError('No eligible Q4_K/Q6_K main-layer matrices')
    return entries


def cpu_tensor(source, reader, tensor, write):
    maximum = 0.
    for raw in source_chunks(source, tensor, reader.data_start):
        maximum = max(maximum, float(np.max(np.abs(dequantize(raw, tensor.type_id)))))
    global_scale = np.float32(maximum / (6 * 448) if maximum else 1.)
    statistics = dict(error2=0., norm2=0., max_error=0., zero_scales=0, elements=0)
    digest = hashlib.sha256()
    for raw in source_chunks(source, tensor, reader.data_start):
        packed, result = quantize(dequantize(raw, tensor.type_id), global_scale)
        write(packed)
        digest.update(packed)
        for key in statistics:
            statistics[key] = (max(statistics[key], result[key]) if key == 'max_error'
                               else statistics[key] + result[key])
    return float(global_scale), statistics, digest.hexdigest()


def convert_main(source, output, work, backend_name, gpu_library=None):
    source, output = Path(source), Path(output)
    source_identity = identity(source)
    implementation = implementation_identity()
    request = dict(algorithm=ALGORITHM, kind='main', backend=backend_name,
                   source_sha256=source_identity['sha256'], source_bytes=source_identity['bytes'],
                   implementation=implementation)
    if backend_name == 'gpu':
        request['gpu_library'] = identity(gpu_library)
    previous = reuse(output, request)
    if previous:
        return previous
    reader = GGUFFile(source)
    validate_reader(reader)
    entries = plan_entries(reader)
    records = metadata_records(source, reader)
    raw_header, data_start = header(entries, records, reader.alignment)
    backend = None
    if backend_name == 'gpu':
        from gpu_conversion import GpuConverter
        # The historical native ABI admits at most 200M elements per tensor.
        if any(e.get('replace') and e['tensor'].elements > 200_000_000 for e in entries):
            raise ValueError('An eligible matrix exceeds the unchanged GPU ABI limit; select --backend cpu')
        backend = GpuConverter(gpu_library)
    tensor_results, scales = [], {}
    with Transaction(work, output) as transaction:
        transaction.status('converting-main', backend=backend_name,
                           source_sha256=source_identity['sha256'])
        with transaction.partial.open('xb') as destination, source.open('rb') as original:
            def write(raw):
                destination.write(raw)

            def pad_to(position):
                gap = position - destination.tell()
                if gap < 0:
                    raise RuntimeError('Overlapping output tensor extents')
                if gap:
                    write(bytes(gap))

            write(raw_header)
            pad_to(data_start)
            for entry in entries:
                tensor = entry['tensor']
                pad_to(data_start + entry['offset'])
                if entry.get('scalar'):
                    write(struct.pack('<f', scales[tensor.name]))
                    continue
                if entry['replace']:
                    if backend:
                        packed, global_scale, statistics = backend.convert(source, tensor, reader.data_start)
                        write(packed)
                        digest = hashlib.sha256(packed).hexdigest()
                    else:
                        global_scale, statistics, digest = cpu_tensor(source, reader, tensor, write)
                    scales[tensor.name] = float(global_scale)
                    statistics['relative_l2'] = math.sqrt(statistics['error2'] / (statistics['norm2'] + 1e-300))
                    tensor_results.append(dict(name=tensor.name, converted=True,
                                               global_scale=float(global_scale), original_type=tensor.type_id,
                                               sha256=digest, **statistics))
                else:
                    original.seek(reader.data_start + tensor.offset)
                    remaining = entry['size']
                    digest = hashlib.sha256()
                    while remaining:
                        raw = original.read(min(16 * 1024 * 1024, remaining))
                        if not raw:
                            raise ValueError('Truncated unchanged tensor: ' + tensor.name)
                        write(raw)
                        digest.update(raw)
                        remaining -= len(raw)
                    tensor_results.append(dict(name=tensor.name, converted=False, sha256=digest.hexdigest()))
                if destination.tell() != data_start + entry['offset'] + entry['size']:
                    raise RuntimeError('Converted tensor extent mismatch: ' + tensor.name)
                transaction.status('converting-main', completed_tensors=len(tensor_results),
                                   total_tensors=len(reader.tensors), last_tensor=tensor.name)
            destination.flush()
            os.fsync(destination.fileno())
        validate_output(transaction.partial, entries, data_start)
        verify_source(source, source_identity)
        if implementation_identity() != implementation:
            raise RuntimeError('Converter source files changed during conversion')
        if backend and identity(gpu_library) != request['gpu_library']:
            raise RuntimeError('GPU library changed during conversion')
        return transaction.finish(request, source=source_identity, backend=backend_name,
                                  converted_tensors=sum(bool(e.get('replace')) for e in entries),
                                  tensor_results=tensor_results, full_model_loads=0)
