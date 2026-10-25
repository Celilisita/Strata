"""Main-stack routed experts only; every expert has its own F32 global scale."""
from pathlib import Path
import hashlib
import math
import os
import re
import shutil
import struct
import numpy as np
from gguf_reader import GGUFFile
from quantize_nvfp4 import dequantize, quantize
from conversion_io import (ALGORITHM, Transaction, align, header, identity, implementation_identity,
                           reuse, source_chunks, string_records, validate_output, validate_reader)

SUFFIX = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\Z')
BLOCK_BYTES = {12: 144, 13: 176, 14: 210}


def geometry(reader, expected=None):
    m = reader.metadata
    if m.get('general.architecture') != 'qwen35moe':
        raise ValueError('This converter requires actual qwen35moe metadata')
    blocks = int(m['qwen35moe.block_count'])
    mtp = int(m.get('qwen35moe.nextn_predict_layers', 0))
    shape = (blocks - mtp, int(m['qwen35moe.expert_count']),
             int(m['qwen35moe.embedding_length']), int(m['qwen35moe.expert_feed_forward_length']))
    if not 0 <= mtp < blocks or not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 4096 or any(v <= 0 for v in shape):
        raise ValueError('Invalid MoE main-stack geometry')
    if expected is not None and shape != tuple(expected):
        raise ValueError('MoE geometry differs from requested contract: ' + str(shape))
    if any(v % 256 for v in shape[2:]):
        raise ValueError('K-quantized expert widths must align256')
    return shape, mtp


def plan_entries(reader, expected=None):
    shape, mtp = geometry(reader, expected)
    layers, experts, embedding, hidden = shape
    names = {t.name for t in reader.tensors}
    entries, selected, offset = [], set(), 0
    for t in reader.tensors:
        match = SUFFIX.fullmatch(t.name)
        replace = bool(match and int(match[1]) < layers)
        if replace:
            role = match[2]
            required = [hidden, embedding, experts] if role == 'down' else [embedding, hidden, experts]
            if t.shape != required or t.type_id not in BLOCK_BYTES:
                raise ValueError('Actual expert shape/format mismatch: ' + t.name)
            if embedding * hidden > 1048576:
                raise ValueError('This bounded expert service admits at most1048576 elements/expert')
            selected.add((int(match[1]), role))
        size = t.elements // 64 * 36 if replace else t.expected_bytes()
        offset = align(offset, reader.alignment)
        entries.append(dict(tensor=t, name=t.name, shape=t.shape, type=40 if replace else t.type_id,
                            offset=offset, size=size, replace=replace))
        offset += size
        if replace:
            scale_name = t.name[:-7] + '.scale'
            if scale_name in names:
                raise ValueError('Scale companion name collision: ' + scale_name)
            offset = align(offset, reader.alignment)
            entries.append(dict(tensor=t, name=scale_name, shape=[experts], type=0,
                                offset=offset, size=experts * 4, scale=True))
            offset += experts * 4
    if selected != {(layer, role) for layer in range(layers) for role in ('gate', 'up', 'down')}:
        raise ValueError('Incomplete main-stack routed FFN inventory')
    return entries, shape, mtp


def metadata_records(source, reader, source_identity, source_header_sha):
    if any(k.startswith('strata.moe.nvfp4.') for k in reader.metadata):
        raise ValueError('Source is already a converted MoE artifact')
    records = []
    with Path(source).open('rb') as stream:
        stream.seek(16)
        count, = struct.unpack('<Q', stream.read(8))
        for _ in range(count):
            start = stream.tell(); name = reader._str(stream); reader._value(stream); end = stream.tell()
            if name != 'general.file_type':
                stream.seek(start); records.append(stream.read(end - start))
    extras = {'strata.moe.nvfp4.scope': 'main_routed_experts',
              'strata.moe.nvfp4.algorithm': ALGORITHM,
              'strata.moe.nvfp4.scale_suffix': '.scale',
              'strata.source.gguf.sha256': source_identity['sha256'],
              'strata.source.gguf.header_sha256': source_header_sha,
              'strata.source.gguf.name': Path(source).name}
    if any(k in reader.metadata for k in extras):
        raise ValueError('Source conversion metadata collision')
    records += string_records(extras)
    from conversion_io import text
    for key, kind, value in [('strata.moe.nvfp4.version', 4, 1), ('strata.moe.nvfp4.scale_axis', 4, 2),
                             ('strata.source.gguf.bytes', 10, source_identity['bytes'])]:
        records.append(text(key) + struct.pack('<I', kind) + struct.pack('<I' if kind == 4 else '<Q', value))
    return records


def expert_chunks(source, offset, elements, type_id, chunk_blocks=1024):
    if not 1 <= chunk_blocks <= 8192:
        raise ValueError('CPU chunk bound1..8192 K-blocks')
    yield from source_chunks(source, offset, elements // 256 * BLOCK_BYTES[type_id],
                             chunk_blocks * BLOCK_BYTES[type_id])


def cpu_expert(source, offset, elements, type_id, write, source_hasher=None):
    maximum = 0.; original = hashlib.sha256()
    for raw in expert_chunks(source, offset, elements, type_id):
        original.update(raw)
        if source_hasher is not None:
            source_hasher.update(raw)
        maximum = max(maximum, float(np.max(np.abs(dequantize(raw, type_id)))))
    global_scale = np.float32(maximum / 2688 if maximum else 1.)
    stats = dict(error2=0., norm2=0., max_error=0., zero_scales=0, elements=0)
    second, packed_hash = hashlib.sha256(), hashlib.sha256()
    for raw in expert_chunks(source, offset, elements, type_id):
        second.update(raw); packed, row = quantize(dequantize(raw, type_id), global_scale)
        write(packed); packed_hash.update(packed)
        for key in stats:
            stats[key] = max(stats[key], row[key]) if key == 'max_error' else stats[key] + row[key]
    if second.digest() != original.digest():
        raise RuntimeError('Original expert changed between global-scale and quantization passes')
    return float(global_scale), stats, original.hexdigest(), packed_hash.hexdigest()


def convert(source, output, work, backend_name='cpu', gpu_library=None, expected=(40, 256, 2048, 512)):
    source, output, work = Path(source), Path(output), Path(work)
    original_identity, implementation = identity(source), implementation_identity()
    reader = GGUFFile(source); validate_reader(reader)
    if reader.data_start > 128 * 1024 * 1024:
        raise ValueError('MoE source header exceeds bounded128MiB')
    with source.open('rb') as stream:
        source_header = stream.read(reader.data_start)
    header_sha = hashlib.sha256(source_header).hexdigest()
    entries, shape, mtp = plan_entries(reader, expected)
    records = metadata_records(source, reader, original_identity, header_sha)
    raw_header, data_start = header(entries, records, reader.alignment)
    request = dict(algorithm=ALGORITHM, backend=backend_name, source_sha256=original_identity['sha256'],
                   source_bytes=original_identity['bytes'], source_name=source.name, geometry=list(shape), implementation=implementation)
    backend = None
    if backend_name == 'gpu':
        from gpu_conversion import GpuConverter
        request['gpu_library'] = identity(gpu_library); backend = GpuConverter(gpu_library)
    elif backend_name != 'cpu':
        raise ValueError('Backend must be cpu or gpu')
    prior = reuse(output, request)
    if prior:
        return prior
    work.mkdir(parents=True, exist_ok=True); output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = data_start + entries[-1]['offset'] + entries[-1]['size']
    if shutil.disk_usage(work).free < expected_bytes * 2 + (256 << 20):
        raise RuntimeError('Need space for complete partial plus safe copy-publication fallback on D')
    rows, scales = [], {}
    with Transaction(work, output) as transaction:
        transaction.status('converting-MoE', backend=backend_name, geometry=shape, expected_output_bytes=expected_bytes)
        with transaction.partial.open('xb') as destination:
            def pad(position):
                gap = position - destination.tell()
                if gap < 0:
                    raise RuntimeError('Overlapping converted tensor extents')
                if gap:
                    destination.write(bytes(gap))
            destination.write(raw_header); pad(data_start)
            for entry in entries:
                t = entry['tensor']; pad(data_start + entry['offset'])
                if entry.get('scale'):
                    destination.write(np.asarray(scales.pop(t.name), dtype='<f4').tobytes())
                    continue
                before, after = hashlib.sha256(), hashlib.sha256()
                if entry['replace']:
                    elements = t.shape[0] * t.shape[1]
                    old_stride = elements // 256 * BLOCK_BYTES[t.type_id]
                    global_scales, source_experts, packed_experts = [], [], []
                    stats = dict(error2=0., norm2=0., max_error=0., zero_scales=0, elements=0)
                    max_relative = 0.
                    for expert in range(shape[1]):
                        offset = reader.data_start + t.offset + expert * old_stride
                        def write(raw):
                            destination.write(raw); after.update(raw)
                        if backend:
                            raw = b''.join(source_chunks(source, offset, old_stride))
                            before.update(raw); packed, scale, result = backend.convert_raw(raw, elements, t.type_id)
                            write(packed); source_sha = hashlib.sha256(raw).hexdigest(); packed_sha = hashlib.sha256(packed).hexdigest()
                        else:
                            scale, result, source_sha, packed_sha = cpu_expert(source, offset, elements, t.type_id, write, before)
                        global_scales.append(scale); source_experts.append(source_sha); packed_experts.append(packed_sha)
                        max_relative = max(max_relative, math.sqrt(result['error2'] / (result['norm2'] + 1e-300)))
                        for key in stats:
                            stats[key] = max(stats[key], result[key]) if key == 'max_error' else stats[key] + result[key]
                    scales[t.name] = global_scales
                    scale_hash = hashlib.sha256(np.asarray(global_scales, dtype='<f4').tobytes()).hexdigest()
                    rows.append(dict(name=t.name, converted=True, source_type=t.type_id, source_bytes=t.expected_bytes(),
                        source_sha256=before.hexdigest(), converted_sha256=after.hexdigest(), scale_name=t.name[:-7]+'.scale',
                        scale_shape=[shape[1]], scale_sha256=scale_hash, expert_elements=elements,
                        source_expert_stride=old_stride, converted_expert_stride=elements//64*36,
                        source_expert_sha256=source_experts, converted_expert_sha256=packed_experts,
                        global_min=min(global_scales), global_max=max(global_scales),
                        relative_l2=math.sqrt(stats['error2']/(stats['norm2']+1e-300)), maximum_expert_relative_l2=max_relative, **stats))
                else:
                    for raw in source_chunks(source, reader.data_start + t.offset, entry['size']):
                        destination.write(raw); before.update(raw)
                    rows.append(dict(name=t.name, converted=False, source_type=t.type_id, source_bytes=entry['size'],
                                     source_sha256=before.hexdigest(), converted_sha256=before.hexdigest()))
                if destination.tell() != data_start + entry['offset'] + entry['size']:
                    raise RuntimeError('MoE converted extent mismatch: ' + t.name)
                transaction.status('converting-MoE', completed_tensors=len(rows), total_tensors=len(reader.tensors), last_tensor=t.name)
            destination.flush(); os.fsync(destination.fileno())
        validate_output(transaction.partial, entries, data_start)
        # One initial full source SHA only. Every tensor/expert read above is
        # hashed while processing; do not reread20GB merely for a second hash.
        final_source = source.stat()
        if (final_source.st_size, final_source.st_mtime_ns) != (original_identity['bytes'], original_identity['mtime_ns']):
            raise RuntimeError('Original GGUF changed during conversion')
        if implementation_identity() != implementation or (backend and identity(gpu_library) != request['gpu_library']):
            raise RuntimeError('Converter implementation/library changed during conversion')
        return transaction.finish(request, source=original_identity, geometry=shape, mtp_layers_preserved=mtp,
            backend=backend_name, converted_tensors=3*shape[0], tensor_results=rows,
            source_header_sha256=header_sha, retained_source_metadata_records_sha256=hashlib.sha256(b''.join(records[:-9])).hexdigest(),
            output_header_sha256=hashlib.sha256(raw_header+bytes(data_start-len(raw_header))).hexdigest(),
            format_version=1, scale_contract='sameGGUF companion F32[E],axis2,expertID', full_model_loads=0)
