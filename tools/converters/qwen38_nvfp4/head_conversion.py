"""Independent Q6_K head: original two CPU passes, one whole-head global scale."""
from pathlib import Path
import hashlib
import math
import os
import struct
import numpy as np
from gguf_reader import GGUFFile
from quantize_nvfp4 import dequantize, quantize
from conversion_io import (ALGORITHM, Transaction, align, header, identity,
                           implementation_identity, reuse, source_chunks, string_records,
                           validate_output, validate_reader, verify_source)

# A logical label is retained only for this exact historical four-part identity.
# External file names may change; these constants never replace hashing real input.
HISTORICAL_MAIN = '0611180fd2368fa914739b4037435dad27cff76118b8036b0d770814a2079d46'
HISTORICAL_HEADER = '289085adcaca7960964ce6a11da0c5587acf75a0b4ac39e1b3123c29e6745f3b'
HISTORICAL_Q6_HEAD = '2155d6e1f7c5549077438c4760e6e150945dd5e406484b29162dff7b03d05da0'
HISTORICAL_NORM = 'b3ba4f3231a21b1559f52115d3bffc7c28dcd03e4be22df269ef97a3f5d4bfbf'
HISTORICAL_NAME = 'Qwen3.8-27B-original-derived-NVFP4.gguf'
HEAD_RULE = ('Q6_K F32 decode; one whole-head global=max/(6*448); '
             'block16 E4M3 three-neighbor MSE; E2M1 RNE')


def convert_head(source, output, work, main_receipt):
    source, output = Path(source), Path(output)
    source_identity = identity(source)
    if (source_identity['sha256'], source_identity['bytes']) != (main_receipt['sha256'], main_receipt['bytes']):
        raise ValueError('Head source is not the fully verified main conversion output')
    implementation = implementation_identity()
    request = dict(algorithm=ALGORITHM, kind='independent-head', backend='cpu',
                   source_sha256=source_identity['sha256'], source_bytes=source_identity['bytes'],
                   implementation=implementation)
    previous = reuse(output, request)
    if previous:
        return previous
    reader = GGUFFile(source)
    validate_reader(reader)
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    head, norm = by_name['output.weight'], by_name['output_norm.weight']
    if (head.shape, head.type_id, head.expected_bytes()) != ([5120, 248320], 14, 1042944000):
        raise ValueError('Expected original Q6_K output.weight [5120,248320]')
    if (norm.shape, norm.type_id, norm.expected_bytes()) != ([5120], 0, 20480):
        raise ValueError('Expected original F32 output_norm.weight [5120]')
    with source.open('rb') as original:
        header_hash = hashlib.sha256(original.read(reader.data_start)).hexdigest()
        original.seek(reader.data_start + norm.offset)
        norm_raw = original.read(norm.expected_bytes())
    if len(norm_raw) != 20480 or not np.isfinite(np.frombuffer(norm_raw, dtype='<f4')).all():
        raise ValueError('Invalid source norm payload')
    norm_hash = hashlib.sha256(norm_raw).hexdigest()
    with Transaction(work, output) as transaction:
        transaction.status('head-first-pass-global-max', elements=head.elements)
        maximum, blocks = 0., 0
        head_hash = hashlib.sha256()
        for index, raw in enumerate(source_chunks(source, head, reader.data_start)):
            head_hash.update(raw)
            maximum = max(maximum, float(np.max(np.abs(dequantize(raw, 14)))))
            blocks += len(raw) // 210
            if index % 64 == 0:
                transaction.status('head-first-pass-global-max', blocks=blocks, elements=head.elements)
        if blocks != head.elements // 256 or not math.isfinite(maximum):
            raise RuntimeError('Incomplete or nonfinite original head')
        global_scale = np.float32(maximum / (6 * 448) if maximum else 1.)
        source_head_sha = head_hash.hexdigest()
        historical = (source_identity['sha256'] == HISTORICAL_MAIN and
                      header_hash == HISTORICAL_HEADER and source_head_sha == HISTORICAL_Q6_HEAD and
                      norm_hash == HISTORICAL_NORM)
        logical_name = HISTORICAL_NAME if historical else source.name
        # Exact historical strings/order reproduce its container only after exact
        # identity verification. No new metadata fields are inserted into GGUF.
        metadata = {
            'general.name': 'Strata original-derived Q6 head to NVFP4 operator fixture',
            'strata.head_fp4.source_model': logical_name,
            'strata.head_fp4.source_model_sha256': source_identity['sha256'],
            'strata.head_fp4.source_head_sha256': source_head_sha,
            'strata.head_fp4.source_header_sha256': header_hash,
            'strata.head_fp4.source_norm_sha256': norm_hash,
            'strata.head_fp4.rule': HEAD_RULE}
        records = string_records(metadata)
        from conversion_io import text
        records.append(text('general.alignment') + struct.pack('<II', 4, reader.alignment))
        entries, offset = [], 0
        for name, shape, kind, size in [
                ('output.weight', [5120, 248320], 40, head.elements // 64 * 36),
                ('output.scale', [1], 0, 4), ('output_norm.weight', [5120], 0, 20480)]:
            offset = align(offset, reader.alignment)
            entries.append(dict(name=name, shape=shape, type=kind, offset=offset, size=size))
            offset += size
        raw_header, data_start = header(entries, records, reader.alignment)
        statistics = dict(error2=0., norm2=0., max_error=0., zero_scales=0, elements=0)
        second_hash, packed_hash = hashlib.sha256(), hashlib.sha256()
        written = 0
        transaction.status('head-second-pass-fixed-global', global_scale=float(global_scale))
        with transaction.partial.open('xb') as destination:
            destination.write(raw_header)
            destination.write(bytes(data_start - len(raw_header)))
            for index, raw in enumerate(source_chunks(source, head, reader.data_start)):
                second_hash.update(raw)
                packed, result = quantize(dequantize(raw, 14), global_scale)
                destination.write(packed)
                packed_hash.update(packed)
                written += len(packed)
                for key in statistics:
                    statistics[key] = (max(statistics[key], result[key]) if key == 'max_error'
                                       else statistics[key] + result[key])
                if index % 64 == 0:
                    transaction.status('head-second-pass-fixed-global', written_bytes=written,
                                       global_scale=float(global_scale))
            if written != entries[0]['size'] or second_hash.hexdigest() != source_head_sha:
                raise RuntimeError('Original head changed between the two CPU passes')
            for entry, raw in [(entries[1], struct.pack('<f', global_scale)), (entries[2], norm_raw)]:
                gap = data_start + entry['offset'] - destination.tell()
                if gap < 0:
                    raise RuntimeError('Head tensor extents overlap')
                destination.write(bytes(gap))
                destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        validate_output(transaction.partial, entries, data_start)
        verify_source(source, source_identity)
        if implementation_identity() != implementation:
            raise RuntimeError('Converter source files changed during conversion')
        return transaction.finish(
            request, source=source_identity, logical_source_model=logical_name,
            historical_logical_name_verified=historical, source_header_sha256=header_hash,
            source_head=dict(name=head.name, shape=head.shape, type=14, offset=head.offset,
                             bytes=head.expected_bytes(), sha256=source_head_sha),
            source_norm_sha256=norm_hash, entries=entries, packed_weight_sha256=packed_hash.hexdigest(),
            global_scale=float(global_scale), block_chunks=16384, statistics=statistics,
            relative_l2=math.sqrt(statistics['error2'] / (statistics['norm2'] + 1e-300)),
            source_model_rehashed=True, source_head_hashed_twice=True, CPU_only=True,
            full_model_loads=0)
