"""GGUF I/O and exclusive, verified publication; no quantization policy here."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import uuid

ALGORITHM = 'strata-q4km-nvfp4-three-neighbor-v1'
BUFFER_BYTES = 16 * 1024 * 1024


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for raw in iter(lambda: stream.read(BUFFER_BYTES), b''):
            digest.update(raw)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    initial = path.stat()
    digest = sha_file(path)
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise RuntimeError('File changed during full SHA verification: ' + str(path))
    return dict(path=str(path), bytes=final.st_size, sha256=digest,
                mtime_ns=final.st_mtime_ns)


def verify_source(path, expected):
    actual = identity(path)
    if (actual['bytes'], actual['sha256']) != (expected['bytes'], expected['sha256']):
        raise RuntimeError('Source bytes changed during conversion: ' + str(path))


def setup_work_dir(path):
    if os.name != 'nt':
        raise RuntimeError('Conversion execution requires Windows and a Y:\\temp work directory')
    path = Path(path).resolve()
    if path.drive.upper() != 'Y:' or len(path.parts) < 2 or path.parts[1].lower() != 'temp':
        raise ValueError('--work-dir must resolve inside Y:\\temp')
    path.mkdir(parents=True, exist_ok=True)
    # Resolve again after creation so a junction cannot redirect temporary files.
    path = path.resolve()
    if path.drive.upper() != 'Y:' or path.parts[1].lower() != 'temp':
        raise ValueError('Work directory resolves outside Y:\\temp')
    cache = path / 'cuda-cache'
    cache.mkdir(exist_ok=True)
    os.environ.update(TEMP=str(path), TMP=str(path), TMPDIR=str(path),
                      TEMPDIR=str(path), CUDA_CACHE_PATH=str(cache),
                      PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    os.chdir(path)
    return path


def text(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def string_records(values):
    return [text(key) + struct.pack('<I', 8) + text(value)
            for key, value in values.items()]


def header(entries, records, alignment):
    descriptors = b''.join(
        text(entry['name']) + struct.pack('<I', len(entry['shape'])) +
        struct.pack('<' + 'Q' * len(entry['shape']), *entry['shape']) +
        struct.pack('<IQ', entry['type'], entry['offset']) for entry in entries)
    raw = struct.pack('<IIQQ', 0x46554747, 3, len(entries), len(records))
    raw += b''.join(records) + descriptors
    return raw, align(len(raw), alignment)


def validate_reader(reader):
    alignment = reader.alignment
    if alignment <= 0 or alignment > 4096 or alignment & (alignment - 1):
        raise ValueError('Unsupported GGUF alignment')
    names, extents = set(), []
    size = reader.path.stat().st_size
    for tensor in reader.tensors:
        count = tensor.expected_bytes()
        if tensor.name in names or not tensor.shape or any(d <= 0 for d in tensor.shape):
            raise ValueError('Duplicate name or invalid tensor dimensions: ' + tensor.name)
        names.add(tensor.name)
        if count is None or tensor.offset % alignment:
            raise ValueError('Unsupported tensor format/alignment: ' + tensor.name)
        end = reader.data_start + tensor.offset + count
        if end > size:
            raise ValueError('Truncated tensor: ' + tensor.name)
        extents.append((tensor.offset, tensor.offset + count))
    extents.sort()
    if any(left[1] > right[0] for left, right in zip(extents, extents[1:])):
        raise ValueError('Overlapping GGUF tensor extents')


def validate_output(path, entries, data_start):
    from gguf_reader import GGUFFile
    reader = GGUFFile(path)
    validate_reader(reader)
    if reader.data_start != data_start or len(reader.tensors) != len(entries):
        raise RuntimeError('Converted GGUF directory size mismatch')
    for entry, tensor in zip(entries, reader.tensors):
        actual = (tensor.name, tensor.shape, tensor.type_id, tensor.offset, tensor.expected_bytes())
        expected = (entry['name'], entry['shape'], entry['type'], entry['offset'], entry['size'])
        if actual != expected:
            raise RuntimeError('Converted GGUF descriptor mismatch: ' + entry['name'])
    if Path(path).stat().st_size != data_start + entries[-1]['offset'] + entries[-1]['size']:
        raise RuntimeError('Converted GGUF final extent mismatch')


def source_chunks(source, tensor, data_start):
    block_size = {12: 144, 14: 210}[tensor.type_id]
    remaining = tensor.elements // 256
    with Path(source).open('rb') as stream:
        stream.seek(data_start + tensor.offset)
        while remaining:
            count = min(16384, remaining)
            raw = stream.read(count * block_size)
            if len(raw) != count * block_size:
                raise ValueError('Truncated original tensor: ' + tensor.name)
            yield raw
            remaining -= count


def implementation_identity():
    root = Path(__file__).resolve().parent
    files = ['convert_q4_k_m_to_nvfp4.py', 'conversion_io.py', 'main_conversion.py',
             'head_conversion.py', 'quantize_nvfp4.py', 'gguf_reader.py', 'gpu_conversion.py',
             'gpu/convert_gpu.cu', 'gpu/codec.cuh',
             'gpu/include/strata/qwen35/native/quant.hpp']
    return {name: sha_file(root / name) for name in files}


def receipt_path(output):
    return Path(output).with_suffix('.conversion.json')


def reuse(output, request):
    output, report = Path(output), receipt_path(output)
    if not output.exists() and not report.exists():
        return None
    if not output.is_file() or not report.is_file():
        raise RuntimeError('Existing target/receipt pair is incomplete; refusing overwrite: ' + str(output))
    receipt = json.loads(report.read_text(encoding='utf-8-sig'))
    actual = identity(output)  # Reuse always verifies full contents, never just size/mtime.
    if not receipt.get('passed') or receipt.get('request') != request:
        raise RuntimeError('Existing receipt belongs to another conversion request: ' + str(report))
    if (actual['bytes'], actual['sha256']) != (receipt['bytes'], receipt['sha256']):
        raise RuntimeError('Existing output full SHA mismatch: ' + str(output))
    return receipt


def exclusive_publish(source, destination, expected_sha):
    """Cross-volume publication is a copy, not an atomic rename. No overwrite."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, BUFFER_BYTES)
        dst.flush()
        os.fsync(dst.fileno())
    if sha_file(destination) != expected_sha:
        raise RuntimeError('Published output full SHA mismatch; inspect retained work files: ' + str(destination))


class Transaction:
    def __init__(self, work, output):
        self.directory = Path(work) / ('conversion-' + uuid.uuid4().hex)
        self.directory.mkdir()
        self.partial = self.directory / (Path(output).name + '.partial')
        self.output = Path(output)
        self.started = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.status('failed', error_type=kind.__name__, error=str(error),
                        partial=str(self.partial), partial_preserved=self.partial.exists())
        return False

    def status(self, phase, **fields):
        payload = dict(phase=phase, output=str(self.output),
                       wall_s=time.perf_counter() - self.started, **fields)
        temp = self.directory / 'status.next.json'
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        os.replace(temp, self.directory / 'status.json')

    def finish(self, request, **fields):
        result = identity(self.partial)
        receipt = dict(algorithm=ALGORITHM, passed=True, request=request,
                       output=str(self.output), bytes=result['bytes'], sha256=result['sha256'],
                       wall_s=time.perf_counter() - self.started,
                       publication='exclusive-copy-fsync-full-sha256; cross-volume copy is not atomic',
                       runtime_qualified=False, **fields)
        staged_receipt = self.directory / 'receipt.json'
        staged_receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        if receipt_path(self.output).exists():
            raise RuntimeError('Receipt appeared during conversion; refusing publication')
        exclusive_publish(self.partial, self.output, result['sha256'])
        exclusive_publish(staged_receipt, receipt_path(self.output), sha_file(staged_receipt))
        self.status('complete', bytes=result['bytes'], sha256=result['sha256'])
        # Success frees large Ytemp partials; failures retain them for explicit recovery.
        self.partial.unlink()
        return receipt
