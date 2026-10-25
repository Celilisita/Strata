"""Explicit original GGUF -> per-expert NVFP4 companion GGUF; no model loading."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--gpu-library', type=Path)
    args = parser.parse_args(argv)
    source, output, work = args.source.resolve(strict=True), args.output.resolve(), args.work_dir.resolve()
    library = args.gpu_library.resolve(strict=True) if args.gpu_library else None
    if not source.is_file() or source.suffix.lower() != '.gguf' or output.suffix.lower() != '.gguf':
        raise ValueError('Source/output must be GGUF files')
    if os.name == 'nt' and output.drive.upper() != 'D:':
        raise ValueError('5060 converted model output must be on D:')
    from conversion_io import receipt_path, setup_work_dir
    if len({str(p).casefold() for p in (source, output, receipt_path(output))}) != 3:
        raise ValueError('Source/output/receipt must be distinct')
    if args.backend == 'gpu' and library is None:
        raise ValueError('--backend gpu requires explicit --gpu-library; no automatic build')
    setup_work_dir(work)
    from moe_conversion import convert
    receipt = convert(source, output, work, args.backend, library)
    print(json.dumps({k:receipt[k] for k in ('passed','algorithm','output','bytes','sha256','backend',
                     'converted_tensors','geometry','runtime_qualified','full_model_loads')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__,error=str(error))), file=sys.stderr)
        raise SystemExit(1)
