"""Explicit main+independent-head conversion entry. --help never imports CUDA/NumPy."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import json


def parser():
    result = argparse.ArgumentParser(
        description='Convert Qwen35 Q4_K_M main-layer matrices to NVFP4 and write an independent head.',
        epilog='No automatic CUDA compilation. GPU applies to main matrices; head always uses the original CPU two-pass algorithm.')
    result.add_argument('--source', required=True, type=Path, help='Original Q4_K_M GGUF; read-only')
    result.add_argument('--output', required=True, type=Path, help='New main GGUF; existing targets never overwritten')
    result.add_argument('--head-output', required=True, type=Path, help='New independent three-tensor head GGUF')
    result.add_argument('--work-dir', required=True, type=Path, help=r'All temporary files, partials and caches inside Y:\temp')
    result.add_argument('--backend', required=True, choices=('cpu', 'gpu'), help='Main conversion backend; head remains CPU')
    result.add_argument('--gpu-library', type=Path,
                        default=Path(__file__).resolve().parent / 'bin' / 'convert_gpu.dll',
                        help='Explicit prebuilt GPU DLL, used only with --backend gpu')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    # Resolve paths before setup changes cwd. Parsing/help does not create files.
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    head_output = args.head_output.resolve()
    work = args.work_dir.resolve()
    gpu_library = args.gpu_library.resolve()
    if not source.is_file():
        raise ValueError('--source must be a regular GGUF file')
    from conversion_io import receipt_path, setup_work_dir
    names = [source, output, head_output, receipt_path(output), receipt_path(head_output)]
    if len({str(path).casefold() for path in names}) != len(names):
        raise ValueError('Source, both outputs and both receipt paths must be distinct')
    for target in (output, head_output):
        report = receipt_path(target)
        if target.exists() != report.exists():
            raise ValueError('Incomplete existing target/receipt pair; inspect without overwriting: ' + str(target))
        if target.exists() and (not target.is_file() or not report.is_file()):
            raise ValueError('Output and receipt must be regular files: ' + str(target))
    if args.backend == 'gpu' and not gpu_library.is_file():
        raise ValueError('GPU DLL does not exist; build separately or select --backend cpu')
    work = setup_work_dir(work)
    # Imports occur after TEMP/TMP/cwd/cache are confined to Ytemp.
    from main_conversion import convert_main
    from head_conversion import convert_head
    main_receipt = convert_main(source, output, work, args.backend, gpu_library)
    head_receipt = convert_head(output, head_output, work, main_receipt)
    summary = dict(passed=True, algorithm=main_receipt['algorithm'],
                   main={key: main_receipt[key] for key in ('output', 'bytes', 'sha256', 'backend', 'converted_tensors')},
                   head={key: head_receipt[key] for key in ('output', 'bytes', 'sha256', 'CPU_only', 'global_scale')},
                   receipts=[str(receipt_path(output)), str(receipt_path(head_output))],
                   work_dir=str(work), runtime_qualified=False, full_model_loads=0)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(passed=False, error_type=type(error).__name__, error=str(error))), file=sys.stderr)
        raise SystemExit(1)
