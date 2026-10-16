"""One synthetic mHC protocol canary; never counts as real-model correctness.

Uses the frozen official Block.hc_pre/hc_post functions unchanged. The TileLang
Sinkhorn kernel's documented arithmetic is evaluated with Torch on CPU, since
this local preparation host is not the designated inference GPU.
"""
import argparse
import ast
import hashlib
import json
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace


def sinkhorn(mixes, scale, base, hc, iterations, epsilon):
    import torch

    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + epsilon
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    combination = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).reshape(1, 1, hc, hc)
    combination = combination.softmax(-1) + epsilon
    combination /= combination.sum(-2, keepdim=True) + epsilon
    for _ in range(iterations - 1):
        combination /= combination.sum(-1, keepdim=True) + epsilon
        combination /= combination.sum(-2, keepdim=True) + epsilon
    return pre, post, combination


def official_functions(path):
    import torch
    import torch.nn.functional as F

    tree = ast.parse(path.read_text())
    block = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "Block")
    methods = [item for item in block.body if isinstance(item, ast.FunctionDef) and item.name in ("hc_pre", "hc_post")]
    if len(methods) != 2:
        raise ValueError("frozen reference Block functions not found")
    namespace = {"torch": torch, "F": F, "hc_split_sinkhorn": sinkhorn}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["hc_pre"], namespace["hc_post"]


def main(args):
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(123456)
    dim, width = 4096, 4 * 4096
    context = SimpleNamespace(norm_eps=1e-20, hc_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20)
    residual = (torch.randn(1, 1, 4, dim) * 0.7).bfloat16()
    projection = torch.randn(24, width) * 0.003
    scales = torch.tensor([0.8, 1.1, 0.9])
    bases = torch.randn(24) * 0.25
    sublayer = torch.randn(1, 1, dim).bfloat16()
    hc_pre, hc_post = official_functions(args.reference_model)
    reduced, post, combination = hc_pre(context, residual, projection, scales, bases)
    expected_output = hc_post(context, sublayer, residual, post, combination)
    normalized_projection = torch.nn.functional.linear(residual.flatten(2).float(), projection)
    normalized_projection *= torch.rsqrt(residual.flatten(2).float().square().mean(-1, keepdim=True) + context.norm_eps)
    pre, _, _ = sinkhorn(normalized_projection, scales, bases, 4, 20, 1e-6)
    arrays = (residual, projection, scales, bases, sublayer, pre, post, combination, reduced, expected_output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture = args.output_dir / "synthetic-mhc-preflight.bin"
    with fixture.open("wb") as handle:
        handle.write(b"DSV4HC01" + struct.pack("<IIff", dim, 20, 1e-20, 1e-6))
        for array in arrays:
            handle.write(array.float().contiguous().numpy().astype("<f4").tobytes())
    result = subprocess.run([str(args.benchmark.resolve()), str(fixture.resolve())],
                            capture_output=True, text=True, check=True)
    measurement = json.loads(result.stdout)
    report = {"origin": "synthetic_protocol_preflight", "real_model_gate_passed": False,
              "oracle": "official_Block_functions_with_CPU_Torch_Sinkhorn_translation",
              "reference_model_sha256": hashlib.sha256(args.reference_model.read_bytes()).hexdigest(),
              "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
              "measurement": measurement}
    (args.output_dir / "synthetic-mhc-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
