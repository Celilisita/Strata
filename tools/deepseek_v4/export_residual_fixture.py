"""Export an official-reference mHC checkpoint, retaining explicit identity and provenance.

The .npz must contain actual reference activations and expected outputs, not candidate
outputs. This does not run inference, read a model payload, or define a second oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

ARRAYS = ("residual", "projection", "scales", "bases", "sublayer",
          "expected_pre", "expected_post", "expected_combination",
          "expected_reduced", "expected_output")
REQUIRED_IDENTITY = ("model", "reference_source", "input_token_ids", "layer", "sublayer")


def main(args):
    import numpy as np

    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    missing = set(REQUIRED_IDENTITY) - identity.keys()
    if missing:
        raise ValueError(f"missing checkpoint identity fields: {sorted(missing)}")
    if identity.get("origin") != "official_reference_checkpoint":
        raise ValueError("requires an official_reference_checkpoint origin")
    if not 1 <= args.dim <= 16384 or not 1 <= args.iterations <= 100:
        raise ValueError("invalid mHC geometry")
    width = args.dim * 4
    shapes = ((4, args.dim), (24, width), (3,), (24,), (args.dim,),
              (4,), (4,), (4, 4), (args.dim,), (4, args.dim))
    digest = hashlib.sha256()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(args.checkpoint, allow_pickle=False) as checkpoint, args.output.open("wb") as output:
        header = b"DSV4HC01" + struct.pack("<IIff", args.dim, args.iterations, args.norm_epsilon, args.hc_epsilon)
        output.write(header)
        digest.update(header)
        for name, shape in zip(ARRAYS, shapes):
            array = checkpoint[name]
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid checkpoint array: {name}, expected {shape}")
            raw = np.asarray(array, dtype="<f4", order="C").tobytes()
            output.write(raw)
            digest.update(raw)
    identity.update({"fixture_schema": "DSV4HC01", "fixture_sha256": digest.hexdigest(),
                     "dim": args.dim, "streams": 4, "sinkhorn_iterations": args.iterations,
                     "norm_epsilon": args.norm_epsilon, "hc_epsilon": args.hc_epsilon})
    sidecar = args.output.with_suffix(args.output.suffix + ".identity.json")
    sidecar.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": digest.hexdigest(), "identity": str(sidecar)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--norm-epsilon", type=float, default=1e-20)
    parser.add_argument("--hc-epsilon", type=float, default=1e-6)
    main(parser.parse_args())
