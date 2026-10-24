"""Compare CPU/device logits and repeated rounds emitted by model-check.cpp."""

import argparse
import json
from pathlib import Path
import struct

import numpy as np


def read(path):
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"incomplete logits header: {path}")
        vocab, steps, rounds = struct.unpack("<iii", header)
        if vocab <= 0 or steps <= 0 or rounds < 2:
            raise ValueError(f"invalid logits shape: {path}")
        if path.stat().st_size != 12 + 4 * vocab * steps * rounds:
            raise ValueError(f"incomplete or extra logits data: {path}")
        values = np.fromfile(stream, dtype="<f4")
    return values.reshape(rounds, steps, vocab).astype(np.float64)


def compare(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    result = {"shape": list(reference.shape), "finite": finite, "nmse_limit": 1e-5,
              "repeat_nmse_limit": 1e-6, "passed": False}
    if not finite:
        result["reason"] = "non-finite logits"
        return result
    nmse = np.mean((reference - candidate) ** 2, axis=-1) / np.maximum(np.mean(reference ** 2, axis=-1), 1e-30)
    repeat = np.mean((candidate[1:] - candidate[0]) ** 2, axis=-1) / np.maximum(
        np.mean(candidate[0] ** 2, axis=-1), 1e-30)
    matches = int(np.sum(reference.argmax(-1) == candidate.argmax(-1)))
    total = int(np.prod(reference.shape[:2]))
    result.update(nmse_per_step=nmse.tolist(), max_abs_error=float(np.max(np.abs(reference - candidate))),
                  top1_matches=matches, top1_total=total, device_repeat_nmse=repeat.max(axis=0).tolist(),
                  cpu_repeat_max_abs=float(np.max(np.abs(reference[1:] - reference[0]))))
    result["passed"] = bool((nmse < result["nmse_limit"]).all() and matches == total and
                            (repeat < result["repeat_nmse_limit"]).all())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = compare(read(args.reference), read(args.candidate))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
