#!/usr/bin/env python3
"""Measure RTFx for the converted window across every CoreML compute unit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np

UNITS = {
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "ALL": ct.ComputeUnit.ALL,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)

    print(f"{'compute units':<14} {'load (s)':>9} {'mean (ms)':>10} {'p50 (ms)':>9} {'RTFx':>9}")
    for name, unit in UNITS.items():
        started = time.perf_counter()
        model = ct.models.MLModel(str(args.mlpackage), compute_units=unit)
        spec = model.get_spec().description.input[0]
        frames = int(spec.type.multiArrayType.shape[1])
        dim = int(spec.type.multiArrayType.shape[2])
        load_seconds = time.perf_counter() - started

        # 512 stacked frames == 1024 hops == 10.24 s of audio.
        window_seconds = frames * 2 * 160 / 16_000
        features = rng.standard_normal((1, frames, dim), dtype=np.float32)

        for _ in range(args.warmup):
            model.predict({"input_features": features})

        timings = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            model.predict({"input_features": features})
            timings.append(time.perf_counter() - started)

        mean = float(np.mean(timings))
        p50 = float(np.median(timings))
        print(
            f"{name:<14} {load_seconds:>9.2f} {mean * 1000:>10.2f} "
            f"{p50 * 1000:>9.2f} {window_seconds / mean:>9.1f}"
        )


if __name__ == "__main__":
    main()
