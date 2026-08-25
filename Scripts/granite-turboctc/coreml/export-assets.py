#!/usr/bin/env python3
"""Emit the Swift-side runtime assets and a parity fixture.

Writes:
  mel_filters.bin  -- (n_mels, n_freqs) float32, row-major, matching torchaudio
  reference.json   -- features + expected token ids for a known wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

import coreml_patches

DEFAULT_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"


def export_mel_filters(processor, output_dir: Path) -> tuple[int, int]:
    extractor = processor.feature_extractor
    # torchaudio stores the bank as (n_freqs, n_mels); Swift multiplies
    # (n_mels x n_freqs) by the power spectrum, so transpose on the way out.
    bank = extractor.mel_filters.mel_scale.fb.detach().cpu().numpy()
    bank = np.ascontiguousarray(bank.T.astype(np.float32))
    (output_dir / "mel_filters.bin").write_bytes(bank.tobytes())
    return bank.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--audio", type=Path, default=Path("audio/yc_first_minute_16k.wav"))
    parser.add_argument("--reference-seconds", type=float, default=10.24)
    parser.add_argument("--window-frames", type=int, default=512)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coreml_patches.apply()
    processor = AutoProcessor.from_pretrained(args.model_id)

    shape = export_mel_filters(processor, args.output_dir)
    print(f"mel_filters.bin: {shape[0]} mels x {shape[1]} freq bins")

    sample_rate = processor.feature_extractor.sampling_rate
    audio, rate = sf.read(str(args.audio), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    assert rate == sample_rate, f"{args.audio} is {rate} Hz"
    audio = audio[: int(args.reference_seconds * sample_rate)]

    inputs = processor([audio], sampling_rate=sample_rate)
    features = np.asarray(inputs["input_features"], dtype=np.float32)[0]

    model = AutoModelForCTC.from_pretrained(args.model_id, dtype=torch.float32).eval()

    # Run the SAME fixed-window schedule the Swift manager uses. A single
    # full-length forward pass is not equivalent: the subsampling convolutions
    # and self-conditioning cross block boundaries, so an unwindowed reference
    # disagrees with windowed inference by a token or two at each seam.
    window_frames = args.window_frames
    subsample = 4
    token_ids: list[int] = []
    for start in range(0, features.shape[0], window_frames):
        chunk = features[start : start + window_frames]
        valid = chunk.shape[0]
        if valid < window_frames:
            chunk = np.pad(chunk, ((0, window_frames - valid), (0, 0)))
        with torch.no_grad():
            logits = model(input_features=torch.from_numpy(chunk).unsqueeze(0)).logits
        frame_ids = logits.argmax(dim=-1)[0].tolist()
        token_ids.extend(frame_ids[: -(-valid // subsample)])

    reference = {
        "audio": args.audio.name,
        "sample_rate": sample_rate,
        "samples": int(len(audio)),
        "feature_shape": list(features.shape),
        "window_frames": window_frames,
        # Only a slice is stored; enough to catch a front-end bug, small enough to read.
        "features_head": features[:2, :8].tolist(),
        "features_tail": features[-2:, :8].tolist(),
        "feature_checksum": float(np.abs(features).sum()),
        "token_ids": token_ids,
    }
    (args.output_dir / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"reference.json: {features.shape[0]} frames -> {len(token_ids)} tokens")


if __name__ == "__main__":
    main()
