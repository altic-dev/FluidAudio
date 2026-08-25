#!/usr/bin/env python3
"""Verify the CoreML export reproduces the PyTorch model token-for-token.

Both paths are fed the *same* log-mel features, so any difference is attributable
to the conversion alone rather than to front-end drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor

import coreml_patches

BLANK_ID = 0


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != sample_rate:
        raise SystemExit(f"{path} is {rate} Hz, expected {sample_rate} Hz")
    return audio


def ctc_collapse(token_ids: np.ndarray) -> list[int]:
    """Standard greedy CTC squash: drop repeats, then drop blanks."""
    collapsed: list[int] = []
    previous = -1
    for token in token_ids.tolist():
        if token != previous and token != BLANK_ID:
            collapsed.append(token)
        previous = token
    return collapsed


def windows(features: np.ndarray, frames: int):
    """Split (T, 320) features into fixed windows, zero-padding the tail."""
    total = features.shape[0]
    for start in range(0, total, frames):
        chunk = features[start : start + frames]
        valid = chunk.shape[0]
        if valid < frames:
            chunk = np.pad(chunk, ((0, frames - valid), (0, 0)))
        yield chunk, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="ibm-granite/granite-speech-5.0-470m-turboctc")
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--audio", type=Path, default=Path("audio/yc_first_minute_16k.wav"))
    args = parser.parse_args()

    coreml_patches.apply()
    processor = AutoProcessor.from_pretrained(args.model_id)
    torch_model = AutoModelForCTC.from_pretrained(args.model_id, dtype=torch.float32).eval()

    sample_rate = processor.feature_extractor.sampling_rate
    audio = load_audio(args.audio, sample_rate)

    inputs = processor([audio], sampling_rate=sample_rate)
    features = np.asarray(inputs["input_features"], dtype=np.float32)[0]

    mlmodel = ct.models.MLModel(str(args.mlpackage), compute_units=ct.ComputeUnit.ALL)
    spec_input = mlmodel.get_spec().description.input[0]
    window_frames = int(spec_input.type.multiArrayType.shape[1])
    subsample = 4

    print(f"audio: {len(audio) / sample_rate:.2f}s -> {features.shape[0]} stacked frames")
    print(f"window: {window_frames} frames -> {window_frames // subsample} encoder frames\n")

    torch_ids: list[int] = []
    coreml_ids: list[int] = []
    matched = 0
    compared = 0

    for index, (chunk, valid) in enumerate(windows(features, window_frames)):
        batch = torch.from_numpy(chunk).unsqueeze(0)
        with torch.no_grad():
            logits = torch_model(input_features=batch).logits
        reference = logits.argmax(dim=-1)[0].numpy().astype(np.int32)

        prediction = mlmodel.predict({"input_features": chunk[None, ...]})
        candidate = np.asarray(prediction["token_ids"], dtype=np.int32).reshape(-1)

        # Only the frames backed by real audio are meaningful in the padded tail.
        valid_frames = min(len(reference), max(1, -(-valid // subsample)))
        reference = reference[:valid_frames]
        candidate = candidate[:valid_frames]

        agree = int((reference == candidate).sum())
        matched += agree
        compared += valid_frames
        torch_ids.extend(reference.tolist())
        coreml_ids.extend(candidate.tolist())

        status = "ok" if agree == valid_frames else f"{valid_frames - agree} mismatched"
        print(f"  window {index}: {valid_frames} frames, {status}")

    # Decode with the raw tokenizer, NOT processor.batch_decode: the latter runs its
    # own CTC squash, which collapses a second time and silently deletes genuine
    # repeated words. Blank-separated duplicates like [371, 0, 0, 371] ("had had")
    # are real repeats and must survive.
    tokenizer = processor.tokenizer
    torch_text = tokenizer.decode(ctc_collapse(np.array(torch_ids))).strip().lower()
    coreml_text = tokenizer.decode(ctc_collapse(np.array(coreml_ids))).strip().lower()

    print(f"\nframe-level token parity: {matched}/{compared} ({100.0 * matched / compared:.3f}%)")
    print(f"text identical: {torch_text == coreml_text}")
    print(f"\n[pytorch] {torch_text}")
    print(f"\n[coreml ] {coreml_text}")


if __name__ == "__main__":
    main()
