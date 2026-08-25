#!/usr/bin/env python3
"""Export IBM Granite Speech 5.0 470M TurboCTC (encoder + CTC head) to CoreML.

The architecture is identical between the Apache-2.0 and the CC-BY-NC-SA "-nc"
checkpoints, so the same script converts both -- only ``--model-id`` changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCTC, AutoProcessor

import coreml_patches

AUTHOR = "Fluid Inference"

APACHE_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"
NC_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc-nc"

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
STACK_FACTOR = 2
FEATURE_DIM = 320
ENCODER_SUBSAMPLE = 4  # on top of the stack-by-2 already done in the front-end
CONTEXT_SIZE = 128  # block-attention block size

# The encoder right-pads each attention block to a multiple of CONTEXT_SIZE and
# builds a bool padding mask to suppress the pad -- which lowers to a
# `bitwise_not` that CoreML rejects. Attention runs at L, L/2 and L/4, so
# aligning L to CONTEXT_SIZE * ENCODER_SUBSAMPLE makes num_padded zero at every
# layer, the mask stays None, and the whole branch drops out of the graph.
FRAME_ALIGNMENT = CONTEXT_SIZE * ENCODER_SUBSAMPLE  # 512 stacked frames = 10.24 s


@dataclass(frozen=True)
class WindowGeometry:
    """Frame counts for one fixed-size CoreML window."""

    seconds: float
    samples: int
    mel_frames: int
    stacked_frames: int
    encoder_frames: int

    @staticmethod
    def for_seconds(seconds: float) -> "WindowGeometry":
        mel_frames = int(round(seconds * SAMPLE_RATE / HOP_LENGTH))
        stacked_frames = mel_frames // STACK_FACTOR
        stacked_frames = max(
            FRAME_ALIGNMENT, stacked_frames - stacked_frames % FRAME_ALIGNMENT
        )
        mel_frames = stacked_frames * STACK_FACTOR
        return WindowGeometry(
            seconds=mel_frames * HOP_LENGTH / SAMPLE_RATE,
            samples=mel_frames * HOP_LENGTH,
            mel_frames=mel_frames,
            stacked_frames=stacked_frames,
            encoder_frames=stacked_frames // ENCODER_SUBSAMPLE,
        )


class GraniteTurboCtcWrapper(nn.Module):
    """Traceable wrapper: log-mel features in, greedy CTC token ids out.

    The argmax is folded into the graph so the 16384-wide logits never cross the
    CoreML boundary -- that tensor is ~2 MB per second of audio otherwise.
    """

    def __init__(self, model: nn.Module, emit_logits: bool = False) -> None:
        super().__init__()
        self.model = model
        self.emit_logits = emit_logits

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = self.model(input_features=input_features).logits
        if self.emit_logits:
            return (logits,)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        confidence, token_ids = log_probs.max(dim=-1)
        return token_ids.to(torch.int32), confidence


def build_traced_module(
    model_id: str, geometry: WindowGeometry, emit_logits: bool
) -> torch.jit.ScriptModule:
    coreml_patches.apply()
    model = AutoModelForCTC.from_pretrained(model_id, dtype=torch.float32).eval()
    wrapper = GraniteTurboCtcWrapper(model, emit_logits=emit_logits).eval()

    example = torch.zeros(1, geometry.stacked_frames, FEATURE_DIM, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
    return torch.jit.freeze(traced)


def convert(
    traced: torch.jit.ScriptModule,
    geometry: WindowGeometry,
    precision: ct.precision,
    emit_logits: bool,
) -> ct.models.MLModel:
    inputs = [
        ct.TensorType(
            name="input_features",
            shape=(1, geometry.stacked_frames, FEATURE_DIM),
            dtype=np.float32,
        )
    ]
    outputs = (
        [ct.TensorType(name="logits")]
        if emit_logits
        else [ct.TensorType(name="token_ids"), ct.TensorType(name="confidence")]
    )
    return ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )


def write_manifest(
    output_dir: Path, model_id: str, geometry: WindowGeometry, package: str
) -> None:
    manifest = {
        "model_id": model_id,
        "package": package,
        "mel_filters": "mel_filters.bin",
        "tokenizer": "processor",
        "sample_rate": SAMPLE_RATE,
        "n_fft": 512,
        "win_length": 400,
        "hop_length": HOP_LENGTH,
        "n_mels": 80,
        "stack_factor": STACK_FACTOR,
        "deltas": True,
        "delta_win_length": 3,
        "logmel_floor_db": 8.0,
        "feature_dim": FEATURE_DIM,
        "encoder_subsample": ENCODER_SUBSAMPLE,
        "context_size": CONTEXT_SIZE,
        "blank_token_id": 0,
        "window": {
            "seconds": geometry.seconds,
            "samples": geometry.samples,
            "mel_frames": geometry.mel_frames,
            "stacked_frames": geometry.stacked_frames,
            "encoder_frames": geometry.encoder_frames,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def retarget_tokenizer_class(processor_dir: Path) -> None:
    """Rewrite `tokenizer_class` to a name swift-transformers recognises.

    The upstream config declares `ParakeetTokenizer`, which swift-transformers
    rejects outright. The file is a plain ByteLevel BPE, so pointing it at
    `PreTrainedTokenizerFast` lets the Swift loader dispatch on `model.type`
    instead. Nothing about the vocabulary or merges changes.
    """
    config_path = processor_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=APACHE_MODEL_ID)
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--emit-logits",
        action="store_true",
        help="Emit raw logits instead of folded argmax (for parity debugging).",
    )
    args = parser.parse_args()

    geometry = WindowGeometry.for_seconds(args.window_seconds)
    print(
        f"window: {geometry.seconds:.2f}s -> {geometry.stacked_frames} stacked frames "
        f"-> {geometry.encoder_frames} encoder frames"
    )

    traced = build_traced_module(args.model_id, geometry, args.emit_logits)
    precision = (
        ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    )
    mlmodel = convert(traced, geometry, precision, args.emit_logits)

    mlmodel.author = AUTHOR
    mlmodel.short_description = (
        f"Granite Speech 5.0 470M TurboCTC encoder + CTC head "
        f"({geometry.seconds:.2f}s window, {args.precision})"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "logits" if args.emit_logits else "greedy"
    package = args.output_dir / f"granite-turboctc-{geometry.seconds:.2f}s-{suffix}.mlpackage"
    mlmodel.save(str(package))
    write_manifest(args.output_dir, args.model_id, geometry, package.name)

    # The tokenizer travels with the package; the Swift side already loads
    # tokenizer.json for the existing Granite pipeline.
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor_dir = args.output_dir / "processor"
    processor.save_pretrained(str(processor_dir))
    retarget_tokenizer_class(processor_dir)

    print(f"saved {package}")


if __name__ == "__main__":
    main()
