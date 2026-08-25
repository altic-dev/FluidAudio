# Granite Speech 5.0 470M TurboCTC — CoreML

CoreML export of IBM's [`granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)
(Apache 2.0). The [`-nc` checkpoint](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc)
is architecturally identical, so `--model-id` swaps between them with no other change.

> The `-nc` weights are CC-BY-NC-SA-4.0. Convert them locally for measurement if
> you want the WER comparison, but do not publish the resulting artifact.

## Status

| | |
|---|---|
| Token parity, Python CoreML vs PyTorch | **100.000%** (750/750 frames, FP16) |
| Token parity, **Swift end-to-end** vs PyTorch | **exact** (231/231 tokens, 60 s, every compute unit) |
| Best RTFx, model only (Python) | **1382x** (`ALL`) |
| Best RTFx, **end-to-end** (Swift release, incl. front-end) | **521x** (ANE) |
| Window | 10.24 s → 512 stacked frames → 128 encoder frames |

End-to-end Swift on a 60 s clip, M5 Max:

| Compute units | Processing | RTFx | Parity |
|---|---|---|---|
| ANE | 0.115 s | **521x** | exact |
| GPU | 0.211 s | 285x | exact |
| ALL | 0.215 s | 279x | exact |
| CPU | 0.381 s | 157x | exact |

## Quick start

```bash
uv sync
uv run python convert-coreml.py --window-seconds 10.24 --output-dir build
uv run python compare-parity.py --mlpackage build/granite-turboctc-10.24s-greedy.mlpackage
uv run python export-assets.py --reference-seconds 60.0004 --output-dir build
uv run python bench.py --mlpackage build/granite-turboctc-10.24s-greedy.mlpackage
```

Swift end-to-end, from the FluidAudio root:

```bash
swift build -c release
.build/out/Products/Release/fluidaudiocli granite-turboctc \
    Scripts/granite-turboctc/coreml/audio/yc_first_minute_16k.wav \
    --model-dir Scripts/granite-turboctc/coreml/build \
    --reference-json Scripts/granite-turboctc/coreml/build/reference.json \
    --compute-units ane
```

The fixture in `reference.json` runs PyTorch through the **same fixed-window
schedule** as the Swift manager. An unwindowed full-length forward pass is not a
valid reference: the subsampling convolutions and self-conditioning cross block
boundaries, so it disagrees by a token or two at every seam.

## I/O contract

**Input** — `input_features`: `(1, 512, 320)` float32. Log-mel front-end: 80 mels
+ 80 deltas, stacked by 2. `n_fft=512`, `win_length=400`, `hop_length=160`,
`logmel_floor_db=8.0`.

**Outputs**
- `token_ids`: `(1, 128)` int32 — per-frame greedy argmax
- `confidence`: `(1, 128)` float32 — log-softmax value of the winning token

The 16384-wide logits are reduced inside the graph on purpose: emitting them
would move ~8 MB per window across the CoreML boundary for no benefit to greedy
decoding. Use `--emit-logits` when debugging parity.

Decode by collapsing repeats and dropping blank (id `0`), then running the
tokenizer over what remains.

## Window alignment matters

`--window-seconds` is rounded **down** to a multiple of 512 stacked frames
(10.24 s). This is not cosmetic:

The encoder right-pads each attention block to a multiple of `context_size=128`
and builds a bool mask to suppress the pad. That mask lowers to `bitwise_not`,
which coremltools rejects. Attention runs at L, L/2 and L/4, so aligning L to
`128 * 4 = 512` makes the pad zero-width at every layer, the mask stays `None`,
and the entire branch disappears from the traced graph.

## Patches

`coreml_patches.py` holds the compatibility rewrites. Each is numerically
identical to upstream:

| Op | Why | Replacement |
|---|---|---|
| `Tensor.unfold(1, 2, 2).mean(-1)` | no coremltools lowering | reshape over non-overlapping pairs, mean over the pair axis |

`export-assets.py` also rewrites `tokenizer_class` from `ParakeetTokenizer` to
`PreTrainedTokenizerFast`: swift-transformers rejects the former outright, and
the file is a plain ByteLevel BPE, so dispatching on `model.type` loads it
correctly. Vocabulary and merges are untouched.

## `processor.batch_decode` double-collapses — do not use it

`batch_decode` runs its **own** CTC squash. Feeding it an already-collapsed
sequence collapses a second time and silently deletes genuine repeated words.

Real case from the test clip: raw frames `[371, 0, 0, 371]` — two ` had` tokens
separated by blanks. Blank separation is precisely how CTC encodes a true repeat,
so "had had" is the correct output. `batch_decode` returns "had".

Decode with the bare tokenizer over the collapsed ids instead. The Swift path
(`GraniteTurboCtcManager.collapseCTC` + `GraniteTokenizer.decode`) already does
this; `compare-parity.py` was fixed to match.

## Notes for the next pass

- **GPU vs ANE flips between harnesses.** Model-only in Python, GPU wins
  (1363x vs 642x); end-to-end in Swift, ANE wins (521x vs 285x). The Python
  number is dominated by per-call `predict` overhead, so trust the Swift figure.
  ANE still costs ~4.7 s to load versus ~1.4 s.
- The Swift front-end is a near-copy of `GraniteFeatureExtractor` plus deltas and
  320-wide stacking. Worth folding the two into one shared core once this model
  earns its place; kept separate for now so the shipping 4.1 NAR path is untouched.
- Padded-tail frames are trimmed by the caller, not masked in-graph. Fine for
  greedy CTC; revisit if the last window's final tokens ever look wrong.
- No streaming support yet — fixed windows only.
