# Parakeet Unified streaming

Parakeet Unified EN 0.6B transcribes English audio incrementally using a
chunked-attention CoreML encoder and a persistent RNNT decoder. It uses the
existing `StreamingAsrEngine` API alongside EOU and Nemotron. Existing Qwen3,
TDT and other model APIs are unchanged.

```swift
let engine = StreamingUnifiedAsrManager(
    config: UnifiedConfig(chunkFrames: 7, rightFrames: 1)
)
try await engine.loadModels()
await engine.setPartialTranscriptCallback { text in
    // Replace the displayed partial transcript with this complete snapshot.
}

// Feed owned 16 kHz mono Float32 PCM, in capture order.
try await engine.appendAudio(samples)
try await engine.processBufferedAudio()

// After capture stops, drain all queued audio before flushing right context.
let finalText = try await engine.finish()
try await engine.reset() // Keep the models loaded for the next recording.
```

`appendAudio(AVAudioPCMBuffer)` also accepts audio through the existing FluidAudio
converter. For continuous capture, applications can supply already converted PCM
using `appendAudio([Float])`. Call `processBufferedAudio()` regularly to bound the
audio buffer, and `consumeTokenTimings()` if token timings are needed. Timings are
drained separately from the transcript.

Serialize loading, decoding, finishing and reset operations. `finish()` is
idempotent and releases the withheld right context exactly once. After finishing
or a decoding error, call `reset()` before appending another recording. Cancel
and await an in-flight decoding task before resetting or cleaning up the engine.
Partials are for display; use the result of `finish()` for the final transcript.

| Factory variant | Attention context (left, chunk, right) | Chunk + look-ahead |
| --- | --- | --- |
| `.parakeetUnified320ms` | 70, 2, 2 | 320 ms |
| `.parakeetUnified640ms` | 70, 7, 1 | 640 ms |
| `.parakeetUnified1120ms` | 70, 7, 7 | 1120 ms |
| `.parakeetUnified2080ms` | 70, 13, 13 | 2080 ms |

These are audio-context durations, not measured end-to-end latency. Model loading,
inference and scheduling add to them. The default initializer uses 2080 ms.

The downloader selects only the requested context's encoder and the shared
decoder, joint, vocabulary and metadata from
[`FluidInference/parakeet-unified-en-0.6b-coreml`](https://huggingface.co/FluidInference/parakeet-unified-en-0.6b-coreml).
INT8 is the default; FP16 is also available. INT8 uses CPU and Neural Engine when
the supplied configuration requests `.all`, avoiding an unsupported quantized
GPU execution path. This backend requires CoreML artifacts; GGUF files cannot be
loaded by CoreML.

## Backport provenance

The encoder windowing, native mel normalization, RNNT decoder, feature providers
and token timing implementation are adapted from
[FluidInference/FluidAudio at `2d33f857`](https://github.com/FluidInference/FluidAudio/tree/2d33f857e89cf42275887cc2b25284d7dab1500d/Sources/FluidAudio/ASR/Parakeet/Unified).
This focused backport retains the fork's `StreamingAsrEngine` names and downloader;
it does not require upstream's later model-hub rewrite or replace the fork's
Qwen3 integration. The separate full-attention offline encoder is outside this
change. Saved recordings can be fed through the streaming engine and finalized
with `finish()`.

## Validation

`UnifiedWindowingTests` covers initial context, sliding windows, all four latency
tiers, short recordings, exact chunk boundaries and unaligned final buffers.
`UnifiedIntegrationTests` uses real speech and CoreML weights and is opt-in:

```sh
FLUID_UNIFIED_TEST_AUDIO=/path/to/whisper.cpp/samples/jfk.wav \
FLUID_UNIFIED_TEST_MODELS=/path/to/model-cache \
swift test --filter 'Unified|StreamingAsrEngineTests'
```

The integration test checks partials before finalization, equivalence across
different audio delivery sizes, repeated finish, reset and reuse after cancellation.
Missing weights are downloaded into the specified cache.
