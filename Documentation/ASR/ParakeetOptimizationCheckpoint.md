# Parakeet optimization checkpoint

This document records the experimental state of branch
`B/parakeet-vocab-boost-optimization` as of August 12, 2026. The branch is a
recovery/checkpoint branch, not a production recommendation.

## Included work

- Unified incremental preview and final transcription, with finalized windows
  reused at stop while retaining batch-equivalent merge behavior.
- Isolated decoder state for incremental previews and independent sessions.
- Faster custom-vocabulary rescoring: direct Core ML buffer conversion,
  reusable CTC log-probabilities, normalized-word caching, linear-memory edit
  distance, and guards against phrase-boundary and inflection corruption.
- Split preprocessor/encoder frontend pipelining for supported Parakeet models.
- Experimental independent chunk workers and hybrid pipeline/worker execution.
- Resource-aware gates based on physical and currently available memory.

## Runtime policy

- `ASRConfig.parallelChunkConcurrency` defaults to `1`.
- Automatic frontend pipelining requires a split frontend, at least 24 GiB of
  physical memory, and at least 8 GiB of available memory.
- `FLUIDAUDIO_FRONTEND_PIPELINE_MODE=serial` forces the conservative serial path.
- `FLUIDAUDIO_FRONTEND_PIPELINE_MODE=pipeline` forces single-manager frontend
  pipelining.
- `FLUIDAUDIO_FRONTEND_PIPELINE_MODE=workers` forces the configured worker pool.
- `FLUIDAUDIO_FRONTEND_PIPELINE_MODE=hybrid` forces the experimental combined
  path. Do not enable this in production until exact transcript parity passes.
- Vocabulary customization forces worker concurrency to one.

## Validation evidence

- Hardened incremental output matched batch text, confidence, duration, token
  IDs, timestamps, and token confidence across repeated English tests, critical
  sample boundaries, and 24 multilingual comparisons over eight FLEURS fixtures.
- The full 2 h 25 m 58 s Jensen fixture matched batch text and all 23,417 word
  timings while reducing stop finalization from 32.61 s to 0.123 s when previous
  windows had been processed during recording.
- Custom-vocabulary batch and incremental results were identical in controlled
  Jensen tests. Vocabulary rescoring remains materially slower than unboosted
  transcription because it runs an additional CTC inference path.
- On an Apple M5 Max, the installed FluidVoice file path with concurrency one
  processed 8,758.36 s of decoded Jensen audio in about 27.87 s end to end
  (314x), with about 22.36 s inside provider inference (392x).

## Known gaps

- Forced multi-worker and hybrid modes are experimental. Hybrid improved
  throughput in exploration but changed the transcript, so it failed the
  byte-identical acceptance gate.
- Automatic high-memory pipelining still needs installed-app stress testing on
  8 GiB and 16 GiB older Apple Silicon Macs before broader enablement.
- FluidVoice's meeting UI can report exactly double the real duration for some
  stereo M4A files because `AVAsset.duration` is doubled. Calculate benchmark
  RTFx from decoded 16 kHz mono sample count until that app-level display bug is
  fixed.
- A speed result is acceptable only when text, timings, confidence, model
  version, vocabulary setting, decoded sample count, and wall-clock boundaries
  are recorded together.
