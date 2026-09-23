import Foundation

/// Configuration for locally converted Nemotron streaming diarization exports.
///
/// Nemotron is a Sortformer-family model with a different tensor geometry (8 speaker slots, a much
/// longer chunk, a larger speaker cache). Everything in ``SortformerConfig/NemotronGeometry`` is
/// transcribed from an actual `MLModel` description, not from a file name or a README — the model's
/// own descriptions are authoritative and are re-validated at load time by
/// ``SortformerModelGeometry/validate(against:)``.
///
/// - Important: Only *runtime* compatibility (feature names, shapes, dtypes, streaming state layout)
///   is covered here. Exact mel front-end and cache-configuration parity with the upstream NeMo
///   checkpoint cannot be claimed without the `.nemo` file and reference mel fixtures.
extension SortformerConfig {

    /// Tensor geometry of the Nemotron diarization export.
    ///
    /// Reference model description (`nemotron_diar_fp16.mlpackage`):
    /// - in  `chunk`                      Float32 [1, 3040, 128]
    /// - in  `chunk_lengths`              Int32   [1]
    /// - in  `spkcache`                   Float32 [1, 264, 512]
    /// - in  `spkcache_lengths`           Int32   [1]
    /// - in  `fifo`                       Float32 [1, 40, 512]
    /// - in  `fifo_lengths`               Int32   [1]
    /// - out `spkcache_fifo_chunk_preds`  Float16 [1, 684, 8]
    /// - out `chunk_pre_encode_embs`      Float16 [1, 380, 512]
    /// - out `chunk_pre_encode_lengths`   Int32   [1]
    ///
    /// 3040 mel frames / subsampling 8 = 380 encoder frames = 0 left + 340 core + 40 right, and
    /// 264 + 40 + 380 = 684 prediction rows.
    public enum NemotronGeometry {
        public static let numSpeakers = 8
        public static let chunkLen = 340
        public static let chunkLeftContext = 0
        public static let chunkRightContext = 40
        public static let fifoLen = 40
        public static let spkcacheLen = 264

        /// Mel frames on the `chunk` input: (340 + 0 + 40) * 8.
        public static let chunkMelFrames = 3040
        /// Encoder frames on the `chunk_pre_encode_embs` output: 3040 / 8.
        public static let chunkEncoderFrames = 380
        /// Rows on the `spkcache_fifo_chunk_preds` output: 264 + 40 + 380.
        public static let predictionFrames = 684

        /// Update periods the streaming state updater can run safely with this geometry.
        ///
        /// `SortformerStateUpdater` pops `max(spkcacheUpdatePeriod, contextLength - fifoLen)` frames,
        /// capped at `contextLength`, so any period in `[1, fifoLen + chunkLen]` keeps the FIFO and
        /// speaker cache within their declared capacities. A period below `chunkLen` only makes the
        /// cache update lag one chunk behind; it never pops more than the FIFO holds.
        public static let spkcacheUpdatePeriodRange = 1...(fifoLen + chunkLen)

        /// Update period assigned by the reference Nemotron conversion script.
        ///
        /// - Important: This value is **not** verified against the upstream NeMo checkpoint; it is
        ///   recorded here only so callers can pass it explicitly. See the note on
        ///   ``SortformerConfig/nemotron(spkcacheUpdatePeriod:spkcacheSilFramesPerSpk:predScoreThreshold:silenceThreshold:scoresBoostLatest:strongBoostRate:weakBoostRate:minPosScoresRate:learnedSilenceEmbedding:debugMode:)``
        ///   for how it differs from the generic initializer's clamp.
        public static let referenceScriptSpkcacheUpdatePeriod = 300

        /// Reject an update period the streaming updater cannot run with this geometry.
        static func validate(spkcacheUpdatePeriod: Int) throws {
            guard spkcacheUpdatePeriodRange.contains(spkcacheUpdatePeriod) else {
                throw SortformerError.configurationError(
                    "spkcacheUpdatePeriod \(spkcacheUpdatePeriod) is outside the range supported by the "
                        + "Nemotron geometry [\(spkcacheUpdatePeriodRange.lowerBound), "
                        + "\(spkcacheUpdatePeriodRange.upperBound)] "
                        + "(chunkLen=\(chunkLen), fifoLen=\(fifoLen))"
                )
            }
        }
    }

    /// Build a configuration for a Nemotron export.
    ///
    /// The tensor geometry is pinned to ``NemotronGeometry``. The parameters below are *not* derivable
    /// from tensor sizes — they come from the conversion/NeMo streaming configuration — so they are
    /// required rather than guessed from a size formula.
    ///
    /// ## Update period: 300 vs 340
    ///
    /// The reference conversion script assigns `spkcache_update_period = 300`
    /// (``NemotronGeometry/referenceScriptSpkcacheUpdatePeriod``), which is *below* this geometry's
    /// `chunkLen` of 340. ``SortformerConfig/init(modelVariant:numSpeakers:chunkLen:chunkLeftContext:chunkRightContext:fifoLen:spkcacheLen:spkcacheUpdatePeriod:silenceThreshold:spkcacheSilFramesPerSpk:predScoreThreshold:scoresBoostLatest:strongBoostRate:weakBoostRate:minPosScoresRate:debugMode:)``
    /// clamps any period up to `chunkLen`, so the generic path would silently turn 300 into 340 — a
    /// different streaming cadence (340 drains the FIFO completely on each update, 300 leaves the
    /// 40-frame FIFO full). This factory does **not** clamp: it validates the value against
    /// ``NemotronGeometry/spkcacheUpdatePeriodRange`` and keeps it exactly as given, so passing 300
    /// yields 300 and passing an unusable value throws.
    ///
    /// - Important: Keeping the value the reference script uses is a *runtime* compatibility measure
    ///   only. It does not establish numerical parity with the upstream NeMo checkpoint: that would
    ///   require the `.nemo` file and reference fixtures, neither of which is checked here.
    ///
    /// - Parameters:
    ///   - spkcacheUpdatePeriod: Frames popped from the FIFO into the speaker cache per update.
    ///     Must be within ``NemotronGeometry/spkcacheUpdatePeriodRange`` = `[1, 380]`; it is used
    ///     verbatim, never clamped.
    ///   - spkcacheSilFramesPerSpk: Silence placeholder frames reserved per speaker during cache
    ///     compression. Must satisfy `spkcacheLen / numSpeakers - value > 0`, i.e. `< 33`.
    ///   - predScoreThreshold: Probability clamp used when scoring cached frames for compression.
    ///   - silenceThreshold: Summed-probability threshold below which a frame updates the silence profile.
    ///   - learnedSilenceEmbedding: The checkpoint's trained silence embedding (`learnable_sil_emb`).
    ///     Nemotron 3 is trained with it; without it, cache compression falls back to a running mean.
    ///   - debugMode: Enable verbose logging.
    /// - Throws: `SortformerError.configurationError` if the resulting geometry is unusable.
    public static func nemotron(
        spkcacheUpdatePeriod: Int,
        spkcacheSilFramesPerSpk: Int,
        predScoreThreshold: Float,
        silenceThreshold: Float = 0.2,
        scoresBoostLatest: Float = 0.05,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        learnedSilenceEmbedding: [Float]? = nil,
        debugMode: Bool = false
    ) throws -> SortformerConfig {
        try NemotronGeometry.validate(spkcacheUpdatePeriod: spkcacheUpdatePeriod)
        guard (0..<33).contains(spkcacheSilFramesPerSpk),
            predScoreThreshold.isFinite, predScoreThreshold > 0, predScoreThreshold < 1
        else {
            throw SortformerError.configurationError("Invalid Nemotron cache scoring configuration")
        }
        var config = SortformerConfig(
            // Nemotron is local-path only: there is no HuggingFace bundle to resolve.
            modelVariant: nil,
            melFrontend: .nemotron3Diarization,
            numSpeakers: NemotronGeometry.numSpeakers,
            chunkLen: NemotronGeometry.chunkLen,
            chunkLeftContext: NemotronGeometry.chunkLeftContext,
            chunkRightContext: NemotronGeometry.chunkRightContext,
            fifoLen: NemotronGeometry.fifoLen,
            spkcacheLen: NemotronGeometry.spkcacheLen,
            spkcacheUpdatePeriod: spkcacheUpdatePeriod,
            silenceThreshold: silenceThreshold,
            spkcacheSilFramesPerSpk: spkcacheSilFramesPerSpk,
            predScoreThreshold: predScoreThreshold,
            scoresBoostLatest: scoresBoostLatest,
            strongBoostRate: strongBoostRate,
            weakBoostRate: weakBoostRate,
            minPosScoresRate: minPosScoresRate,
            debugMode: debugMode
        )

        // Preserve the caller's explicit reference cadence after the generic legacy clamp.
        config.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        config.learnedSilenceEmbedding = learnedSilenceEmbedding
        guard config.spkcacheLen == NemotronGeometry.spkcacheLen else {
            throw SortformerError.configurationError(
                "spkcacheSilFramesPerSpk \(spkcacheSilFramesPerSpk) forces spkcacheLen to "
                    + "\(config.spkcacheLen), but the model requires \(NemotronGeometry.spkcacheLen)"
            )
        }

        try config.validateGeometry()
        return config
    }
}
