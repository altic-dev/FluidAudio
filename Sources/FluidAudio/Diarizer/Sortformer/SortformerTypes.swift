import Foundation

// MARK: - Mel Frontend Contract

/// Immutable host-side feature contract for a Sortformer-family export.
///
/// The CoreML graph starts after feature extraction, so these values are part of the model ABI just
/// as much as its tensor shapes. `family` deliberately remains part of the value even when two
/// exports currently share the same numbers; this prevents a new model from silently inheriting a
/// different model family's defaults.
public struct SortformerMelFrontendContract: Sendable, Equatable {
    public enum Family: String, Sendable {
        case streamingSortformer
        case nemotron3Diarization
    }

    public enum Padding: String, Sendable {
        case centeredConstant
    }

    public enum Window: String, Sendable {
        case symmetricHann
    }

    public enum MelScale: String, Sendable {
        case slaneyNormalized
    }

    public enum LogGuard: String, Sendable {
        case additive
    }

    public enum Normalization: String, Sendable {
        case none
    }

    public let family: Family
    public let sampleRate: Int
    public let featureCount: Int
    public let fftLength: Int
    public let windowLength: Int
    public let hopLength: Int
    public let preemphasis: Float
    public let dither: Float
    public let padding: Padding
    public let window: Window
    public let melScale: MelScale
    public let magnitudePower: Float
    public let logGuard: LogGuard
    public let logGuardValue: Float
    public let normalization: Normalization
    public let padTo: Int
    public let padValue: Float

    private init(family: Family) {
        self.family = family
        self.sampleRate = 16_000
        self.featureCount = 128
        self.fftLength = 512
        self.windowLength = 400
        self.hopLength = 160
        self.preemphasis = 0.97
        self.dither = 0
        self.padding = .centeredConstant
        self.window = .symmetricHann
        self.melScale = .slaneyNormalized
        self.magnitudePower = 2
        self.logGuard = .additive
        self.logGuardValue = powf(2, -24)
        self.normalization = .none
        self.padTo = 0
        self.padValue = 0
    }

    public static let streamingSortformer = SortformerMelFrontendContract(family: .streamingSortformer)

    /// Host contract extracted from Nemotron-3's NeMo preprocessor configuration.
    ///
    /// Dither is frozen to zero for deterministic inference. NeMo's configured 25 ms window and
    /// 10 ms hop produce mel frames; the encoder's 8x feature stacking is what produces the 80 ms
    /// diarization cadence.
    public static let nemotron3Diarization = SortformerMelFrontendContract(family: .nemotron3Diarization)

    /// Number of valid NeMo feature frames for an unpadded signal.
    public func validFrameCount(sampleCount: Int) -> Int {
        guard sampleCount > 0 else { return 0 }
        return sampleCount / hopLength
    }

    /// Number of physical frames emitted by centered STFT before NeMo masks the final frame.
    public func storageFrameCount(sampleCount: Int) -> Int {
        guard sampleCount > 0 else { return 0 }
        return validFrameCount(sampleCount: sampleCount) + 1
    }

    func makeSpectrogram() -> AudioMelSpectrogram {
        AudioMelSpectrogram(
            sampleRate: sampleRate,
            nMels: featureCount,
            nFFT: fftLength,
            hopLength: hopLength,
            winLength: windowLength,
            preemph: preemphasis,
            padTo: padTo,
            logFloor: logGuardValue,
            logFloorMode: .additive,
            windowPeriodic: false
        )
    }
}

// MARK: - Saturating Arithmetic

extension Int {
    /// `self + other`, pinned to `Int.min`/`Int.max` instead of trapping.
    ///
    /// Used only by `SortformerConfig.init`, whose clamping must survive absurd inputs long enough
    /// for `validateGeometry()` to describe them. Identical to `+` for every non-overflowing pair.
    static func fluidSaturatingAdd(_ lhs: Int, _ rhs: Int) -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard overflow else { return value }
        return rhs > 0 ? .max : .min
    }

    /// `lhs * rhs`, pinned to `Int.min`/`Int.max` instead of trapping.
    static func fluidSaturatingMultiply(_ lhs: Int, _ rhs: Int) -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard overflow else { return value }
        return (lhs > 0) == (rhs > 0) ? .max : .min
    }
}

// MARK: - Configuration

/// Configuration for Sortformer streaming diarization.
///
/// Based on NVIDIA's Streaming Sortformer 4-speaker model.
/// Reference: NeMo sortformer_modules.py
public struct SortformerConfig: Sendable {
    public typealias ModelVariant = ModelNames.Sortformer.Variant

    // MARK: - Model Architecture

    public let modelVariant: ModelVariant?

    /// Host-side preprocessing ABI paired with this model export.
    public let melFrontend: SortformerMelFrontendContract

    /// Number of speaker slots.
    ///
    /// Must match the trailing dimension of the model's prediction output. The shipped Sortformer
    /// exports use 4; other exports (e.g. a locally converted Nemotron model) may use more.
    public let numSpeakers: Int

    /// Pre-encoder embedding dimension
    public let preEncoderDims: Int = 512

    /// Subsampling factor (8:1 downsampling in encoder)
    public let subsamplingFactor: Int = 8

    // MARK: - Streaming Parameters

    /// Output diarization frames per chunk
    /// Must match the value used in CoreML conversion
    public var chunkLen: Int = 6

    /// Left context frames for chunk processing
    public var chunkLeftContext: Int = 1

    /// Right context frames for chunk processing
    public var chunkRightContext: Int = 7

    /// Maximum FIFO queue length (recent embeddings)
    /// Must match CoreML conversion: fifo_len=40
    public var fifoLen: Int = 40

    /// Maximum speaker cache length (historical embeddings)
    /// Must match CoreML conversion: spkcache_len=188
    public var spkcacheLen: Int = 188

    /// Period for speaker cache updates (frames)
    public var spkcacheUpdatePeriod: Int = 31

    /// Silence frames per speaker in compressed cache
    public var spkcacheSilFramesPerSpk: Int = 3

    // MARK: - Debug

    /// Enable debug logging
    public var debugMode: Bool = false

    // MARK: - Audio Parameters

    /// Sample rate in Hz
    public var sampleRate: Int { melFrontend.sampleRate }

    /// Mel spectrogram window size in samples (25ms)
    public var melWindow: Int { melFrontend.windowLength }

    /// Mel spectrogram stride in samples (10ms)
    public var melStride: Int { melFrontend.hopLength }

    /// Number of mel filterbank features
    public var melFeatures: Int { melFrontend.featureCount }

    // MARK: - Thresholds

    /// Threshold for silence detection (sum of speaker probs)
    public var silenceThreshold: Float = 0.2

    /// Threshold for speech prediction
    public var predScoreThreshold: Float = 0.25

    /// Boost factor for latest frames in cache compression
    public var scoresBoostLatest: Float = 0.05

    /// Strong boost rate for top-k selection
    public var strongBoostRate: Float = 0.75

    /// Weak boost rate for preventing speaker dominance
    public var weakBoostRate: Float = 1.5

    /// Minimum positive scores rate
    public var minPosScoresRate: Float = 0.5

    /// Maximum index placeholder for disabled frames in spkcache compression
    public let maxIndex: Int = 99999

    // MARK: - Computed Properties

    /// Total chunk frames for CoreML model input (includes left/right context)
    /// Formula: (chunk_len + left_context + right_context) * subsampling
    /// Default: (6 + 1 + 1) * 8 = 64 frames
    public var chunkMelFrames: Int {
        (chunkLen + chunkLeftContext + chunkRightContext) * subsamplingFactor
    }

    /// Core frames per chunk (without context)
    public var coreFrames: Int {
        chunkLen * subsamplingFactor
    }

    /// Encoder (post-subsampling) frames produced for one chunk, including left/right context.
    /// This is the second dimension of the model's chunk embedding output.
    public var chunkEncoderFrames: Int {
        chunkLen + chunkLeftContext + chunkRightContext
    }

    /// Rows in the model's prediction output: speaker cache + FIFO + chunk encoder frames.
    public var predictionFrames: Int {
        spkcacheLen + fifoLen + chunkEncoderFrames
    }

    /// Frame duration in seconds
    public var frameDurationSeconds: Float {
        Float(subsamplingFactor) * Float(melStride) / Float(sampleRate)
    }

    // MARK: - Initialization

    /// Configuration matching Gradient Descent's Streaming-Sortformer-Conversion models
    public static let `default` = SortformerConfig(
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2 weights (~1.04s latency, smallest context).
    /// May handle high-speaker-count scenarios better than v2.1 (v2.1 can degrade when many speakers overlap).
    public static let `fastV2` = SortformerConfig(
        modelVariant: .fastV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2.1 weights (~1.04s latency, smallest context).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let `fastV2_1` = SortformerConfig(
        modelVariant: .fastV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Balanced config with Sortformer v2 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM. May handle high-speaker-count scenarios better than v2.1.
    public static let balancedV2 = SortformerConfig(
        modelVariant: .balancedV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// Balanced config with Sortformer v2.1 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM.
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let balancedV2_1 = SortformerConfig(
        modelVariant: .balancedV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// High-context config with Sortformer v2 weights (~30.4s latency, most context window).
    /// May handle high-speaker-count scenarios better than v2.1.
    public static let highContextV2 = SortformerConfig(
        modelVariant: .highContextV2,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// High-context config with Sortformer v2.1 weights (~30.4s latency, most context window).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let highContextV2_1 = SortformerConfig(
        modelVariant: .highContextV2_1,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// - Warning: If you don't use one of the default configurations, you must use a local model converted with that configuration.
    public init(
        modelVariant: ModelVariant? = .fastV2_1,
        melFrontend: SortformerMelFrontendContract = .streamingSortformer,
        numSpeakers: Int = 4,
        chunkLen: Int = 6,
        chunkLeftContext: Int = 1,
        chunkRightContext: Int = 7,
        fifoLen: Int = 40,
        spkcacheLen: Int = 188,
        spkcacheUpdatePeriod: Int = 31,
        silenceThreshold: Float = 0.2,
        spkcacheSilFramesPerSpk: Int = 3,
        predScoreThreshold: Float = 0.25,
        scoresBoostLatest: Float = 0.05,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        debugMode: Bool = false
    ) {
        self.modelVariant = modelVariant
        self.melFrontend = melFrontend
        self.numSpeakers = max(1, numSpeakers)
        self.chunkLen = max(1, chunkLen)
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.fifoLen = fifoLen
        self.silenceThreshold = silenceThreshold
        self.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        self.debugMode = debugMode
        self.predScoreThreshold = predScoreThreshold
        self.scoresBoostLatest = scoresBoostLatest
        self.strongBoostRate = strongBoostRate
        self.weakBoostRate = weakBoostRate
        self.minPosScoresRate = minPosScoresRate

        // The following parameters must meet certain constraints. The arithmetic saturates instead of
        // trapping so that extreme inputs reach `validateGeometry()` and are reported; for every
        // in-range configuration the result is identical to plain `+` and `*`.
        let silencePlaceholders = Int.fluidSaturatingMultiply(
            Int.fluidSaturatingAdd(1, self.spkcacheSilFramesPerSpk), self.numSpeakers)
        self.spkcacheLen = max(spkcacheLen, silencePlaceholders)
        let fifoPlusChunk = Int.fluidSaturatingAdd(self.fifoLen, self.chunkLen)
        self.spkcacheUpdatePeriod = max(min(spkcacheUpdatePeriod, fifoPlusChunk), self.chunkLen)
    }

    /// Whether two configurations describe the same model tensor geometry.
    ///
    /// Compares every dimension that shapes a CoreML input or output, so a configuration that only
    /// differs in tuning knobs (thresholds, boost rates, debug mode) is still compatible, while one
    /// that would need a differently-shaped export is not.
    public func isCompatible(with other: SortformerConfig) -> Bool {
        return self.melFrontend == other.melFrontend
            && self.numSpeakers == other.numSpeakers
            && self.chunkMelFrames == other.chunkMelFrames
            && self.melFeatures == other.melFeatures
            && self.fifoLen == other.fifoLen
            && self.spkcacheLen == other.spkcacheLen
            && self.preEncoderDims == other.preEncoderDims
            && self.subsamplingFactor == other.subsamplingFactor
            && self.chunkEncoderFrames == other.chunkEncoderFrames
    }

    /// Validate that the configured geometry can drive the streaming state updater safely.
    ///
    /// This checks only self-consistency; it does not know anything about a particular model file.
    /// Use ``SortformerModelGeometry/validate(against:)`` for that.
    ///
    /// - Throws: `SortformerError.configurationError` describing every violated constraint.
    public func validateGeometry() throws {
        var problems: [String] = []

        if numSpeakers < 1 {
            problems.append("numSpeakers must be >= 1 (got \(numSpeakers))")
        }
        if chunkLen < 1 {
            problems.append("chunkLen must be >= 1 (got \(chunkLen))")
        }
        if chunkLeftContext < 0 {
            problems.append("chunkLeftContext must be >= 0 (got \(chunkLeftContext))")
        }
        if chunkRightContext < 0 {
            problems.append("chunkRightContext must be >= 0 (got \(chunkRightContext))")
        }
        if fifoLen < 1 {
            problems.append("fifoLen must be >= 1 (got \(fifoLen))")
        }
        if spkcacheLen < 1 {
            problems.append("spkcacheLen must be >= 1 (got \(spkcacheLen))")
        }
        if spkcacheSilFramesPerSpk < 0 {
            problems.append("spkcacheSilFramesPerSpk must be >= 0 (got \(spkcacheSilFramesPerSpk))")
        }

        // Cache compression budgets one slice of the speaker cache per speaker and reserves
        // `spkcacheSilFramesPerSpk` of it for silence placeholders; the remainder must stay positive.
        if numSpeakers >= 1, spkcacheLen >= 1, spkcacheSilFramesPerSpk >= 0 {
            let perSpeaker = spkcacheLen / numSpeakers - spkcacheSilFramesPerSpk
            if perSpeaker <= 0 {
                problems.append(
                    "spkcacheLen / numSpeakers - spkcacheSilFramesPerSpk must be > 0 "
                        + "(\(spkcacheLen) / \(numSpeakers) - \(spkcacheSilFramesPerSpk) = \(perSpeaker))"
                )
            }
        }

        // Every derived size is computed with overflow reporting *before* anything uses it, so an
        // absurd configuration produces a listed problem instead of trapping. `nil` means "this size
        // overflowed"; a problem has already been recorded for it and dependent checks are skipped.
        func sum(_ label: String, _ values: Int...) -> Int? {
            var total = 0
            for value in values {
                let (next, overflow) = total.addingReportingOverflow(value)
                if overflow {
                    problems.append("\(label) overflows Int (\(values.map(String.init).joined(separator: " + ")))")
                    return nil
                }
                total = next
            }
            return total
        }

        func product(_ label: String, _ lhs: Int?, _ rhs: Int) -> Int? {
            guard let lhs else { return nil }
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            if overflow {
                problems.append("\(label) overflows Int (\(lhs) * \(rhs))")
                return nil
            }
            return value
        }

        // `fifoLen + chunkLen` bounds the update period; it is also the FIFO buffer's reserved size.
        let fifoPlusChunk = sum("fifoLen + chunkLen", fifoLen, chunkLen)
        if let fifoPlusChunk {
            // The lower bound is 1, not chunkLen: `SortformerStateUpdater` raises the pop length to
            // `contextLength - fifoCapacity` when the period is smaller, so a shorter period only
            // changes cache cadence, it cannot pop more frames than the FIFO holds. The default
            // initializer still clamps to `>= chunkLen`; configurations that intentionally keep a
            // smaller reference period (see `SortformerConfig.nemotron`) set it explicitly.
            if spkcacheUpdatePeriod < 1 || spkcacheUpdatePeriod > fifoPlusChunk {
                problems.append(
                    "spkcacheUpdatePeriod must be within [1, fifoLen + chunkLen] = "
                        + "[1, \(fifoPlusChunk)] (got \(spkcacheUpdatePeriod))"
                )
            }
        }

        // chunkMelFrames = (chunkLen + chunkLeftContext + chunkRightContext) * subsamplingFactor
        let encoderFrames = sum("chunkEncoderFrames", chunkLen, chunkLeftContext, chunkRightContext)
        let melFrames = product("chunkMelFrames", encoderFrames, subsamplingFactor)
        _ = product("chunk input element count", melFrames, melFeatures)

        // predictionFrames = spkcacheLen + fifoLen + chunkEncoderFrames, and the prediction buffer.
        let predictionRows = encoderFrames.flatMap { sum("predictionFrames", spkcacheLen, fifoLen, $0) }
        _ = product("prediction element count", predictionRows, numSpeakers)

        // Streaming buffers reserved in `SortformerStreamingState`.
        _ = product("fifo buffer element count", fifoPlusChunk, preEncoderDims)
        _ = product(
            "spkcache buffer element count",
            sum("spkcacheLen + spkcacheUpdatePeriod", spkcacheLen, spkcacheUpdatePeriod),
            preEncoderDims)
        _ = product("chunk embedding element count", encoderFrames, preEncoderDims)

        guard problems.isEmpty else {
            throw SortformerError.configurationError(
                "Invalid Sortformer geometry: " + problems.joined(separator: "; ")
            )
        }
    }
}

// MARK: - Streaming State

/// State maintained across streaming chunks for Sortformer diarization.
///
/// This mirrors NeMo's StreamingSortformerState dataclass.
/// Reference: NeMo sortformer_modules.py
public struct SortformerStreamingState: Sendable {
    /// Speaker cache embeddings from start of audio
    /// Shape: [spkcacheLen, fcDModel] (e.g., [188, 512])
    public var spkcache: [Float]

    /// Current valid length of speaker cache
    public var spkcacheLength: Int

    /// Speaker predictions for cached embeddings
    /// Shape: [spkcacheLen, numSpeakers] (e.g., [188, 4])
    public var spkcachePreds: [Float]?

    /// FIFO queue of recent chunk embeddings
    /// Shape: [fifoLen, fcDModel] (e.g., [188, 512])
    public var fifo: [Float]

    /// Current valid length of FIFO queue
    public var fifoLength: Int

    /// Speaker predictions for FIFO embeddings
    /// Shape: [fifoLen, numSpeakers] (e.g., [188, 4])
    public var fifoPreds: [Float]?

    /// Running mean of silence embeddings
    /// Shape: [fcDModel] (e.g., [512])
    public var meanSilenceEmbedding: [Float]

    /// Count of silence frames observed
    public var silenceFrameCount: Int

    /// Initialize empty streaming state
    public init(config: SortformerConfig) {
        self.spkcache = []
        self.spkcachePreds = nil
        self.spkcacheLength = 0

        self.fifo = []
        self.fifoPreds = nil
        self.fifoLength = 0

        self.fifo.reserveCapacity((config.fifoLen + config.chunkLen) * config.preEncoderDims)
        self.spkcache.reserveCapacity((config.spkcacheLen + config.spkcacheUpdatePeriod) * config.preEncoderDims)

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: config.preEncoderDims)
        self.silenceFrameCount = 0
    }

    public mutating func cleanup() {
        self.fifo.removeAll(keepingCapacity: false)
        self.spkcache.removeAll(keepingCapacity: false)
        self.fifoPreds = nil
        self.spkcachePreds = nil
        self.spkcacheLength = 0
        self.fifoLength = 0
        self.meanSilenceEmbedding.removeAll(keepingCapacity: false)
        self.silenceFrameCount = 0
    }
}

// MARK: - Streaming Feature Provider

/// Feature loader for Sortformer's file processing
public struct SortformerFeatureLoader: Sendable {
    public let numChunks: Int

    private let lc: Int
    private let rc: Int
    private let chunkLen: Int
    private let melFeatures: Int
    private let emitsPartialTail: Bool

    private let featSeq: [Float]
    private let featLength: Int
    private let featSeqLength: Int

    private var startFeat: Int
    private var endFeat: Int

    public init(config: SortformerConfig, audio: [Float]) {
        self.lc = config.chunkLeftContext * config.subsamplingFactor
        self.rc = config.chunkRightContext * config.subsamplingFactor
        self.chunkLen = config.chunkLen * config.subsamplingFactor
        self.melFeatures = config.melFeatures
        self.emitsPartialTail = config.melFrontend.family == .nemotron3Diarization

        self.startFeat = 0
        self.endFeat = 0
        let frontend = config.melFrontend
        if self.emitsPartialTail {
            let storageFrames = frontend.storageFrameCount(sampleCount: audio.count)
            let features = frontend.makeSpectrogram().computeFlatTransposed(
                audio: audio,
                paddingMode: .center,
                expectedFrameCount: storageFrames
            )
            self.featSeq = features.mel
            self.featLength = min(frontend.validFrameCount(sampleCount: audio.count), features.numFrames)
            self.featSeqLength = self.featLength
            self.numChunks = self.featLength == 0 ? 0 : (self.featLength + self.chunkLen - 1) / self.chunkLen
        } else {
            let features = frontend.makeSpectrogram().computeFlatTransposed(audio: audio)
            self.featSeq = features.mel
            self.featLength = features.melLength
            self.featSeqLength = features.numFrames
            self.numChunks = max(0, (self.featLength - self.rc) / self.chunkLen)
        }
    }

    public mutating func next() -> (chunkFeatures: [Float], chunkLength: Int, leftOffset: Int, rightOffset: Int)? {
        // Calculate end of core chunk
        endFeat = min(startFeat + chunkLen, featLength)

        // Need at least one core frame
        guard endFeat > startFeat else { return nil }

        if !emitsPartialTail {
            guard endFeat + rc <= featLength else { return nil }
        }

        let leftOffset = min(lc, startFeat)
        let rightOffset = min(rc, featLength - endFeat)

        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        let chunkStartIndex = chunkStartFrame * melFeatures
        let chunkEndIndex = chunkEndFrame * melFeatures
        let chunkFeatures = Array(featSeq[chunkStartIndex..<chunkEndIndex])
        let chunkLength = max(min(featSeqLength - startFeat + leftOffset, chunkEndFrame - chunkStartFrame), 0)

        startFeat = endFeat
        return (chunkFeatures, chunkLength, leftOffset, rightOffset)
    }
}

// MARK: - Result Types

/// Result from streaming state update containing both confirmed and tentative predictions.
///
/// - `confirmed`: Predictions for frames that have passed beyond the right context window.
///   These are final and will not change.
/// - `tentative`: Predictions for frames still within the right context window.
///   These may change when the next chunk arrives with more future context.
///
/// This enables real-time UI display without waiting for the full right context delay.
/// With rightContext=7 and 80ms frames, tentative predictions provide 560ms earlier feedback.
public struct StreamingUpdateResult: Sendable {
    /// Final predictions for confirmed frames [chunkLen * numSpeakers]
    public let confirmed: [Float]

    /// Tentative predictions for right context frames [rightContext * numSpeakers]
    /// May change with next chunk. Empty if rightContext=0.
    public let tentative: [Float]

    /// Number of speakers
    public let numSpeakers: Int

    /// Number of confirmed frames
    public var confirmedFrameCount: Int { confirmed.count / numSpeakers }

    /// Number of tentative frames
    public var tentativeFrameCount: Int { tentative.count / numSpeakers }

    public init(confirmed: [Float], tentative: [Float], numSpeakers: Int = 4) {
        self.confirmed = confirmed
        self.tentative = tentative
        self.numSpeakers = numSpeakers
    }
}

// MARK: - Errors

public enum SortformerError: Error, LocalizedError {
    case notInitialized
    case modelLoadFailed(String)
    case preprocessorFailed(String)
    case inferenceFailed(String)
    case invalidAudioData
    case invalidState(String)
    case configurationError(String)
    case insufficientChunkLength(String)
    case insufficientPredsLength(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Sortformer diarizer not initialized. Call initialize() first."
        case .modelLoadFailed(let message):
            return "Failed to load Sortformer model: \(message)"
        case .preprocessorFailed(let message):
            return "Preprocessor failed: \(message)"
        case .inferenceFailed(let message):
            return "Inference failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .insufficientChunkLength(let message):
            return "Insufficient chunk length: \(message)"
        case .insufficientPredsLength(let message):
            return "Insufficient preds length: \(message)"
        }
    }
}
