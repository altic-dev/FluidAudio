import Foundation

struct ParakeetIncrementalState: Sendable {
    typealias TokenWindow = AsrChunkTokenMerger.TokenWindow

    let layout: ParakeetChunkLayout
    private(set) var retainedSamples: [Float] = []
    private(set) var retainedSampleOffset = 0
    private(set) var totalSampleCount = 0
    private(set) var nextChunkStart = 0
    private(set) var nextChunkIndex = 0
    private(set) var finalizedWindows: [[TokenWindow]] = []

    init(layout: ParakeetChunkLayout = .standard) {
        self.layout = layout
    }

    var hasStableWindow: Bool {
        layout.isStableWindow(
            chunkStart: nextChunkStart,
            availableSamples: totalSampleCount
        )
    }

    mutating func append<C: Collection>(_ newSamples: C) where C.Element == Float {
        retainedSamples.append(contentsOf: newSamples)
        totalSampleCount += newSamples.count
    }

    func makeNextStableWork() throws -> ParakeetChunkWork? {
        guard hasStableWindow else { return nil }
        return try makeWork(chunkStart: nextChunkStart, chunkIndex: nextChunkIndex)
    }

    func makeTailWork() throws -> ParakeetChunkWork? {
        try makeWork(chunkStart: nextChunkStart, chunkIndex: nextChunkIndex)
    }

    mutating func recordFinalizedWindow(_ window: [TokenWindow]) {
        precondition(hasStableWindow, "Only stable Parakeet windows can be finalized")
        finalizedWindows.append(window)
        nextChunkStart += layout.strideSamples
        nextChunkIndex += 1
        trimRetainedPrefix()
    }

    private func makeWork(chunkStart: Int, chunkIndex: Int) throws -> ParakeetChunkWork? {
        try layout.makeWork(
            totalSamples: totalSampleCount,
            chunkStart: chunkStart,
            chunkIndex: chunkIndex,
            readSamples: { offset, count in
                try readSamples(offset: offset, count: count)
            }
        )
    }

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        let localOffset = offset - retainedSampleOffset
        guard localOffset >= 0, count >= 0, localOffset + count <= retainedSamples.count else {
            throw ASRError.processingFailed("Incremental audio buffer no longer contains the requested window")
        }
        return Array(retainedSamples[localOffset..<(localOffset + count)])
    }

    private mutating func trimRetainedPrefix() {
        let requiredOffset = max(0, nextChunkStart - layout.melContextSamples)
        let removableCount = requiredOffset - retainedSampleOffset
        guard removableCount > 0 else { return }
        retainedSamples.removeFirst(removableCount)
        retainedSampleOffset = requiredOffset
    }
}

/// An append-only Parakeet session that reuses finalized batch windows.
///
/// The session intentionally uses the same chunk layout, inference path, and overlap merger as
/// `AsrManager.transcribe`. Only the unfinished tail is reprocessed for previews and finalization.
/// Keep one session per recording and append only newly captured 16 kHz mono PCM samples.
/// A temporary window from normal recognition. Samples, token times and feature indices are
/// window-local; sampleOffset maps accepted matches back into the recording. Not retained by the session.
public struct PronunciationChunk: Sendable {
    public let sampleOffset: Int
    public let samples: [Float]
    public let features: EncoderFeatureSequence
    public let result: ASRResult
    public let matches: [PronunciationWindowMatch]
}

public typealias PronunciationChunkRefiner = @Sendable (PronunciationChunk) async throws -> [PronunciationWindowMatch]

public actor ParakeetIncrementalSession {
    private typealias TokenWindow = AsrChunkTokenMerger.TokenWindow

    private enum Lifecycle {
        case active
        case finished
        case failed
    }

    private let manager: AsrManager
    private let source: AudioSource
    private let appliesVocabularyBoosting: Bool
    private let layout = ParakeetChunkLayout.standard

    private var state = ParakeetIncrementalState()
    private var lastPreviewSampleCount = -1
    private var lastPreviewResult: ASRResult?
    private var lifecycle = Lifecycle.active
    private var operationInFlight = false
    private let pronunciationPrototypes: [PronunciationEmbedding]
    private let pronunciationThreshold: Float
    private let pronunciationRefiner: PronunciationChunkRefiner?
    private let pronunciationEnabled: (@Sendable () -> Bool)?
    private var pronunciationWindows = PronunciationWindowMatches()

    /// Latest decoded text before optional vocabulary rescoring changes its spelling.
    public var unboostedText: String? { lastPreviewResult?.text }

    /// Includes stable hits plus the current tail, in recording-global encoder frames.
    public var pronunciationMatches: [PronunciationWindowMatch] {
        pronunciationEnabled?() == false ? [] : pronunciationWindows.all
    }

    init(
        manager: AsrManager, source: AudioSource, appliesVocabularyBoosting: Bool,
        pronunciationPrototypes: [PronunciationEmbedding] = [],
        pronunciationThreshold: Float = PronunciationCustomizationDefaults.acceptanceThreshold,
        pronunciationRefiner: PronunciationChunkRefiner? = nil,
        pronunciationEnabled: (@Sendable () -> Bool)? = nil
    ) {
        self.manager = manager
        self.source = source
        self.appliesVocabularyBoosting = appliesVocabularyBoosting
        self.pronunciationPrototypes = pronunciationPrototypes
        self.pronunciationThreshold = pronunciationThreshold
        self.pronunciationRefiner = pronunciationRefiner
        self.pronunciationEnabled = pronunciationEnabled
    }

    /// Number of source samples accepted by this session.
    public var acceptedSampleCount: Int {
        state.totalSampleCount
    }

    /// Number of stable model windows that will not be inferred again.
    public var finalizedWindowCount: Int {
        state.finalizedWindows.count
    }

    var retainedSampleCount: Int { state.retainedSamples.count }

    /// Appends only the samples captured since the preceding call.
    public func append(_ newSamples: [Float]) async throws {
        try requireActive()
        guard !operationInFlight else { throw ASRError.processingFailed("Incremental operation already in progress") }
        operationInFlight = true
        defer { operationInFlight = false }
        guard !newSamples.isEmpty else { return }
        do {
            try Task.checkCancellation()
            var offset = newSamples.startIndex
            while offset < newSamples.endIndex {
                let end = newSamples.index(
                    offset,
                    offsetBy: min(layout.maxModelSamples, newSamples.distance(from: offset, to: newSamples.endIndex))
                )
                state.append(newSamples[offset..<end])
                pronunciationWindows.replaceTail([])
                lastPreviewResult = nil
                lastPreviewSampleCount = -1
                try await processStableWindows()
                offset = end
                try Task.checkCancellation()
            }
        } catch {
            lifecycle = .failed
            pronunciationWindows = PronunciationWindowMatches()
            throw error
        }
    }

    /// Produces a transcript for all accepted samples while retaining reusable stable windows.
    public func preview() async throws -> ASRResult {
        try requireActive()
        guard !operationInFlight else { throw ASRError.processingFailed("Incremental operation already in progress") }
        operationInFlight = true
        defer { operationInFlight = false }
        guard state.totalSampleCount >= layout.sampleRate else {
            throw ASRError.invalidAudioData
        }
        do {
            return try await makePreview()
        } catch {
            lifecycle = .failed
            pronunciationWindows = PronunciationWindowMatches()
            throw error
        }
    }

    /// Finalizes the session. If the last preview covered the same sample count, no model work is repeated.
    public func finish(finalAudioSamples: [Float]? = nil) async throws -> ASRResult {
        try requireActive()
        guard !operationInFlight else { throw ASRError.processingFailed("Incremental operation already in progress") }
        operationInFlight = true
        defer { operationInFlight = false }
        guard state.totalSampleCount >= layout.sampleRate else {
            throw ASRError.invalidAudioData
        }
        if appliesVocabularyBoosting {
            guard let finalAudioSamples, finalAudioSamples.count == state.totalSampleCount else {
                throw ASRError.processingFailed(
                    "Vocabulary-boosted finalization requires the complete waveform"
                )
            }
        }
        do {
            var result = try await makePreview()
            if appliesVocabularyBoosting {
                guard let finalAudioSamples else { preconditionFailure("Validated above") }
                result = await manager.applyVocabularyRescoring(
                    result: result,
                    audioSamples: finalAudioSamples
                )
            }
            try Task.checkCancellation()
            lifecycle = .finished
            return result
        } catch {
            lifecycle = .failed
            pronunciationWindows = PronunciationWindowMatches()
            throw error
        }
    }

    private func requireActive() throws {
        switch lifecycle {
        case .active:
            return
        case .finished:
            throw ASRError.processingFailed("Incremental session is already finalized")
        case .failed:
            throw ASRError.processingFailed("Incremental session is invalid after a previous failure")
        }
    }

    private func makePreview() async throws -> ASRResult {
        try Task.checkCancellation()
        if lastPreviewSampleCount == state.totalSampleCount, let lastPreviewResult {
            return lastPreviewResult
        }

        let startedAt = Date()
        let result: ASRResult
        if state.totalSampleCount <= layout.maxModelSamples && pronunciationPrototypes.isEmpty {
            result = try await manager.transcribeWithState(
                state.retainedSamples,
                source: source,
                applyVocabularyBoosting: false,
                isolatedDecoderState: true
            )
        } else {
            try await processStableWindows()
            let tailWindow = try await transcribeTailWindow()
            result = await makeResult(
                windows: state.finalizedWindows + [tailWindow],
                processingTime: Date().timeIntervalSince(startedAt)
            )
        }

        try Task.checkCancellation()
        lastPreviewSampleCount = state.totalSampleCount
        lastPreviewResult = result
        return result
    }

    private func processStableWindows() async throws {
        guard state.totalSampleCount > layout.maxModelSamples else { return }

        while state.hasStableWindow {
            try Task.checkCancellation()
            guard let work = try state.makeNextStableWork() else {
                break
            }

            var decoderState = TdtDecoderState.make(
                decoderLayers: await manager.getDecoderLayers()
            )
            decoderState.reset()
            let output = try await ParakeetChunkInference.transcribeWithPronunciation(
                work: work,
                using: manager,
                decoderState: &decoderState, prototypes: pronunciationPrototypes, threshold: pronunciationThreshold,
                pronunciationRefiner: pronunciationRefiner, pronunciationEnabled: pronunciationEnabled
            )
            try Task.checkCancellation()
            pronunciationWindows.finalize(output.matches)
            state.recordFinalizedWindow(output.window)
        }
    }

    private func transcribeTailWindow() async throws -> [TokenWindow] {
        let work: ParakeetChunkWork
        if state.totalSampleCount <= layout.maxModelSamples {
            // Mirror short-utterance frame alignment and decoder end handling.
            let count = state.totalSampleCount
            let aligned = min(
                layout.maxModelSamples,
                ((count + ASRConstants.samplesPerEncoderFrame - 1) / ASRConstants.samplesPerEncoderFrame)
                    * ASRConstants.samplesPerEncoderFrame)
            let samples = state.retainedSamples + Array(repeating: Float(0), count: aligned - count)
            work = ParakeetChunkWork(
                samples: samples,
                paddedSamples: samples + Array(repeating: 0, count: layout.maxModelSamples - aligned),
                contextSamples: 0, chunkStart: 0, chunkEnd: count, isLastChunk: false
            )
        } else {
            guard let tail = try state.makeTailWork() else { return [] }
            work = tail
        }
        var decoderState = TdtDecoderState.make(decoderLayers: await manager.getDecoderLayers())
        decoderState.reset()
        let output = try await ParakeetChunkInference.transcribeWithPronunciation(
            work: work, using: manager, decoderState: &decoderState, prototypes: pronunciationPrototypes,
            threshold: pronunciationThreshold, pronunciationRefiner: pronunciationRefiner,
            pronunciationEnabled: pronunciationEnabled
        )
        try Task.checkCancellation()
        pronunciationWindows.replaceTail(output.matches)
        return output.window
    }

    private func makeResult(
        windows: [[TokenWindow]],
        processingTime: TimeInterval
    ) async -> ASRResult {
        var mergedTokens = AsrChunkTokenMerger.merge(
            windows,
            sampleRate: layout.sampleRate,
            overlapSeconds: layout.overlapSeconds
        )
        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }
        return await manager.processTranscriptionResult(
            tokenIds: mergedTokens.map(\.token),
            timestamps: mergedTokens.map(\.timestamp),
            confidences: mergedTokens.map(\.confidence),
            tokenDurations: mergedTokens.map(\.duration),
            encoderSequenceLength: 0,
            audioSamples: [],
            processingTime: processingTime,
            audioSampleCount: state.totalSampleCount
        )
    }
}

extension AsrManager {
    /// Creates an incremental session that is text-equivalent to batch Parakeet chunking.
    /// `pronunciationEnabled` is checked for each window; false skips pronunciation work without resetting speech.
    /// Pass the complete waveform to `finish(finalAudioSamples:)` when vocabulary boosting is configured.
    public func makeIncrementalSession(
        source: AudioSource = .microphone,
        pronunciationPrototypes: [PronunciationEmbedding] = [],
        pronunciationThreshold: Float = PronunciationCustomizationDefaults.acceptanceThreshold,
        pronunciationRefiner: PronunciationChunkRefiner? = nil,
        pronunciationEnabled: (@Sendable () -> Bool)? = nil
    ) throws -> ParakeetIncrementalSession {
        guard isAvailable else { throw ASRError.notInitialized }
        return ParakeetIncrementalSession(
            manager: self,
            source: source,
            appliesVocabularyBoosting: vocabBoostingEnabled,
            pronunciationPrototypes: pronunciationPrototypes,
            pronunciationThreshold: pronunciationThreshold, pronunciationRefiner: pronunciationRefiner,
            pronunciationEnabled: pronunciationEnabled
        )
    }
}
