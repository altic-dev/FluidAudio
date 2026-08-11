import Foundation

/// An append-only Parakeet session that reuses finalized batch windows.
///
/// The session intentionally uses the same chunk layout, inference path, and overlap merger as
/// `AsrManager.transcribe`. Only the unfinished tail is reprocessed for previews and finalization.
/// Keep one session per recording and append only newly captured 16 kHz mono PCM samples.
public actor ParakeetIncrementalSession {
    private typealias TokenWindow = AsrChunkTokenMerger.TokenWindow

    private let manager: AsrManager
    private let source: AudioSource
    private let appliesVocabularyBoosting: Bool
    private let layout = ParakeetChunkLayout.standard

    private var retainedSamples: [Float] = []
    private var retainedSampleOffset = 0
    private var totalSampleCount = 0
    private var nextChunkStart = 0
    private var nextChunkIndex = 0
    private var finalizedWindows: [[TokenWindow]] = []
    private var lastPreviewSampleCount = -1
    private var lastPreviewResult: ASRResult?
    private var isFinished = false

    init(manager: AsrManager, source: AudioSource, appliesVocabularyBoosting: Bool) {
        self.manager = manager
        self.source = source
        self.appliesVocabularyBoosting = appliesVocabularyBoosting
    }

    /// Number of source samples accepted by this session.
    public var acceptedSampleCount: Int {
        totalSampleCount
    }

    /// Number of stable model windows that will not be inferred again.
    public var finalizedWindowCount: Int {
        finalizedWindows.count
    }

    /// Appends only the samples captured since the preceding call.
    public func append(_ newSamples: [Float]) async throws {
        guard !isFinished else {
            throw ASRError.processingFailed("Cannot append audio after incremental finalization")
        }
        guard !newSamples.isEmpty else { return }
        try Task.checkCancellation()

        retainedSamples.append(contentsOf: newSamples)
        totalSampleCount += newSamples.count
        lastPreviewResult = nil
        lastPreviewSampleCount = -1
        try await processStableWindows()
    }

    /// Produces a transcript for all accepted samples while retaining reusable stable windows.
    public func preview() async throws -> ASRResult {
        guard !isFinished else {
            throw ASRError.processingFailed("Incremental session is already finalized")
        }
        guard totalSampleCount >= layout.sampleRate else {
            throw ASRError.invalidAudioData
        }
        if lastPreviewSampleCount == totalSampleCount, let lastPreviewResult {
            return lastPreviewResult
        }

        let startedAt = Date()
        let result: ASRResult
        if totalSampleCount <= layout.maxModelSamples {
            result = try await manager.transcribe(retainedSamples, source: source)
        } else {
            try await processStableWindows()
            let tailWindow = try await transcribeTailWindow()
            result = await makeResult(
                windows: finalizedWindows + [tailWindow],
                processingTime: Date().timeIntervalSince(startedAt)
            )
        }

        lastPreviewSampleCount = totalSampleCount
        lastPreviewResult = result
        return result
    }

    /// Finalizes the session. If the last preview covered the same sample count, no model work is repeated.
    public func finish(finalAudioSamples: [Float]? = nil) async throws -> ASRResult {
        guard !isFinished else {
            throw ASRError.processingFailed("Incremental session is already finalized")
        }
        var result = try await preview()
        if appliesVocabularyBoosting, totalSampleCount > layout.maxModelSamples {
            guard let finalAudioSamples, finalAudioSamples.count == totalSampleCount else {
                throw ASRError.processingFailed(
                    "Vocabulary-boosted finalization requires the complete waveform"
                )
            }
            result = await manager.applyVocabularyRescoring(
                result: result,
                audioSamples: finalAudioSamples
            )
        }
        isFinished = true
        return result
    }

    private func processStableWindows() async throws {
        guard totalSampleCount > layout.maxModelSamples else { return }

        while layout.isStableWindow(
            chunkStart: nextChunkStart,
            availableSamples: totalSampleCount
        ) {
            try Task.checkCancellation()
            guard
                let work = try makeWork(
                    chunkStart: nextChunkStart,
                    chunkIndex: nextChunkIndex
                )
            else {
                break
            }

            var decoderState = TdtDecoderState.make(
                decoderLayers: await manager.getDecoderLayers()
            )
            decoderState.reset()
            let window = try await ParakeetChunkInference.transcribe(
                work: work,
                using: manager,
                decoderState: &decoderState
            )
            finalizedWindows.append(window)
            nextChunkStart += layout.strideSamples
            nextChunkIndex += 1
            trimRetainedPrefix()
        }
    }

    private func transcribeTailWindow() async throws -> [TokenWindow] {
        guard
            let work = try makeWork(
                chunkStart: nextChunkStart,
                chunkIndex: nextChunkIndex
            )
        else {
            return []
        }
        var decoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )
        decoderState.reset()
        return try await ParakeetChunkInference.transcribe(
            work: work,
            using: manager,
            decoderState: &decoderState
        )
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

    private func trimRetainedPrefix() {
        let requiredOffset = max(0, nextChunkStart - layout.melContextSamples)
        let removableCount = requiredOffset - retainedSampleOffset
        guard removableCount > 0 else { return }
        retainedSamples.removeFirst(removableCount)
        retainedSampleOffset = requiredOffset
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
            audioSampleCount: totalSampleCount
        )
    }
}

extension AsrManager {
    /// Creates an incremental session that is text-equivalent to batch Parakeet chunking.
    /// Pass the complete waveform to `finish(finalAudioSamples:)` when vocabulary boosting is configured.
    public func makeIncrementalSession(
        source: AudioSource = .microphone
    ) throws -> ParakeetIncrementalSession {
        guard isAvailable else { throw ASRError.notInitialized }
        return ParakeetIncrementalSession(
            manager: self,
            source: source,
            appliesVocabularyBoosting: vocabBoostingEnabled
        )
    }
}
