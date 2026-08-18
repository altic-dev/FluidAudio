import Foundation

struct ParakeetChunkWork: Sendable {
    let samples: [Float]
    let paddedSamples: [Float]
    let contextSamples: Int
    let chunkStart: Int
    let chunkEnd: Int
    let isLastChunk: Bool
}

struct ParakeetChunkLayout: Sendable {
    static let standard = ParakeetChunkLayout()

    let sampleRate = 16_000
    let overlapSeconds = 2.0
    let melContextSamples = ASRConstants.samplesPerEncoderFrame

    var maxModelSamples: Int { ASRConstants.maxModelSamples }

    var chunkSamples: Int {
        let maxActualChunk = maxModelSamples - melContextSamples
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    var strideSamples: Int {
        let raw = max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    func isStableWindow(chunkStart: Int, availableSamples: Int) -> Bool {
        availableSamples > chunkStart + chunkSamples
    }

    func makeWork(
        totalSamples: Int,
        chunkStart: Int,
        chunkIndex: Int,
        readSamples: (_ offset: Int, _ count: Int) throws -> [Float]
    ) throws -> ParakeetChunkWork? {
        guard chunkStart < totalSamples else { return nil }

        let candidateEnd = chunkStart + chunkSamples
        let isLastChunk = candidateEnd >= totalSamples
        let chunkEnd = isLastChunk ? totalSamples : candidateEnd
        guard chunkEnd > chunkStart else { return nil }

        let contextSamples = chunkIndex > 0 ? melContextSamples : 0
        let contextStart = chunkStart - contextSamples
        let chunkLengthWithContext = chunkEnd - contextStart
        let samples = try readSamples(contextStart, chunkLengthWithContext)
        let paddedSamples =
            samples.count < maxModelSamples
            ? samples + Array(repeating: 0, count: maxModelSamples - samples.count)
            : samples

        return ParakeetChunkWork(
            samples: samples,
            paddedSamples: paddedSamples,
            contextSamples: contextSamples,
            chunkStart: chunkStart,
            chunkEnd: chunkEnd,
            isLastChunk: isLastChunk
        )
    }
}

enum ParakeetChunkInference {
    typealias TokenWindow = AsrChunkTokenMerger.TokenWindow

    static func transcribe(
        work: ParakeetChunkWork,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> [TokenWindow] {
        var preparedPreprocessor: PreparedParakeetPreprocessorHandle? =
            try await manager.prepareParakeetPreprocessorOutput(
                work.paddedSamples,
                originalLength: work.samples.count
            )
        var preparedEncoder: PreparedParakeetEncoderHandle?
        do {
            guard let preprocessor = preparedPreprocessor else {
                throw ASRError.processingFailed("Preprocessor output was not prepared")
            }
            preparedEncoder = try await manager.prepareParakeetEncoderOutput(
                preparedPreprocessor: preprocessor
            )
            preparedPreprocessor = nil
            guard let encoder = preparedEncoder else {
                throw ASRError.processingFailed("Encoder output was not prepared")
            }
            let output = try await transcribe(
                work: work,
                preparedEncoder: encoder,
                using: manager,
                decoderState: &decoderState
            )
            preparedEncoder = nil
            return try makeTokenWindow(from: output)
        } catch {
            if let preparedPreprocessor {
                await manager.discardParakeetPreprocessorOutput(preparedPreprocessor)
            }
            if let preparedEncoder {
                await manager.discardParakeetEncoderOutput(preparedEncoder)
            }
            throw error
        }
    }

    static func transcribe(
        work: ParakeetChunkWork,
        preparedEncoder: PreparedParakeetEncoderHandle,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        let actualAudioSamples = work.samples.count - work.contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)
        let globalFrameOffset = work.chunkStart / ASRConstants.samplesPerEncoderFrame
        let contextFrames = work.contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            preparedEncoder: preparedEncoder,
            paddedAudio: work.paddedSamples,
            originalLength: work.samples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,
            isLastChunk: work.isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        guard !hypothesis.isEmpty, encoderSequenceLength > 0 else {
            return ([], [], [], [])
        }
        return (
            hypothesis.ySequence,
            hypothesis.timestamps,
            hypothesis.tokenConfidences,
            hypothesis.tokenDurations
        )
    }

    static func makeTokenWindow(
        from output: (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int])
    ) throws -> [TokenWindow] {
        guard output.tokens.count == output.timestamps.count,
            output.tokens.count == output.confidences.count
        else {
            throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
        }

        let durations =
            output.durations.count == output.tokens.count
            ? output.durations : Array(repeating: 0, count: output.tokens.count)
        return zip(
            zip(zip(output.tokens, output.timestamps), output.confidences), durations
        ).map {
            (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }
    }
}
