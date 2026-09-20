import Foundation

extension AsrManager {
    /// Build one pronunciation example from an original recording context and its selected word/phrase.
    ///
    /// Input is mono 16 kHz PCM, at most 15 seconds, with a focal range in sample coordinates.
    /// Runs only preprocessing and the acoustic encoder. No decoder, shared capture flag, or
    /// most-recent-transcription state is used. Call after approval, outside dictation delivery.
    public func pronunciationEmbedding(
        audioSamples: [Float], focalSampleRange: Range<Int>
    ) async throws -> PronunciationEmbedding {
        guard !audioSamples.isEmpty, audioSamples.count <= ASRConstants.maxModelSamples,
            !focalSampleRange.isEmpty, focalSampleRange.lowerBound >= 0,
            focalSampleRange.upperBound <= audioSamples.count,
            audioSamples.allSatisfy(\.isFinite)
        else { throw ASRError.processingFailed("Invalid original pronunciation audio or focal range") }
        try Task.checkCancellation()
        let padded = audioSamples + Array(repeating: Float(0), count: ASRConstants.maxModelSamples - audioSamples.count)
        var preprocessor: PreparedParakeetPreprocessorHandle?
        var encoder: PreparedParakeetEncoderHandle?
        do {
            let prepared = try await prepareParakeetPreprocessorOutput(padded, originalLength: audioSamples.count)
            preprocessor = prepared
            try Task.checkCancellation()
            let encoded = try await prepareParakeetEncoderOutput(preparedPreprocessor: prepared)
            preprocessor = nil
            encoder = encoded
            try Task.checkCancellation()
            guard
                let features = try pronunciationFeatures(
                    preparedEncoder: encoded,
                    actualAudioFrames: ASRConstants.calculateEncoderFrames(from: audioSamples.count),
                    contextFrameAdjustment: 0, globalFrameOffset: 0
                )
            else { throw ASRError.processingFailed("Original pronunciation has no encoder frames") }
            let first = focalSampleRange.lowerBound / ASRConstants.samplesPerEncoderFrame
            let last = min(
                features.frameCount,
                (focalSampleRange.upperBound + ASRConstants.samplesPerEncoderFrame - 1)
                    / ASRConstants.samplesPerEncoderFrame)
            guard first < last,
                let embedding = PronunciationEmbeddingMatcher.embedding(from: features, frameRange: first..<last)
            else { throw ASRError.processingFailed("Original pronunciation has no usable focal embedding") }
            await discardParakeetEncoderOutput(encoded)
            encoder = nil
            try Task.checkCancellation()
            return embedding
        } catch {
            if let preprocessor { await discardParakeetPreprocessorOutput(preprocessor) }
            if let encoder { await discardParakeetEncoderOutput(encoder) }
            throw error
        }
    }

    /// Copies features from the caller-owned inference handle; the caller remains responsible for releasing it.
    func pronunciationFeatures(
        preparedEncoder handle: PreparedParakeetEncoderHandle,
        actualAudioFrames: Int, contextFrameAdjustment: Int, globalFrameOffset: Int
    ) throws -> EncoderFeatureSequence? {
        guard let output = preparedParakeetEncoderOutputs[handle.id] else {
            throw ASRError.processingFailed("Prepared pronunciation output is unavailable")
        }
        let encoder = try extractFeatureValue(
            from: output.encoderOutput, key: "encoder", errorMessage: "Invalid encoder output")
        let length = try extractFeatureValue(
            from: output.encoderOutput, key: "encoder_length", errorMessage: "Invalid encoder length")
        return try makePronunciationEncoderFeatures(
            encoder, encoderSequenceLength: length[0].intValue,
            actualAudioFrames: actualAudioFrames, contextFrameAdjustment: contextFrameAdjustment,
            globalFrameOffset: globalFrameOffset)
    }
}
