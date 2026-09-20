import CoreML
import Foundation

/// A contiguous copy of Parakeet encoder frames captured from the normal ASR inference pass.
public struct EncoderFeatureSequence: Sendable {
    public let hiddenSize: Int
    public let frameCount: Int
    public let frameDuration: TimeInterval
    public let globalFrameOffset: Int
    public let captureMicroseconds: Int
    public let values: [Float]

    public init(
        hiddenSize: Int,
        frameCount: Int,
        frameDuration: TimeInterval = 0.08,
        globalFrameOffset: Int = 0,
        captureMicroseconds: Int = 0,
        values: [Float]
    ) {
        precondition(values.count == hiddenSize * frameCount)
        self.hiddenSize = hiddenSize
        self.frameCount = frameCount
        self.frameDuration = frameDuration
        self.globalFrameOffset = globalFrameOffset
        self.captureMicroseconds = captureMicroseconds
        self.values = values
    }

    public func frame(at index: Int) -> ArraySlice<Float> {
        precondition(index >= 0 && index < frameCount)
        let start = index * hiddenSize
        return values[start..<(start + hiddenSize)]
    }
}

extension AsrManager {
    /// Enable or disable pronunciation customization feature capture.
    ///
    /// Capture is disabled by default. When disabled, Parakeet does not copy encoder frames and callers should not run
    /// pronunciation matching.
    public func setPronunciationCustomizationEnabled(_ enabled: Bool) {
        pronunciationCustomizationEnabled = enabled
        lastPronunciationEncoderFeatures = nil
    }

    /// Return and clear the encoder sequence captured for the most recent transcription.
    public func consumePronunciationEncoderFeatures() -> EncoderFeatureSequence? {
        defer { lastPronunciationEncoderFeatures = nil }
        return lastPronunciationEncoderFeatures
    }

    internal func capturePronunciationEncoderFeaturesIfEnabled(
        _ encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        contextFrameAdjustment: Int,
        globalFrameOffset: Int
    ) throws {
        guard pronunciationCustomizationEnabled else { return }
        lastPronunciationEncoderFeatures = try makePronunciationEncoderFeatures(
            encoderOutput, encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames, contextFrameAdjustment: contextFrameAdjustment,
            globalFrameOffset: globalFrameOffset
        )
    }

    internal func makePronunciationEncoderFeatures(
        _ encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        contextFrameAdjustment: Int,
        globalFrameOffset: Int
    ) throws -> EncoderFeatureSequence? {
        let captureStartedAt = ContinuousClock.now

        let view = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderSequenceLength,
            expectedHiddenSize: config.encoderHiddenSize
        )
        let firstFrame = min(max(0, contextFrameAdjustment), view.count)
        let frameCount = min(max(0, actualAudioFrames), view.count - firstFrame)
        guard frameCount > 0 else {
            return nil
        }

        var values = [Float](repeating: 0, count: frameCount * view.hiddenSize)
        try values.withUnsafeMutableBufferPointer { destination in
            guard let baseAddress = destination.baseAddress else { return }
            for frameIndex in 0..<frameCount {
                try view.copyFrame(
                    at: firstFrame + frameIndex,
                    into: baseAddress.advanced(by: frameIndex * view.hiddenSize),
                    destinationStride: 1
                )
            }
        }

        return EncoderFeatureSequence(
            hiddenSize: view.hiddenSize,
            frameCount: frameCount,
            globalFrameOffset: globalFrameOffset,
            captureMicroseconds: Self.microseconds(captureStartedAt.duration(to: .now)),
            values: values
        )
    }

    private static func microseconds(_ duration: Duration) -> Int {
        let seconds = Double(duration.components.seconds)
        let fractional = Double(duration.components.attoseconds) / 1e18
        return Int(((seconds + fractional) * 1_000_000).rounded())
    }
}
