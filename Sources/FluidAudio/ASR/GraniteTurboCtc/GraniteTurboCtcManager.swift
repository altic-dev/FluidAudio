@preconcurrency import CoreML
import Foundation
import OSLog

private let turboCtcLogger = Logger(subsystem: "FluidAudio", category: "GraniteTurboCtcManager")

@available(macOS 14, iOS 17, *)
public struct GraniteTurboCtcToken: Sendable {
    public let id: Int
    /// Global encoder frame index; one frame covers 80 ms of audio.
    public let frameIndex: Int
    /// Log-softmax score of the winning label at this frame.
    public let logProbability: Float
}

@available(macOS 14, iOS 17, *)
public struct GraniteTurboCtcResult: Sendable {
    public let text: String
    public let tokens: [GraniteTurboCtcToken]
    public let durationSeconds: Double
    public let elapsedSeconds: Double
    public let windowCount: Int

    public var tokenIDs: [Int] { tokens.map(\.id) }

    public var realTimeFactorX: Double {
        durationSeconds / max(elapsedSeconds, 1e-6)
    }
}

/// Greedy CTC transcription with Granite Speech 5.0 470M TurboCTC.
///
/// The CoreML package has a fixed 10.24 s window, so audio is featurised once and
/// then sliced into whole windows. There is no overlap: CTC output frames are
/// 12.5 Hz and independent, so a window boundary costs at most one frame of
/// context rather than the duplicated text an autoregressive decoder would emit.
@available(macOS 14, iOS 17, *)
public actor GraniteTurboCtcManager {
    private var models: GraniteTurboCtcModels?
    private var featureExtractor: GraniteTurboCtcFeatureExtractor?
    private var inputBuffer: MLMultiArray?

    public init() {}

    public func loadModels(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws {
        let loaded = try await GraniteTurboCtcModels.load(from: directory, computeUnits: computeUnits)
        try adopt(models: loaded)
    }

    /// Take ownership of an already-loaded bundle (the `AsrModels` unified path).
    public func adopt(models loaded: GraniteTurboCtcModels) throws {
        models = loaded
        featureExtractor = try GraniteTurboCtcFeatureExtractor(
            manifest: loaded.manifest,
            melFilterbank: loaded.melFilterbank
        )
        inputBuffer = nil
    }

    public func transcribe(audioSamples: [Float]) async throws -> String {
        try await transcribeDetailed(audioSamples: audioSamples).text
    }

    public func transcribeDetailed(audioSamples: [Float]) async throws -> GraniteTurboCtcResult {
        guard let models, let featureExtractor else {
            throw GraniteTurboCtcError.invalidOutput("Granite TurboCTC models are not loaded")
        }
        guard !audioSamples.isEmpty else {
            throw GraniteTurboCtcError.invalidAudio("Audio is empty")
        }

        let started = Date()
        let manifest = models.manifest
        let features = try featureExtractor.extract(audio: audioSamples)

        let windowFrames = manifest.window.stackedFrames
        let subsample = manifest.encoderSubsample
        let buffer = try inputArray(frames: windowFrames, featureDim: manifest.featureDim)

        var frames: [(id: Int, logProbability: Float)] = []
        var windowCount = 0

        var start = 0
        while start < features.stackedFrames {
            let validFrames = min(windowFrames, features.stackedFrames - start)
            fill(buffer, from: features, start: start, validFrames: validFrames)

            let raw = try predict(model: models.model, input: buffer)
            // Frames past the audio are zero-padded; only keep what real samples produced.
            let validEncoderFrames = min(raw.count, (validFrames + subsample - 1) / subsample)
            frames.append(contentsOf: raw[0..<validEncoderFrames])

            windowCount += 1
            start += windowFrames
        }

        let collapsed = Self.collapseCTC(frames, blankTokenID: manifest.blankTokenID)
        let text = models.tokenizer.decode(collapsed.map(\.id))
        let duration = Double(audioSamples.count) / Double(manifest.sampleRate)
        let elapsed = Date().timeIntervalSince(started)

        let rtfx = String(format: "%.2f", duration / max(elapsed, 1e-6))
        let summary = "Granite TurboCTC: \(windowCount) windows, \(collapsed.count) tokens, \(rtfx)x realtime"
        turboCtcLogger.info("\(summary, privacy: .public)")

        return GraniteTurboCtcResult(
            text: text,
            tokens: collapsed,
            durationSeconds: duration,
            elapsedSeconds: elapsed,
            windowCount: windowCount
        )
    }

    // MARK: - Inference

    private func inputArray(frames: Int, featureDim: Int) throws -> MLMultiArray {
        if let inputBuffer {
            return inputBuffer
        }
        let array = try MLMultiArray(
            shape: [1, NSNumber(value: frames), NSNumber(value: featureDim)],
            dataType: .float32
        )
        inputBuffer = array
        return array
    }

    private func fill(
        _ buffer: MLMultiArray,
        from features: GraniteTurboCtcFeatureExtractor.Features,
        start: Int,
        validFrames: Int
    ) {
        let stride = features.featureDim
        let pointer = buffer.dataPointer.bindMemory(to: Float.self, capacity: buffer.count)
        let copyCount = validFrames * stride

        features.values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress! + start * stride, count: copyCount)
        }
        if copyCount < buffer.count {
            pointer.advanced(by: copyCount).update(repeating: 0, count: buffer.count - copyCount)
        }
    }

    private func predict(
        model: MLModel, input: MLMultiArray
    ) throws -> [(id: Int, logProbability: Float)] {
        let provider = try MLDictionaryFeatureProvider(dictionary: ["input_features": input])
        let output = try model.prediction(from: provider)
        guard let tokens = output.featureValue(for: "token_ids")?.multiArrayValue else {
            throw GraniteTurboCtcError.invalidOutput("Model did not return token_ids")
        }
        let confidence = output.featureValue(for: "confidence")?.multiArrayValue

        let tokenPtr = tokens.dataPointer.bindMemory(to: Int32.self, capacity: tokens.count)
        let confidencePtr = confidence.map {
            $0.dataPointer.bindMemory(to: Float.self, capacity: $0.count)
        }
        return (0..<tokens.count).map { index in
            (id: Int(tokenPtr[index]), logProbability: confidencePtr?[index] ?? 0)
        }
    }

    // MARK: - CTC

    /// Greedy CTC squash: drop repeated labels, then drop blanks. The surviving
    /// entry keeps the first frame of its run, which is the token's onset.
    static func collapseCTC(
        _ frames: [(id: Int, logProbability: Float)], blankTokenID: Int
    ) -> [GraniteTurboCtcToken] {
        var collapsed: [GraniteTurboCtcToken] = []
        var previous = -1
        for (frameIndex, frame) in frames.enumerated() {
            if frame.id != previous, frame.id != blankTokenID {
                collapsed.append(
                    GraniteTurboCtcToken(
                        id: frame.id,
                        frameIndex: frameIndex,
                        logProbability: frame.logProbability
                    ))
            }
            previous = frame.id
        }
        return collapsed
    }
}
