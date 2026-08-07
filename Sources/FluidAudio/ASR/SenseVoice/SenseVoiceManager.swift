@preconcurrency import CoreML
import Foundation

/// Manager for SenseVoiceSmall transcription.
///
/// Pipeline: waveform -> preprocessor -> LFR features -> encoder+CTC logits
/// -> host greedy CTC decode.
public actor SenseVoiceManager {
    private let models: SenseVoiceModels
    private let language: Int32
    private let textNorm: Int32
    private static let logger = AppLogger(category: "SenseVoiceManager")

    public init(
        models: SenseVoiceModels,
        language: Int32 = SenseVoiceConfig.defaultLanguage,
        textNorm: Int32 = SenseVoiceConfig.defaultTextNorm
    ) {
        self.models = models
        self.language = language
        self.textNorm = textNorm
    }

    public static func load(
        precision: SenseVoiceEncoderPrecision = .fp16,
        language: Int32 = SenseVoiceConfig.defaultLanguage,
        textNorm: Int32 = SenseVoiceConfig.defaultTextNorm,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SenseVoiceManager {
        let models = try await SenseVoiceModels.downloadAndLoad(
            precision: precision,
            progressHandler: progressHandler
        )
        return SenseVoiceManager(models: models, language: language, textNorm: textNorm)
    }

    /// Transcribe a 16 kHz mono audio file.
    public func transcribe(audioURL: URL) throws -> String {
        let converter = AudioConverter(sampleRate: Double(SenseVoiceConfig.sampleRate))
        let samples = try converter.resampleAudioFile(audioURL)
        return try transcribe(audio: samples)
    }

    /// Transcribe 16 kHz mono float samples in [-1, 1].
    public func transcribe(audio: [Float]) throws -> String {
        let features = try runPreprocessor(audio: audio)
        let (logits, validFrames) = try runEncoder(features: features)
        return decode(logits: logits, validFrames: validFrames)
    }

    private func runPreprocessor(audio: [Float]) throws -> MLMultiArray {
        let waveform = try MLMultiArray(shape: [1, audio.count as NSNumber], dataType: .float32)
        let scale = SenseVoiceConfig.waveformScale
        let pointer = waveform.dataPointer.assumingMemoryBound(to: Float32.self)
        for index in 0..<audio.count {
            pointer[index] = audio[index] * scale
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: waveform)
        ])
        let output = try models.preprocessor.prediction(from: input)
        guard let features = output.featureValue(for: "features")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice preprocessor produced no features.")
        }
        return features
    }

    private func runEncoder(features: MLMultiArray) throws -> (MLMultiArray, Int) {
        let dimension = SenseVoiceConfig.featureDim
        var frameCount = features.shape[1].intValue
        if frameCount > SenseVoiceConfig.maxFrames {
            Self.logger.warning("Audio exceeds max length; truncating \(frameCount) to \(SenseVoiceConfig.maxFrames) frames")
            frameCount = SenseVoiceConfig.maxFrames
        }
        let bucket = SenseVoiceConfig.pickBucket(forFrames: frameCount)

        let speech = try MLMultiArray(
            shape: [1, bucket as NSNumber, dimension as NSNumber],
            dataType: .float32
        )
        let speechPointer = speech.dataPointer.assumingMemoryBound(to: Float32.self)
        memset(speechPointer, 0, bucket * dimension * MemoryLayout<Float32>.size)

        let copiedCount = frameCount * dimension
        if features.dataType == .float32 {
            memcpy(speechPointer, features.dataPointer, copiedCount * MemoryLayout<Float32>.size)
        } else {
            for index in 0..<copiedCount {
                speechPointer[index] = features[index].floatValue
            }
        }

        let lengths = try MLMultiArray(shape: [1], dataType: .int32)
        lengths[0] = NSNumber(value: frameCount)
        let languageArray = try MLMultiArray(shape: [1], dataType: .int32)
        languageArray[0] = NSNumber(value: language)
        let textNormArray = try MLMultiArray(shape: [1], dataType: .int32)
        textNormArray[0] = NSNumber(value: textNorm)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "speech": MLFeatureValue(multiArray: speech),
            "speech_lengths": MLFeatureValue(multiArray: lengths),
            "language": MLFeatureValue(multiArray: languageArray),
            "textnorm": MLFeatureValue(multiArray: textNormArray),
        ])
        let output = try models.encoder.prediction(from: input)
        guard let logits = output.featureValue(for: "ctc_logits")?.multiArrayValue else {
            throw ASRError.processingFailed("SenseVoice encoder produced no ctc_logits.")
        }
        return (logits, SenseVoiceConfig.numQueryTokens + frameCount)
    }

    private func decode(logits: MLMultiArray, validFrames: Int) -> String {
        let vocabularySize = logits.shape[2].intValue
        let frames = min(validFrames, logits.shape[1].intValue)
        var ids: [Int] = []
        var previous = -1

        func appendArgmax(valueAt: (Int) -> Float) {
            var best = 0
            var bestValue = valueAt(0)
            for token in 1..<vocabularySize {
                let value = valueAt(token)
                if value > bestValue {
                    bestValue = value
                    best = token
                }
            }
            if best != SenseVoiceConfig.blankId && best != previous {
                ids.append(best)
            }
            previous = best
        }

        if logits.dataType == .float32 {
            let pointer = logits.dataPointer.assumingMemoryBound(to: Float32.self)
            for frame in 0..<frames {
                let base = frame * vocabularySize
                appendArgmax { pointer[base + $0] }
            }
        } else {
            for frame in 0..<frames {
                appendArgmax { logits[[0, frame as NSNumber, $0 as NSNumber]].floatValue }
            }
        }

        return decodeCtcTokenIds(ids, vocabulary: models.vocabulary)
            .replacingOccurrences(of: "<\\|[^|]*\\|>", with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
