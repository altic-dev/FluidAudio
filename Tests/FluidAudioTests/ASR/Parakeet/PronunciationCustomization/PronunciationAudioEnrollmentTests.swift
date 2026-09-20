import Foundation
import XCTest

@testable import FluidAudio

final class PronunciationAudioEnrollmentTests: XCTestCase {
    func testRejectsInvalidEvidenceBeforeLoadingModels() async throws {
        let manager = AsrManager()
        do {
            _ = try await manager.pronunciationEmbedding(audioSamples: [], focalSampleRange: 0..<1)
            XCTFail("Empty evidence must be rejected")
        } catch ASRError.processingFailed {
            // Validation precedes model access.
        }
        do {
            _ = try await manager.pronunciationEmbedding(audioSamples: [.nan], focalSampleRange: 0..<1)
            XCTFail("Non-finite evidence must be rejected")
        } catch ASRError.processingFailed {}
        do {
            _ = try await manager.pronunciationEmbedding(audioSamples: [0], focalSampleRange: 0..<2)
            XCTFail("Out-of-bounds evidence must be rejected")
        } catch ASRError.processingFailed {}
    }

    func testOriginalContextMatchesNormalEncoderAndDoesNotPublishCapture() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let audioPath = environment["FLUIDAUDIO_PRONUNCIATION_AUDIO"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"]
        else { throw XCTSkip("Set real pronunciation audio and Parakeet model paths") }
        let samples = Array(try AudioConverter().resampleAudioFile(path: audioPath).prefix(160_000))
        let models = try await AsrModels.load(from: URL(fileURLWithPath: modelPath), version: .v3)
        let manager = AsrManager(
            config: ASRConfig(tdtConfig: TdtConfig(blankId: AsrModelVersion.v3.blankId), encoderHiddenSize: 1024))
        try await manager.initialize(models: models)
        do {
            await manager.setPronunciationCustomizationEnabled(true)
            let result = try await manager.transcribe(samples, source: .microphone)
            let captured = await manager.consumePronunciationEncoderFeatures()
            let features = try XCTUnwrap(captured)
            await manager.setPronunciationCustomizationEnabled(false)
            let words = WordAudioChunkExtractor.words(from: result.tokenTimings ?? [])
            let word = try XCTUnwrap(words.first { $0.startTime > 1 && $0.endTime - $0.startTime >= 0.16 })
            let start = Int((word.startTime * 16_000).rounded(.down))
            let end = min(samples.count, Int((word.endTime * 16_000).rounded(.up)))
            let frames = (start / 1_280)..<min(features.frameCount, (end + 1_279) / 1_280)
            let expected = try XCTUnwrap(PronunciationEmbeddingMatcher.embedding(from: features, frameRange: frames))
            for _ in 0..<3 {
                let actual = try await manager.pronunciationEmbedding(
                    audioSamples: samples, focalSampleRange: start..<end)
                XCTAssertEqual(actual.sourceFrameCount, expected.sourceFrameCount)
                XCTAssertEqual(actual.values, expected.values)
                let unexpectedCapture = await manager.consumePronunciationEncoderFeatures()
                XCTAssertNil(unexpectedCapture)
                let encodersReleased = await manager.preparedParakeetEncoderOutputs.isEmpty
                let preprocessorsReleased = await manager.preparedParakeetPreprocessorOutputs.isEmpty
                XCTAssertTrue(encodersReleased)
                XCTAssertTrue(preprocessorsReleased)
            }
            let cancelled = Task {
                try Task.checkCancellation()
                return try await manager.pronunciationEmbedding(audioSamples: samples, focalSampleRange: start..<end)
            }
            cancelled.cancel()
            do {
                _ = try await cancelled.value
                XCTFail("Cancelled extraction must not publish evidence")
            } catch is CancellationError {}
            let afterCancellation = try await manager.pronunciationEmbedding(
                audioSamples: samples, focalSampleRange: start..<end)
            XCTAssertEqual(afterCancellation.values, expected.values)
            await manager.cleanup()
        } catch {
            await manager.cleanup()
            throw error
        }
    }
}
