import Foundation
import XCTest

@testable import FluidAudio

final class ParakeetIncrementalIntegrationTests: XCTestCase {
    private struct ParitySnapshot: Codable {
        let text: String
        let confidence: Float
        let duration: TimeInterval
        let tokenTimings: [TokenTiming]?
        let ctcDetectedTerms: [String]?
        let ctcAppliedTerms: [String]?
    }

    func testMultilingualRealAudioMatchesBatchAcrossFeedAndStopPatterns() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"]
        else {
            throw XCTSkip("Set the incremental fixture and model directories to run real-audio parity")
        }

        let fixtureURLs = try FileManager.default.contentsOfDirectory(
            at: URL(fileURLWithPath: fixturesPath),
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension.lowercased() == "wav" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        XCTAssertGreaterThanOrEqual(fixtureURLs.count, 8)

        let manager = try await Self.makeManager(modelPath: modelPath)
        let patterns = [
            (feedSamples: 9_600, finalTailSamples: 9_600),
            (feedSamples: 116_800, finalTailSamples: 320_000),
            (feedSamples: 239_999, finalTailSamples: 1),
        ]

        for fixtureURL in fixtureURLs {
            let samples = try AudioConverter().resampleAudioFile(path: fixtureURL.path)
            XCTAssertGreaterThan(samples.count, ASRConstants.maxModelSamples, fixtureURL.lastPathComponent)
            let batch = try await manager.transcribe(samples, source: .system)
            let expected = try Self.snapshot(batch)

            for pattern in patterns {
                let incremental = try await Self.transcribeIncrementally(
                    samples,
                    manager: manager,
                    feedSamples: pattern.feedSamples,
                    finalTailSamples: pattern.finalTailSamples
                )
                XCTAssertEqual(
                    try Self.snapshot(incremental),
                    expected,
                    "Parity failed for \(fixtureURL.lastPathComponent), feed=\(pattern.feedSamples), tail=\(pattern.finalTailSamples)"
                )
            }
        }
    }

    func testHighFrequencyPreviewAndNativeVocabularyBoostingMatchBatch() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"],
            let vocabularyPath = environment["FLUIDAUDIO_PARAKEET_VOCAB_PATH"]
        else {
            throw XCTSkip("Set fixture, model, and vocabulary paths to run boosted real-audio parity")
        }

        let fixtureURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("en_us.wav")
        let samples = try AudioConverter().resampleAudioFile(path: fixtureURL.path)
        let manager = try await Self.makeManager(modelPath: modelPath)

        let highFrequencyBatch = try await manager.transcribe(samples, source: .system)
        let highFrequencyIncremental = try await Self.transcribeIncrementally(
            samples,
            manager: manager,
            feedSamples: 1_280,
            finalTailSamples: 1
        )
        XCTAssertEqual(
            try Self.snapshot(highFrequencyIncremental),
            try Self.snapshot(highFrequencyBatch)
        )

        let (vocabulary, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(
            from: vocabularyPath
        )
        try await manager.configureVocabularyBoosting(vocabulary: vocabulary, ctcModels: ctcModels)
        let boostedBatch = try await manager.transcribe(samples, source: .system)
        let boostedIncremental = try await Self.transcribeIncrementally(
            samples,
            manager: manager,
            feedSamples: 9_600,
            finalTailSamples: 9_600
        )
        XCTAssertEqual(try Self.snapshot(boostedIncremental), try Self.snapshot(boostedBatch))
    }

    func testCancellationMakesPartiallyAcceptedSessionTerminal() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"]
        else {
            throw XCTSkip("Set the incremental fixture and model directories to run cancellation stress")
        }

        let samples = try AudioConverter().resampleAudioFile(
            path: URL(fileURLWithPath: fixturesPath).appendingPathComponent("en_us.wav").path
        )
        let manager = try await Self.makeManager(modelPath: modelPath)
        let session = try await manager.makeIncrementalSession(source: .system)
        try await session.append(Array(samples.prefix(ASRConstants.maxModelSamples)))

        let appendTask = Task {
            try await session.append(Array(samples.dropFirst(ASRConstants.maxModelSamples)))
        }
        appendTask.cancel()
        do {
            try await appendTask.value
            XCTFail("Expected cancellation")
        } catch is CancellationError {
            // The session must not reuse partially accepted state after cancellation.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        do {
            _ = try await session.preview()
            XCTFail("Expected terminal failure state")
        } catch let ASRError.processingFailed(message) {
            XCTAssertTrue(message.contains("previous failure"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    private static func makeManager(modelPath: String) async throws -> AsrManager {
        let models = try await AsrModels.load(
            from: URL(fileURLWithPath: modelPath),
            version: .v3
        )
        let manager = AsrManager(
            config: ASRConfig(
                tdtConfig: TdtConfig(blankId: AsrModelVersion.v3.blankId),
                encoderHiddenSize: AsrModelVersion.v3.encoderHiddenSize
            )
        )
        try await manager.initialize(models: models)
        return manager
    }

    private static func transcribeIncrementally(
        _ samples: [Float],
        manager: AsrManager,
        feedSamples: Int,
        finalTailSamples: Int
    ) async throws -> ASRResult {
        let session = try await manager.makeIncrementalSession(source: .system)
        let previewLimit = max(16_000, samples.count - min(finalTailSamples, samples.count - 16_000))
        var offset = 0
        while offset < previewLimit {
            let end = min(offset + feedSamples, previewLimit)
            try await session.append(Array(samples[offset..<end]))
            if end >= 16_000 {
                _ = try await session.preview()
            }
            offset = end
        }
        if offset < samples.count {
            try await session.append(Array(samples[offset...]))
        }
        return try await session.finish(finalAudioSamples: samples)
    }

    private static func snapshot(_ result: ASRResult) throws -> Data {
        let snapshot = ParitySnapshot(
            text: result.text,
            confidence: result.confidence,
            duration: result.duration,
            tokenTimings: result.tokenTimings,
            ctcDetectedTerms: result.ctcDetectedTerms,
            ctcAppliedTerms: result.ctcAppliedTerms
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(snapshot)
    }
}
