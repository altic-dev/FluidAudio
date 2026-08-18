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

    func testShortAudioRepeatedPreviewsAndSessionsMatchBatch() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"],
            let vocabularyPath = environment["FLUIDAUDIO_PARAKEET_VOCAB_PATH"]
        else {
            throw XCTSkip("Set fixture, model, and vocabulary paths to run short-audio parity")
        }

        let fixtureURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("en_us.wav")
        let fixtureSamples = try AudioConverter().resampleAudioFile(path: fixtureURL.path)
        XCTAssertGreaterThanOrEqual(fixtureSamples.count, ASRConstants.maxModelSamples)
        let samples = Array(fixtureSamples.prefix(ASRConstants.maxModelSamples))
        let manager = try await Self.makeManager(modelPath: modelPath)

        let unboostedBatch = try await manager.transcribe(samples, source: .system)
        try await Self.assertRepeatedShortSessionsMatch(
            samples,
            manager: manager,
            expected: unboostedBatch
        )

        let (vocabulary, ctcModels) = try await CustomVocabularyContext.loadWithCtcTokens(
            from: vocabularyPath
        )
        try await manager.configureVocabularyBoosting(vocabulary: vocabulary, ctcModels: ctcModels)
        let boostedBatch = try await manager.transcribe(samples, source: .system)
        try await Self.assertRepeatedShortSessionsMatch(
            samples,
            manager: manager,
            expected: boostedBatch
        )
        let highFrequencyBoosted = try await Self.transcribeIncrementally(
            samples,
            manager: manager,
            feedSamples: 1_280,
            finalTailSamples: 1
        )
        XCTAssertEqual(try Self.snapshot(highFrequencyBoosted), try Self.snapshot(boostedBatch))
    }

    func testCriticalLengthBoundariesAndInterleavedSourcesMatchBatch() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"]
        else {
            throw XCTSkip("Set the incremental fixture and model directories to run boundary parity")
        }

        let fixtureURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("en_us.wav")
        let fixtureSamples = try AudioConverter().resampleAudioFile(path: fixtureURL.path)
        let sampleCounts = [
            16_000,
            16_001,
            6 * 16_000 + 317,
            ASRConstants.maxModelSamples - 1,
            ASRConstants.maxModelSamples,
            ASRConstants.maxModelSamples + 1,
        ]
        XCTAssertGreaterThanOrEqual(fixtureSamples.count, sampleCounts.max() ?? 0)
        let manager = try await Self.makeManager(modelPath: modelPath)

        for sampleCount in sampleCounts {
            let samples = Array(fixtureSamples.prefix(sampleCount))
            let batch = try await manager.transcribe(samples, source: .system)
            let expected = try Self.snapshot(batch)
            let feedSamples = min(max(16_000, sampleCount / 3), 73_117)

            for source in [AudioSource.microphone, .system] {
                let incremental = try await Self.transcribeIncrementally(
                    samples,
                    manager: manager,
                    source: source,
                    feedSamples: feedSamples,
                    finalTailSamples: 1
                )
                XCTAssertEqual(
                    try Self.snapshot(incremental),
                    expected,
                    "Boundary parity failed at \(sampleCount) samples for \(source)"
                )
            }
        }

        let boundarySamples = Array(fixtureSamples.prefix(ASRConstants.maxModelSamples))
        let boundaryBatch = try await manager.transcribe(boundarySamples, source: .system)
        let highFrequencyBoundary = try await Self.transcribeIncrementally(
            boundarySamples,
            manager: manager,
            feedSamples: 1_280,
            finalTailSamples: 1
        )
        XCTAssertEqual(try Self.snapshot(highFrequencyBoundary), try Self.snapshot(boundaryBatch))

        let interleavedSamples = Array(fixtureSamples.prefix(6 * 16_000 + 317))
        let expected = try Self.snapshot(
            await manager.transcribe(interleavedSamples, source: .system)
        )
        let microphoneSession = try await manager.makeIncrementalSession(source: .microphone)
        let systemSession = try await manager.makeIncrementalSession(source: .system)
        var offset = 0
        while offset < interleavedSamples.count {
            let end = min(offset + 16_000, interleavedSamples.count)
            let block = Array(interleavedSamples[offset..<end])
            try await microphoneSession.append(block)
            _ = try await microphoneSession.preview()
            try await systemSession.append(block)
            _ = try await systemSession.preview()
            offset = end
        }

        let microphoneResult = try await microphoneSession.finish(finalAudioSamples: interleavedSamples)
        let systemResult = try await systemSession.finish(finalAudioSamples: interleavedSamples)
        XCTAssertEqual(try Self.snapshot(microphoneResult), expected)
        XCTAssertEqual(try Self.snapshot(systemResult), expected)
    }

    func testRepeatedShortSessionsDoNotLeakStateOrMemory() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixturesPath = environment["FLUIDAUDIO_PARAKEET_INCREMENTAL_FIXTURES_DIR"],
            let modelPath = environment["FLUIDAUDIO_PARAKEET_MODEL_DIR"]
        else {
            throw XCTSkip("Set the incremental fixture and model directories to run session stress")
        }

        let fixtureURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("en_us.wav")
        let fixtureSamples = try AudioConverter().resampleAudioFile(path: fixtureURL.path)
        let samples = Array(fixtureSamples.prefix(6 * 16_000 + 317))
        let manager = try await Self.makeManager(modelPath: modelPath)
        let batch = try await manager.transcribe(samples, source: .system)
        let expected = try Self.snapshot(batch)
        var memoryReadings: [UInt64] = []

        for iteration in 0..<50 {
            let session = try await manager.makeIncrementalSession(source: .system)
            try await session.append(Array(samples.prefix(16_000)))
            _ = try await session.preview()
            try await session.append(Array(samples.dropFirst(16_000)))
            let result = try await session.finish(finalAudioSamples: samples)
            XCTAssertEqual(
                try Self.snapshot(result),
                expected,
                "Repeated-session parity failed at iteration \(iteration)"
            )

            if iteration >= 9, let memory = SystemInfo.currentResidentMemoryBytes() {
                memoryReadings.append(memory)
            }
        }

        let finalBatch = try await manager.transcribe(samples, source: .system)
        XCTAssertEqual(try Self.snapshot(finalBatch), expected)
        if let first = memoryReadings.first, let last = memoryReadings.last {
            let growth = last > first ? last - first : 0
            XCTAssertLessThanOrEqual(growth, 64 * 1_024 * 1_024, "Resident memory grew by \(growth) bytes")
        }
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
        let version: AsrModelVersion =
            ProcessInfo.processInfo.environment["FLUIDAUDIO_PARAKEET_MODEL_VERSION"] == "v2" ? .v2 : .v3
        let models = try await AsrModels.load(
            from: URL(fileURLWithPath: modelPath),
            version: version
        )
        let manager = AsrManager(
            config: ASRConfig(
                tdtConfig: TdtConfig(blankId: version.blankId),
                encoderHiddenSize: version.encoderHiddenSize
            )
        )
        try await manager.initialize(models: models)
        return manager
    }

    private static func transcribeIncrementally(
        _ samples: [Float],
        manager: AsrManager,
        source: AudioSource = .system,
        feedSamples: Int,
        finalTailSamples: Int
    ) async throws -> ASRResult {
        let session = try await manager.makeIncrementalSession(source: source)
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

    private static func assertRepeatedShortSessionsMatch(
        _ samples: [Float],
        manager: AsrManager,
        expected: ASRResult
    ) async throws {
        let expectedSnapshot = try snapshot(expected)
        let feedSamples = 3 * 16_000

        for sessionIndex in 0..<2 {
            let session = try await manager.makeIncrementalSession(source: .system)
            var offset = 0
            while offset < samples.count {
                let end = min(offset + feedSamples, samples.count)
                try await session.append(Array(samples[offset..<end]))
                _ = try await session.preview()
                offset = end
            }

            let result = try await session.finish(finalAudioSamples: samples)
            XCTAssertEqual(
                try snapshot(result),
                expectedSnapshot,
                "Short-audio parity failed for session \(sessionIndex)"
            )
        }
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
