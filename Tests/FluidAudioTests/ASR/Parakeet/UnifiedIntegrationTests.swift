import Foundation
import XCTest

@testable import FluidAudio

/// Opt in with a real JFK speech fixture and a model cache directory.
/// No microphone, test doubles or synthesized audio is used.
@MainActor
final class UnifiedIntegrationTests: XCTestCase {
    func testIncrementalTranscriptionAndReset() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let fixture = environment["FLUID_UNIFIED_TEST_AUDIO"],
            let modelCache = environment["FLUID_UNIFIED_TEST_MODELS"]
        else {
            throw XCTSkip("Set FLUID_UNIFIED_TEST_AUDIO and FLUID_UNIFIED_TEST_MODELS to run real-model validation")
        }
        let config = UnifiedConfig(chunkFrames: 7, rightFrames: 1)
        let engine = StreamingUnifiedAsrManager(config: config)
        try await engine.loadModels(to: URL(fileURLWithPath: modelCache))
        let samples = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: fixture))
        XCTAssertGreaterThan(samples.count, config.windowSamples)

        var partials: [String] = []
        for start in stride(from: 0, to: samples.count, by: 1600) {
            try await engine.appendAudio(Array(samples[start..<min(start + 1600, samples.count)]))
            try await engine.processBufferedAudio()
            let partial = await engine.getPartialTranscript()
            if !partial.isEmpty, partials.last != partial { partials.append(partial) }
        }
        XCTAssertGreaterThan(partials.count, 1, "Speech should appear before finish")
        let streamed = try await engine.finish()
        XCTAssertTrue(streamed.lowercased().contains("country"), streamed)
        XCTAssertTrue(streamed.hasPrefix(partials.first ?? ""))
        let finishedAgain = try await engine.finish()
        XCTAssertEqual(finishedAgain, streamed, "Finish should be idempotent")
        do {
            try await engine.appendAudio(Array(samples.prefix(1600)))
            XCTFail("A finished stream must require reset before accepting audio")
        } catch {}

        try await engine.reset()
        let resetTranscript = await engine.getPartialTranscript()
        XCTAssertEqual(resetTranscript, "")
        try await engine.appendAudio(samples)
        let replayed = try await engine.finish()
        XCTAssertEqual(replayed, streamed, "Chunk delivery must not change the transcript")

        try await engine.reset()
        let cancelled = Task {
            try await engine.appendAudio(samples)
            try await engine.processBufferedAudio()
        }
        cancelled.cancel()
        do {
            try await cancelled.value
            XCTFail("Decoding must honor cancellation")
        } catch is CancellationError {}
        try await engine.reset()
        try await engine.appendAudio(samples)
        let afterCancellation = try await engine.finish()
        XCTAssertEqual(afterCancellation, streamed, "A cancelled session must not contaminate the next recording")
        print("Unified real-model validation: \(partials.count) partials; final: \(streamed)")
        await engine.cleanup()
    }
}
