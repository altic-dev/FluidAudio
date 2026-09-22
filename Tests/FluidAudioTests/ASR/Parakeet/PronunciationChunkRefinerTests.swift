import Foundation
import XCTest
@testable import FluidAudio

final class PronunciationChunkRefinerTests: XCTestCase {
    actor Seen {
        var offsets: [Int] = []
        func record(_ offset: Int) { offsets.append(offset) }
        func snapshot() -> [Int] { offsets }
    }

    func testRefinerUsesExistingFramesAndMapsLocalMatchesWithoutChangingText() async throws {
        guard let path = ProcessInfo.processInfo.environment["FLUIDAUDIO_REFINER_PCM"] else {
            throw XCTSkip("Set FLUIDAUDIO_REFINER_PCM to a short 16 kHz Float32 recording")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let one = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let samples = one + one + one
        let models = try await AsrModels.loadLocalOnly(from: AsrModels.defaultCacheDirectory(for: .v2), version: .v2)
        let manager = AsrManager(
            config: ASRConfig(tdtConfig: TdtConfig(blankId: AsrModelVersion.v2.blankId), encoderHiddenSize: 1024))
        try await manager.initialize(models: models)
        let baseline = try await manager.transcribe(samples, source: .microphone)
        let prototype = PronunciationEmbedding(
            values: [Float](repeating: 1 / sqrt(1024), count: 1024), sourceFrameCount: 10)
        let seen = Seen()
        let session = try await manager.makeIncrementalSession(
            pronunciationPrototypes: [prototype],
            pronunciationRefiner: { chunk in
                XCTAssertEqual(chunk.features.globalFrameOffset, 0)
                XCTAssertEqual(chunk.features.values.count, chunk.features.frameCount * 1024)
                XCTAssertFalse(chunk.samples.isEmpty)
                XCTAssertLessThanOrEqual(chunk.samples.count, 240_000)
                XCTAssertTrue(
                    chunk.result.tokenTimings?.allSatisfy {
                        $0.startTime >= 0 && $0.endTime <= Double(chunk.samples.count) / 16000 + 0.16
                    } ?? false)
                await seen.record(chunk.sampleOffset)
                return [.init(prototypeIndex: 0, score: 0.75, frameRange: 1..<2)]
            })
        for lo in stride(from: 0, to: samples.count, by: 16000) {
            try await session.append(Array(samples[lo..<min(samples.count, lo + 16000)]))
        }
        let result = try await session.finish()
        XCTAssertEqual(result.text, baseline.text)
        let offsets = await seen.snapshot()
        XCTAssertGreaterThan(offsets.count, 1)
        let matches = await session.pronunciationMatches
        XCTAssertEqual(matches.map(\.frameRange), offsets.map { ($0 / 1280 + 1)..<($0 / 1280 + 2) })
        do {
            _ = try await session.finish()
            XCTFail("A finalized session cannot restart")
        } catch {}
        let after = await seen.snapshot()
        XCTAssertEqual(after, offsets, "Finished sessions must not rerun the refiner")
        let cancelled = try await manager.makeIncrementalSession(
            pronunciationPrototypes: [prototype], pronunciationRefiner: { _ in throw CancellationError() })
        do {
            try await cancelled.append(samples)
            _ = try await cancelled.finish()
            XCTFail("Refiner cancellation must propagate")
        } catch is CancellationError {} catch { XCTFail("Unexpected cancellation error: \(error)") }
        do {
            _ = try await cancelled.finish()
            XCTFail("A cancelled session cannot restart")
        } catch {}
        let gateKey = "PronunciationChunkRefinerTests-" + UUID().uuidString
        defer { UserDefaults.standard.removeObject(forKey: gateKey) }
        UserDefaults.standard.set(false, forKey: gateKey)
        let gatedSeen = Seen()
        let gated = try await manager.makeIncrementalSession(
            pronunciationPrototypes: [prototype],
            pronunciationRefiner: { chunk in
                await gatedSeen.record(chunk.sampleOffset)
                return [.init(prototypeIndex: 0, score: 0.75, frameRange: 1..<2)]
            },
            pronunciationEnabled: { UserDefaults.standard.bool(forKey: gateKey) }
        )
        try await gated.append(samples)
        let gatedResult = try await gated.finish()
        XCTAssertEqual(gatedResult.text, baseline.text, "Off must preserve ordinary recognition")
        let gatedOffsets = await gatedSeen.snapshot()
        let gatedMatches = await gated.pronunciationMatches
        XCTAssertTrue(gatedOffsets.isEmpty, "Off must never call the audio refiner")
        XCTAssertTrue(gatedMatches.isEmpty)

        UserDefaults.standard.set(true, forKey: gateKey)
        let switchedSeen = Seen()
        let switched = try await manager.makeIncrementalSession(
            pronunciationPrototypes: [prototype],
            pronunciationRefiner: { chunk in
                await switchedSeen.record(chunk.sampleOffset)
                UserDefaults.standard.set(false, forKey: gateKey)
                return [.init(prototypeIndex: 0, score: 0.75, frameRange: 1..<2)]
            },
            pronunciationEnabled: { UserDefaults.standard.bool(forKey: gateKey) }
        )
        try await switched.append(samples)
        let switchedResult = try await switched.finish()
        XCTAssertEqual(switchedResult.text, baseline.text, "Switching off must not reset or truncate recognition")
        let switchedOffsets = await switchedSeen.snapshot()
        let switchedMatches = await switched.pronunciationMatches
        XCTAssertEqual(switchedOffsets.count, 1, "Later windows must skip the refiner after off")
        XCTAssertTrue(switchedMatches.isEmpty, "In-flight and cached matches must be suppressed after off")
        await manager.cleanup()
    }
}
