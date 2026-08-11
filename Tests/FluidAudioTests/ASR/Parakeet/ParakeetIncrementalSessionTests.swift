import XCTest

@testable import FluidAudio

final class ParakeetIncrementalSessionTests: XCTestCase {
    private struct WorkSignature: Equatable {
        let chunkStart: Int
        let chunkEnd: Int
        let contextSamples: Int
        let sampleCount: Int
        let isLastChunk: Bool
        let firstSample: Float?
        let lastSample: Float?
    }

    func testStateMatchesBatchGeometryAcrossCriticalStopBoundaries() throws {
        let layout = ParakeetChunkLayout.standard
        let sampleCounts = [
            layout.maxModelSamples + 1,
            layout.strideSamples + layout.chunkSamples,
            layout.strideSamples + layout.chunkSamples + 1,
            60 * layout.sampleRate,
            100 * layout.sampleRate,
        ]

        for sampleCount in sampleCounts {
            let samples = Self.makeSamples(count: sampleCount)
            let incremental = try Self.incrementalWorks(
                samples: samples,
                feedSizes: [1, 1_279, 1_280, 15_999, 16_000, 73_117, 240_000]
            )
            let batch = try Self.batchWorks(samples: samples)

            XCTAssertEqual(
                incremental.map(Self.signature),
                batch.map(Self.signature),
                "Incremental geometry differed at \(sampleCount) samples"
            )
            for (incrementalWork, batchWork) in zip(incremental, batch) {
                XCTAssertEqual(incrementalWork.samples, batchWork.samples)
                XCTAssertEqual(incrementalWork.paddedSamples, batchWork.paddedSamples)
            }
        }
    }

    func testRandomizedFeedPartitionsMatchBatchGeometry() throws {
        let layout = ParakeetChunkLayout.standard
        let samples = Self.makeSamples(count: 180 * layout.sampleRate + 317)

        for seed in 1...25 {
            var generator = DeterministicGenerator(seed: UInt64(seed))
            var feedSizes: [Int] = []
            var remaining = samples.count
            while remaining > 0 {
                let size = min(remaining, generator.nextInt(in: 1...layout.maxModelSamples))
                feedSizes.append(size)
                remaining -= size
            }

            let incremental = try Self.incrementalWorks(samples: samples, feedSizes: feedSizes)
            let batch = try Self.batchWorks(samples: samples)
            XCTAssertEqual(incremental.map(Self.signature), batch.map(Self.signature), "Seed \(seed)")
        }
    }

    func testTwoHourStateKeepsOnlyBoundedAudioTail() throws {
        let layout = ParakeetChunkLayout.standard
        let totalSamples = 2 * 60 * 60 * layout.sampleRate
        let block = Array(repeating: Float.zero, count: layout.maxModelSamples)
        var state = ParakeetIncrementalState(layout: layout)
        var appended = 0
        var peakRetainedAfterProcessing = 0

        while appended < totalSamples {
            let count = min(block.count, totalSamples - appended)
            state.append(block.prefix(count))
            appended += count
            if state.totalSampleCount > layout.maxModelSamples {
                while state.hasStableWindow {
                    state.recordFinalizedWindow([])
                }
            }
            peakRetainedAfterProcessing = max(peakRetainedAfterProcessing, state.retainedSamples.count)
        }

        XCTAssertEqual(state.totalSampleCount, totalSamples)
        XCTAssertGreaterThan(state.finalizedWindows.count, 500)
        XCTAssertLessThanOrEqual(peakRetainedAfterProcessing, layout.maxModelSamples)
        XCTAssertLessThanOrEqual(state.retainedSamples.count, layout.chunkSamples)
        XCTAssertNotNil(try state.makeTailWork())
    }

    func testTooShortPreviewIsRetryableButInferenceFailureInvalidatesSession() async throws {
        let session = ParakeetIncrementalSession(
            manager: AsrManager(),
            source: .microphone,
            appliesVocabularyBoosting: false
        )
        try await session.append(Array(repeating: 0, count: 15_999))

        do {
            _ = try await session.preview()
            XCTFail("Expected short-audio rejection")
        } catch ASRError.invalidAudioData {
            // A caller can append the missing audio and retry.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        try await session.append([0])
        do {
            _ = try await session.preview()
            XCTFail("Expected the uninitialized manager to fail")
        } catch ASRError.notInitialized {
            // Model inference failures make the partially consumed session terminal.
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        do {
            try await session.append([0])
            XCTFail("Expected terminal failure state")
        } catch let ASRError.processingFailed(message) {
            XCTAssertTrue(message.contains("previous failure"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testMissingBoostingWaveformIsRetryable() async throws {
        let session = ParakeetIncrementalSession(
            manager: AsrManager(),
            source: .microphone,
            appliesVocabularyBoosting: true
        )
        try await session.append(Array(repeating: 0, count: 16_000))

        do {
            _ = try await session.finish()
            XCTFail("Expected complete-waveform validation")
        } catch let ASRError.processingFailed(message) {
            XCTAssertTrue(message.contains("complete waveform"))
        } catch {
            XCTFail("Unexpected error: \(error)")
        }

        try await session.append([0])
        let acceptedSampleCount = await session.acceptedSampleCount
        XCTAssertEqual(acceptedSampleCount, 16_001)
    }

    private static func incrementalWorks(
        samples: [Float],
        feedSizes: [Int]
    ) throws -> [ParakeetChunkWork] {
        let layout = ParakeetChunkLayout.standard
        var state = ParakeetIncrementalState(layout: layout)
        var works: [ParakeetChunkWork] = []
        var offset = 0

        for requestedSize in feedSizes where offset < samples.count {
            let end = min(offset + requestedSize, samples.count)
            state.append(samples[offset..<end])
            offset = end
            if state.totalSampleCount > layout.maxModelSamples {
                while state.hasStableWindow {
                    works.append(try XCTUnwrap(state.makeNextStableWork()))
                    state.recordFinalizedWindow([])
                }
            }
        }
        if offset < samples.count {
            state.append(samples[offset...])
            while state.hasStableWindow {
                works.append(try XCTUnwrap(state.makeNextStableWork()))
                state.recordFinalizedWindow([])
            }
        }
        if let tail = try state.makeTailWork() {
            works.append(tail)
        }
        return works
    }

    private static func batchWorks(samples: [Float]) throws -> [ParakeetChunkWork] {
        let layout = ParakeetChunkLayout.standard
        var works: [ParakeetChunkWork] = []
        var chunkStart = 0
        var chunkIndex = 0
        while let work = try layout.makeWork(
            totalSamples: samples.count,
            chunkStart: chunkStart,
            chunkIndex: chunkIndex,
            readSamples: { offset, count in Array(samples[offset..<(offset + count)]) }
        ) {
            works.append(work)
            if work.isLastChunk { break }
            chunkStart += layout.strideSamples
            chunkIndex += 1
        }
        return works
    }

    private static func makeSamples(count: Int) -> [Float] {
        (0..<count).map { Float($0 % 8_191) / 8_191 }
    }

    private static func signature(_ work: ParakeetChunkWork) -> WorkSignature {
        WorkSignature(
            chunkStart: work.chunkStart,
            chunkEnd: work.chunkEnd,
            contextSamples: work.contextSamples,
            sampleCount: work.samples.count,
            isLastChunk: work.isLastChunk,
            firstSample: work.samples.first,
            lastSample: work.samples.last
        )
    }
}

private struct DeterministicGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func nextInt(in range: ClosedRange<Int>) -> Int {
        self.state = self.state &* 6_364_136_223_846_793_005 &+ 1
        let width = UInt64(range.upperBound - range.lowerBound + 1)
        return range.lowerBound + Int(self.state % width)
    }
}
