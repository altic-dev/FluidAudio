import XCTest

@testable import FluidAudio

final class ASRConfigTests: XCTestCase {
    func testDefaultParallelChunkConcurrencyIsOne() {
        XCTAssertEqual(ASRConfig.default.parallelChunkConcurrency, 1)
    }

    func testParallelChunkConcurrencyIsClampedToOne() {
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: 0).parallelChunkConcurrency, 1)
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: -3).parallelChunkConcurrency, 1)
    }

    func testParallelChunkConcurrencyPreservesExplicitValue() {
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: 2).parallelChunkConcurrency, 2)
    }

    func testPronunciationCaptureForcesSerialChunkProcessing() async {
        let manager = AsrManager(config: ASRConfig(parallelChunkConcurrency: 4))
        let configuredConcurrency = await manager.parallelChunkConcurrency
        XCTAssertEqual(configuredConcurrency, 4)

        await manager.setPronunciationCustomizationEnabled(true)
        let captureConcurrency = await manager.parallelChunkConcurrency
        XCTAssertEqual(captureConcurrency, 1)
    }
}
