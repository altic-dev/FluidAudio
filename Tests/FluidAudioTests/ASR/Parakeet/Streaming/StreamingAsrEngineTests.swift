import XCTest

@testable import FluidAudio

final class StreamingAsrEngineTests: XCTestCase {

    // MARK: - Protocol Conformance (compile-time verification)

    func testStreamingEouAsrManagerConformsToProtocol() async {
        let engine: any StreamingAsrEngine = StreamingEouAsrManager()
        let name = await engine.displayName
        XCTAssertFalse(name.isEmpty)
    }

    func testNemotronStreamingAsrManagerConformsToProtocol() async {
        let engine: any StreamingAsrEngine = NemotronStreamingAsrManager()
        let name = await engine.displayName
        XCTAssertFalse(name.isEmpty)
    }

    func testSlidingWindowAsrManagerDoesNotConformToProtocol() {
        // SlidingWindowAsrManager (TDT) is intentionally NOT a StreamingAsrEngine.
        // TDT uses a sliding-window approach with an offline encoder, not true streaming.
        // This test documents the design decision. If it fails to compile, TDT conformance
        // was accidentally added back.
        let _: SlidingWindowAsrManager = SlidingWindowAsrManager()
        // Uncomment to verify compile error: let _: any StreamingAsrEngine = SlidingWindowAsrManager()
    }

    // MARK: - StreamingModelVariant Tests

    func testAllVariantsCount() {
        // 3 EOU + 2 Nemotron + 4 Unified streaming variants
        XCTAssertEqual(StreamingModelVariant.allCases.count, 9)
    }

    func testAllVariantsHaveDisplayName() {
        for variant in StreamingModelVariant.allCases {
            XCTAssertFalse(variant.displayName.isEmpty, "Variant \(variant) has empty displayName")
        }
    }

    func testAllVariantsHaveRepo() {
        for variant in StreamingModelVariant.allCases {
            let repo = variant.repo
            XCTAssertFalse(repo.rawValue.isEmpty, "Variant \(variant) has empty repo")
        }
    }

    func testRawValueRoundTrip() {
        for variant in StreamingModelVariant.allCases {
            let reconstructed = StreamingModelVariant(rawValue: variant.rawValue)
            XCTAssertEqual(reconstructed, variant)
        }
    }

    func testEngineFamilyGrouping() {
        let eouVariants = StreamingModelVariant.allCases.filter {
            $0.engineFamily == .parakeetEou
        }
        let nemotronVariants = StreamingModelVariant.allCases.filter {
            $0.engineFamily == .nemotron
        }

        XCTAssertEqual(eouVariants.count, 3, "Expected 3 EOU variants")
        XCTAssertEqual(nemotronVariants.count, 2, "Expected 2 Nemotron variants")
    }

    func testEouVariantsHaveChunkSize() {
        for variant in StreamingModelVariant.allCases where variant.engineFamily == .parakeetEou {
            XCTAssertNotNil(
                variant.eouChunkSize, "EOU variant \(variant) should have eouChunkSize")
        }
    }

    func testNemotronVariantsHaveChunkSize() {
        for variant in StreamingModelVariant.allCases where variant.engineFamily == .nemotron {
            XCTAssertNotNil(
                variant.nemotronChunkSize, "Nemotron variant \(variant) should have nemotronChunkSize")
        }
    }

    func testEouVariantsDoNotHaveNemotronChunkSize() {
        for variant in StreamingModelVariant.allCases where variant.engineFamily == .parakeetEou {
            XCTAssertNil(variant.nemotronChunkSize)
        }
    }

    func testNemotronVariantsDoNotHaveEouChunkSize() {
        for variant in StreamingModelVariant.allCases where variant.engineFamily == .nemotron {
            XCTAssertNil(variant.eouChunkSize)
        }
    }

    // MARK: - Factory Tests

    func testFactoryCreatesEouEngine() async {
        let engine = StreamingAsrEngineFactory.create(.parakeetEou160ms)
        XCTAssertTrue(engine is StreamingEouAsrManager)
    }

    func testFactoryCreatesNemotronEngine() async {
        let engine = StreamingAsrEngineFactory.create(.nemotron1120ms)
        XCTAssertTrue(engine is NemotronStreamingAsrManager)
    }

    func testFactoryCreatesAllVariants() async {
        for variant in StreamingModelVariant.allCases {
            let engine = StreamingAsrEngineFactory.create(variant)
            let name = await engine.displayName
            XCTAssertFalse(name.isEmpty, "Engine for \(variant) has empty displayName")
        }
    }

    // MARK: - Engine Initial State Tests

    func testEouEngineInitialState() async {
        let engine = StreamingAsrEngineFactory.create(.parakeetEou320ms)
        let partial = await engine.getPartialTranscript()
        XCTAssertEqual(partial, "")
    }

    func testNemotronEngineInitialState() async {
        let engine = StreamingAsrEngineFactory.create(.nemotron560ms)
        let partial = await engine.getPartialTranscript()
        XCTAssertEqual(partial, "")
    }
}
