import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class ASRTranscriptStabilizerTests: XCTestCase {
    func testMergeKeepsConfirmedPrefixAndUpdatesVolatileTail() async {
        let stabilizer = ASRTranscriptStabilizer(
            config: StreamingStabilizerConfig(
                windowSize: 2,
                emitWordBoundaries: true,
                maxWaitMilliseconds: 600
            )
        )

        let first = await stabilizer.merge(
            result([(1, "▁hello"), (2, "▁wor")]),
            nowMilliseconds: 0
        )
        XCTAssertEqual(first.confirmedText, "hello")
        XCTAssertEqual(first.volatileText, "wor")
        XCTAssertEqual(first.text, "hello wor")

        let revised = await stabilizer.merge(
            result([(1, "▁hello"), (3, "▁world")]),
            nowMilliseconds: 100
        )
        XCTAssertEqual(revised.confirmedText, "hello")
        XCTAssertEqual(revised.volatileText, "world")
        XCTAssertEqual(revised.text, "hello world")

        let stabilized = await stabilizer.merge(
            result([(1, "▁hello"), (3, "▁world"), (4, ".")]),
            nowMilliseconds: 200
        )
        XCTAssertEqual(stabilized.confirmedText, "hello world")
        XCTAssertEqual(stabilized.volatileText, ".")

        let finished = await stabilizer.finish(nowMilliseconds: 300)
        XCTAssertEqual(finished.text, "hello world.")
        XCTAssertEqual(finished.confirmedText, "hello world.")
        XCTAssertEqual(finished.volatileText, "")
    }

    func testResultWithoutTokenTimingsRemainsVolatile() async {
        let stabilizer = ASRTranscriptStabilizer()
        let update = await stabilizer.merge(
            ASRResult(
                text: "preview only",
                confidence: 0.5,
                duration: 1,
                processingTime: 0.1
            ),
            nowMilliseconds: 0
        )

        XCTAssertEqual(update.text, "preview only")
        XCTAssertEqual(update.confirmedText, "")
        XCTAssertEqual(update.volatileText, "preview only")
    }

    private func result(_ tokens: [(Int, String)]) -> ASRResult {
        ASRResult(
            text: tokens.map(\.1).joined().replacingOccurrences(of: "▁", with: " "),
            confidence: 0.9,
            duration: 1,
            processingTime: 0.1,
            tokenTimings: tokens.enumerated().map { index, token in
                TokenTiming(
                    token: token.1,
                    tokenId: token.0,
                    startTime: Double(index) * 0.1,
                    endTime: Double(index + 1) * 0.1,
                    confidence: 0.9
                )
            }
        )
    }
}
