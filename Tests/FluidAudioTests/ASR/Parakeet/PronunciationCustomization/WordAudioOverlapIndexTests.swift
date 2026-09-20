import Foundation
import Testing

@testable import FluidAudio

@Suite("Indexed acoustic overlap")
struct WordAudioOverlapIndexTests {
    private func word(_ start: Double, _ end: Double) -> TimedTranscriptWord {
        TimedTranscriptWord(text: "word", startTime: start, endTime: end, confidence: 1)
    }

    @Test("Indexed queries preserve order, repeated words, nested spans and majority boundaries")
    func indexedParity() {
        let words = [word(4, 5), word(0, 10), word(1, 2), word(1, 2), word(3, 3), word(6, 5), word(8, 9)]
        let index = WordAudioOverlapIndex(words: words)
        for start in stride(from: -1.0, through: 11.0, by: 0.25) {
            for length in [0.0, 0.25, 0.5, 1, 3, 12] {
                for ratio in [-1.0, 0, 0.5, 1, 2, Double.nan] {
                    #expect(
                        index.substantiallyOverlappingWordIndices(
                            startTime: start, endTime: start + length, minimumOverlapRatio: ratio
                        )
                            == WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
                                in: words, startTime: start, endTime: start + length, minimumOverlapRatio: ratio))
                }
            }
        }
        #expect(index.substantiallyOverlappingWordIndices(startTime: 1, endTime: 1.5).isEmpty)
        #expect(index.substantiallyOverlappingWordIndices(startTime: 1, endTime: 2) == [2, 3])
        #expect(WordAudioOverlapIndex(words: []).substantiallyOverlappingWordIndices(startTime: 0, endTime: 1).isEmpty)
    }

    @Test("Malformed non-finite timestamps retain existing extractor behavior")
    func nonFiniteParity() {
        for words in [[word(0, 1)], [word(.nan, 2), word(1, .infinity), word(-.infinity, 1), word(3, 4)]] {
            let index = WordAudioOverlapIndex(words: words)
            for (start, end) in [(0.0, 4.0), (-Double.infinity, Double.infinity), (Double.nan, 1), (0, Double.nan)] {
                #expect(
                    index.substantiallyOverlappingWordIndices(startTime: start, endTime: end)
                        == WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
                            in: words, startTime: start, endTime: end))
            }
        }
    }
}
