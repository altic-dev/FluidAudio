import Foundation
import Testing

@testable import FluidAudio

@Suite("Incremental pronunciation candidates")
struct PronunciationStreamingTests {
    @Test("Repeated disjoint occurrences survive; overlapping alternatives collapse")
    func repeatedOccurrences() throws {
        // Numerical feature vectors, not synthesized audio or a mock model.
        let features = EncoderFeatureSequence(
            hiddenSize: 2, frameCount: 8,
            values: [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        let prototype = PronunciationEmbedding(values: [1, 0], sourceFrameCount: 2)
        let hits = PronunciationEmbeddingMatcher.allMatches(
            prototypes: [prototype], in: features,
            threshold: 0.99, windowFrameCounts: [[2]])
        #expect(hits[0].map(\.frameRange) == [0..<2, 5..<7])
        let best = try #require(
            PronunciationEmbeddingMatcher.bestMatches(
                prototypes: [prototype], in: features,
                windowFrameCounts: [[2]])[0])
        #expect(best.frameRange == hits[0][0].frameRange)
        #expect(best.score == hits[0][0].score)
        #expect(
            PronunciationEmbeddingMatcher.allMatches(prototypes: [prototype], in: features, threshold: 1.1)[0].isEmpty)
        #expect(PronunciationEmbeddingMatcher.allMatches(prototypes: [prototype], in: features, stride: 0)[0].isEmpty)
        #expect(PronunciationEmbeddingMatcher.allMatches(prototypes: [], in: features).isEmpty)
    }

    @Test("New tail replaces old tail; stable chunks are not rematched")
    func provisionalTailLifecycle() {
        var windows = PronunciationWindowMatches()
        let first = PronunciationWindowMatch(prototypeIndex: 0, score: 0.9, frameRange: 3..<8)
        let stale = PronunciationWindowMatch(prototypeIndex: 0, score: 0.9, frameRange: 170..<178)
        let revised = PronunciationWindowMatch(prototypeIndex: 0, score: 0.95, frameRange: 174..<182)
        windows.finalize([first])
        windows.replaceTail([stale])
        windows.replaceTail([revised])
        #expect(windows.all.map(\.frameRange) == [3..<8, 174..<182])
        windows.finalize([revised])
        #expect(windows.tail.isEmpty)
        #expect(windows.all.count == 2)
        windows.replaceTail([])
        #expect(windows.stable.count == 2)
    }
}

@Suite("Batched pronunciation matrix layout")
struct PronunciationMatrixLayoutTests {
    @Test("Non-square feature matrices keep scores and prototype routing")
    func matrixLayout() throws {
        let features = EncoderFeatureSequence(
            hiddenSize: 3, frameCount: 7,
            values: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0])
        let prototypes = [
            PronunciationEmbedding(values: [1, 0, 0], sourceFrameCount: 2),
            PronunciationEmbedding(values: [0, 1, 0], sourceFrameCount: 2),
            PronunciationEmbedding(values: [0, 0, 1], sourceFrameCount: 2),
        ]
        let hits = PronunciationEmbeddingMatcher.allMatches(
            prototypes: prototypes, in: features,
            threshold: 0.99, windowFrameCounts: [[2], [2], [2]])
        #expect(hits.map { $0.map(\.frameRange) } == [[0..<2], [2..<4], [4..<6]])
        #expect(hits.flatMap { $0 }.allSatisfy { $0.score == 1 })
        // Zero-norm windows must compact without overwriting a later SIMD lane or tail.
        let sparse = EncoderFeatureSequence(
            hiddenSize: 3, frameCount: 11,
            values: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        let compacted = PronunciationEmbeddingMatcher.allMatches(
            prototypes: prototypes, in: sparse,
            threshold: 0.99, windowFrameCounts: [[1], [1], [1]])
        #expect(compacted.map { $0.map(\.frameRange) } == [[2..<3, 3..<4], [6..<7, 7..<8], [9..<10, 10..<11]])
        let empty = PronunciationEmbeddingMatcher.allMatches(
            prototypes: prototypes,
            in: EncoderFeatureSequence(hiddenSize: 3, frameCount: 3, values: [Float](repeating: 0, count: 9)))
        #expect(empty.allSatisfy { $0.isEmpty })
    }
}
