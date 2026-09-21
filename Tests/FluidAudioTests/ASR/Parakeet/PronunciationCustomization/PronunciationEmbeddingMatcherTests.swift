import Foundation
import Testing

@testable import FluidAudio

@Suite("Parakeet pronunciation embedding matcher")
struct PronunciationEmbeddingMatcherTests {
    @Test("A custom cutoff retains a near miss without changing the default")
    func customCutoffRetainsNearMiss() throws {
        // Encoder-space vectors only; this checks the score gate without an audio model.
        let features = EncoderFeatureSequence(hiddenSize: 2, frameCount: 1, values: [0.5, 0.8660254])
        let prototype = PronunciationEmbedding(values: [1, 0], sourceFrameCount: 1)
        let defaults = PronunciationEmbeddingMatcher.allMatches(
            prototypes: [prototype], in: features, windowFrameCounts: [[1]])
        let forgiving = PronunciationEmbeddingMatcher.allMatches(
            prototypes: [prototype], in: features, threshold: 0.45, windowFrameCounts: [[1]])
        let strict = PronunciationEmbeddingMatcher.allMatches(
            prototypes: [prototype], in: features, threshold: 0.6, windowFrameCounts: [[1]])
        #expect(defaults[0].isEmpty)
        #expect(strict[0].isEmpty)
        let match = try #require(forgiving[0].first)
        #expect(abs(match.score - 0.5) < 0.0001)
    }

    @Test("Pronunciation capture is explicitly opt-in")
    func captureOptInLifecycle() async {
        let manager = AsrManager()

        let initiallyEnabled = await manager.pronunciationCustomizationEnabled
        #expect(!initiallyEnabled)
        await manager.setPronunciationCustomizationEnabled(true)
        let enabled = await manager.pronunciationCustomizationEnabled
        #expect(enabled)
        await manager.setPronunciationCustomizationEnabled(false)
        let finallyEnabled = await manager.pronunciationCustomizationEnabled
        #expect(!finallyEnabled)
    }

    @Test("Best match finds the embedded frame pattern")
    func findsBestWindow() throws {
        let sequence = EncoderFeatureSequence(
            hiddenSize: 2,
            frameCount: 5,
            values: [
                0, 1,
                1, 0,
                1, 0,
                0, 1,
                0, 1,
            ]
        )
        let enrollmentSequence = EncoderFeatureSequence(
            hiddenSize: 2,
            frameCount: 2,
            values: [1, 0, 1, 0]
        )
        let enrollment = try #require(PronunciationEmbeddingMatcher.embedding(from: enrollmentSequence))

        let match = try #require(
            PronunciationEmbeddingMatcher.bestMatch(
                prototype: enrollment,
                in: sequence,
                windowFrameCounts: [2]
            ))

        #expect(match.frameRange == 1..<3)
        #expect(abs(match.score - 1) < 0.0001)
    }

    @Test("Prototype averages normalized enrollments")
    func averagesPrototype() throws {
        let first = PronunciationEmbedding(values: [1, 0], sourceFrameCount: 8)
        let second = PronunciationEmbedding(values: [0, 1], sourceFrameCount: 10)

        let prototype = try #require(PronunciationEmbeddingMatcher.prototype(from: [first, second]))

        #expect(prototype.sourceFrameCount == 9)
        #expect(abs(prototype.values[0] - 0.7071) < 0.001)
        #expect(abs(prototype.values[1] - 0.7071) < 0.001)
    }

    @Test("Pronunciation embeddings persist with Codable")
    func embeddingCodableRoundTrip() throws {
        let original = PronunciationEmbedding(values: [0.25, 0.75], sourceFrameCount: 8)

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(PronunciationEmbedding.self, from: encoded)

        #expect(decoded.values == original.values)
        #expect(decoded.sourceFrameCount == original.sourceFrameCount)
    }

    @Test("Batch matching returns one result per prototype")
    func batchMatchPreservesPrototypeCount() throws {
        let sequence = EncoderFeatureSequence(
            hiddenSize: 2,
            frameCount: 5,
            values: [
                0, 1,
                1, 0,
                1, 0,
                0, 1,
                0, 1,
            ]
        )
        let prototype = PronunciationEmbedding(values: [1, 0], sourceFrameCount: 2)

        let matches = PronunciationEmbeddingMatcher.bestMatches(
            prototypes: Array(repeating: prototype, count: 100),
            in: sequence,
            windowFrameCounts: Array(repeating: [2], count: 100)
        )

        #expect(matches.count == 100)
        #expect(matches.allSatisfy { $0?.frameRange == 1..<3 })
    }

    @Test("Batched matching agrees with scalar cosine scanning")
    func batchedMatchesScalarReference() throws {
        let sequence = EncoderFeatureSequence(
            hiddenSize: 3,
            frameCount: 7,
            values: [
                0.2, 0.8, 0.1,
                0.9, 0.1, 0.3,
                0.8, 0.2, 0.4,
                0.1, 0.7, 0.6,
                0.3, 0.6, 0.8,
                0.7, 0.3, 0.2,
                0.4, 0.5, 0.9,
            ]
        )
        let prototypes = try [0..<2, 2..<5, 4..<7].map { range in
            try #require(PronunciationEmbeddingMatcher.embedding(from: sequence, frameRange: range))
        }
        let windowCounts = [[2], [3], [2, 3]]

        let batched = PronunciationEmbeddingMatcher.bestMatches(
            prototypes: prototypes,
            in: sequence,
            windowFrameCounts: windowCounts
        )
        let scalar = zip(prototypes, windowCounts).map { prototype, counts in
            scalarBestMatch(prototype: prototype, in: sequence, windowFrameCounts: counts)
        }

        #expect(batched.count == scalar.count)
        for index in batched.indices {
            #expect(batched[index]?.frameRange == scalar[index]?.frameRange)
            #expect(abs((batched[index]?.score ?? 0) - (scalar[index]?.score ?? 0)) < 0.0001)
        }
    }

    private func scalarBestMatch(
        prototype: PronunciationEmbedding,
        in sequence: EncoderFeatureSequence,
        windowFrameCounts: [Int]
    ) -> PronunciationEmbeddingMatch? {
        var best: PronunciationEmbeddingMatch?
        for count in windowFrameCounts.sorted() {
            guard count > 0, count <= sequence.frameCount else { continue }
            for start in 0...(sequence.frameCount - count) {
                let range = start..<(start + count)
                guard let candidate = PronunciationEmbeddingMatcher.embedding(from: sequence, frameRange: range) else {
                    continue
                }
                let score = zip(prototype.values, candidate.values).reduce(Float(0)) { result, pair in
                    result + pair.0 * pair.1
                }
                if best == nil || score > best!.score {
                    best = PronunciationEmbeddingMatch(score: score, frameRange: range)
                }
            }
        }
        return best
    }
}
