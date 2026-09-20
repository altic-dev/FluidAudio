import Accelerate
import Foundation

/// Tunable defaults used by the pronunciation customization experiment.
public enum PronunciationCustomizationDefaults {
    public static let acceptanceThreshold: Float = 0.70
    public static let minimumWordOverlapRatio = 0.50
}

/// A persistable pronunciation vector produced from one or more Parakeet encoder frames.
public struct PronunciationEmbedding: Codable, Sendable {
    public let values: [Float]
    public let sourceFrameCount: Int

    public init(values: [Float], sourceFrameCount: Int) {
        self.values = values
        self.sourceFrameCount = sourceFrameCount
    }
}

public struct PronunciationEmbeddingMatch: Sendable {
    public let score: Float
    public let frameRange: Range<Int>

    public init(score: Float, frameRange: Range<Int>) {
        self.score = score
        self.frameRange = frameRange
    }
}

/// Batched mean-pooled, L2-normalized prototype matching over frozen Parakeet encoder frames.
public enum PronunciationEmbeddingMatcher {
    public static func embedding(
        from sequence: EncoderFeatureSequence,
        frameRange: Range<Int>? = nil
    ) -> PronunciationEmbedding? {
        let range = frameRange ?? 0..<sequence.frameCount
        guard !range.isEmpty, range.lowerBound >= 0, range.upperBound <= sequence.frameCount else { return nil }

        var mean = [Float](repeating: 0, count: sequence.hiddenSize)
        for frameIndex in range {
            let frame = sequence.frame(at: frameIndex)
            for hiddenIndex in 0..<sequence.hiddenSize {
                mean[hiddenIndex] += frame[frame.startIndex + hiddenIndex]
            }
        }
        let scale = 1 / Float(range.count)
        for index in mean.indices {
            mean[index] *= scale
        }
        guard normalize(&mean) else { return nil }
        return PronunciationEmbedding(values: mean, sourceFrameCount: range.count)
    }

    public static func prototype(from embeddings: [PronunciationEmbedding]) -> PronunciationEmbedding? {
        guard let first = embeddings.first, !first.values.isEmpty else { return nil }
        guard embeddings.allSatisfy({ $0.values.count == first.values.count }) else { return nil }

        var mean = [Float](repeating: 0, count: first.values.count)
        for embedding in embeddings {
            for index in mean.indices {
                mean[index] += embedding.values[index]
            }
        }
        let scale = 1 / Float(embeddings.count)
        for index in mean.indices {
            mean[index] *= scale
        }
        guard normalize(&mean) else { return nil }
        let averageFrames = Int(
            (Double(embeddings.reduce(0) { $0 + $1.sourceFrameCount }) / Double(embeddings.count)).rounded()
        )
        return PronunciationEmbedding(values: mean, sourceFrameCount: averageFrames)
    }

    public static func bestMatch(
        prototype: PronunciationEmbedding,
        in sequence: EncoderFeatureSequence,
        windowFrameCounts: [Int]? = nil,
        stride: Int = 1
    ) -> PronunciationEmbeddingMatch? {
        bestMatches(
            prototypes: [prototype],
            in: sequence,
            windowFrameCounts: windowFrameCounts.map { [$0] },
            stride: stride
        ).first ?? nil
    }

    /// Match several prototypes while building the encoder-frame prefix index only once.
    public static func bestMatches(
        prototypes: [PronunciationEmbedding],
        in sequence: EncoderFeatureSequence,
        windowFrameCounts: [[Int]]? = nil,
        stride: Int = 1
    ) -> [PronunciationEmbeddingMatch?] {
        var best = [PronunciationEmbeddingMatch?](repeating: nil, count: prototypes.count)
        scanMatches(prototypes: prototypes, in: sequence, windowFrameCounts: windowFrameCounts, stride: stride) {
            index, match in
            if best[index] == nil || match.score > best[index]!.score { best[index] = match }
        }
        return best
    }

    /// Return repeated, non-overlapping occurrences for each prototype, strongest first.
    public static func allMatches(
        prototypes: [PronunciationEmbedding],
        in sequence: EncoderFeatureSequence,
        threshold: Float = PronunciationCustomizationDefaults.acceptanceThreshold,
        windowFrameCounts: [[Int]]? = nil,
        stride: Int = 1
    ) -> [[PronunciationEmbeddingMatch]] {
        var candidates = [[PronunciationEmbeddingMatch]](repeating: [], count: prototypes.count)
        guard threshold.isFinite else { return candidates }
        scanMatches(prototypes: prototypes, in: sequence, windowFrameCounts: windowFrameCounts, stride: stride) {
            index, match in
            if match.score.isFinite && match.score >= threshold { candidates[index].append(match) }
        }
        return candidates.map { matches in
            var accepted: [PronunciationEmbeddingMatch] = []
            for match in matches.sorted(by: {
                if $0.score != $1.score { return $0.score > $1.score }
                if $0.frameRange.lowerBound != $1.frameRange.lowerBound {
                    return $0.frameRange.lowerBound < $1.frameRange.lowerBound
                }
                return $0.frameRange.count < $1.frameRange.count
            }) {
                guard !accepted.contains(where: { $0.frameRange.overlaps(match.frameRange) }) else { continue }
                accepted.append(match)
            }
            return accepted
        }
    }

    private static func scanMatches(
        prototypes: [PronunciationEmbedding],
        in sequence: EncoderFeatureSequence,
        windowFrameCounts: [[Int]]?,
        stride: Int,
        visit: (Int, PronunciationEmbeddingMatch) -> Void
    ) {
        guard stride > 0, !prototypes.isEmpty else { return }
        let prefix = prefixSums(for: sequence)
        let requestedCounts: [Set<Int>] = prototypes.enumerated().map { index, prototype in
            guard prototype.values.count == sequence.hiddenSize else { return [] }
            let requestedCounts =
                windowFrameCounts?[safe: index]
                ?? nearbyWindowCounts(around: prototype.sourceFrameCount)
            return Set(requestedCounts.filter { $0 > 0 && $0 <= sequence.frameCount })
        }
        let allCounts = Set(requestedCounts.flatMap { $0 }).sorted()

        for count in allCounts {
            let prototypeIndices = prototypes.indices.filter { requestedCounts[$0].contains(count) }
            guard !prototypeIndices.isEmpty else { continue }
            let candidates = normalizedWindows(
                in: sequence,
                prefix: prefix,
                frameCount: count,
                stride: stride
            )
            guard !candidates.ranges.isEmpty else { continue }

            var prototypeValues: [Float] = []
            prototypeValues.reserveCapacity(prototypeIndices.count * sequence.hiddenSize)
            for index in prototypeIndices {
                prototypeValues.append(contentsOf: prototypes[index].values)
            }
            var transposedCandidates = [Float](repeating: 0, count: candidates.values.count)
            for candidateIndex in candidates.ranges.indices {
                for hiddenIndex in 0..<sequence.hiddenSize {
                    transposedCandidates[hiddenIndex * candidates.ranges.count + candidateIndex] =
                        candidates.values[candidateIndex * sequence.hiddenSize + hiddenIndex]
                }
            }
            var scores = [Float](repeating: 0, count: prototypeIndices.count * candidates.ranges.count)
            prototypeValues.withUnsafeBufferPointer { prototypePointer in
                transposedCandidates.withUnsafeBufferPointer { candidatePointer in
                    scores.withUnsafeMutableBufferPointer { scorePointer in
                        guard
                            let prototypeBase = prototypePointer.baseAddress,
                            let candidateBase = candidatePointer.baseAddress,
                            let scoreBase = scorePointer.baseAddress
                        else { return }
                        vDSP_mmul(
                            prototypeBase,
                            1,
                            candidateBase,
                            1,
                            scoreBase,
                            1,
                            vDSP_Length(prototypeIndices.count),
                            vDSP_Length(candidates.ranges.count),
                            vDSP_Length(sequence.hiddenSize)
                        )
                    }
                }
            }

            for (prototypeOffset, prototypeIndex) in prototypeIndices.enumerated() {
                let scoreOffset = prototypeOffset * candidates.ranges.count
                for candidateIndex in candidates.ranges.indices {
                    let score = scores[scoreOffset + candidateIndex]
                    visit(
                        prototypeIndex,
                        PronunciationEmbeddingMatch(
                            score: score, frameRange: candidates.ranges[candidateIndex]
                        ))
                }
            }
        }

    }

    public static func nearbyWindowCounts(around frameCount: Int) -> [Int] {
        let radius = max(2, Int((Double(frameCount) * 0.25).rounded()))
        return Array(Set([max(2, frameCount - radius), max(2, frameCount), max(2, frameCount + radius)])).sorted()
    }

    private static func prefixSums(for sequence: EncoderFeatureSequence) -> [Float] {
        var prefix = [Float](repeating: 0, count: (sequence.frameCount + 1) * sequence.hiddenSize)
        for frameIndex in 0..<sequence.frameCount {
            let sourceOffset = frameIndex * sequence.hiddenSize
            let previousOffset = frameIndex * sequence.hiddenSize
            let destinationOffset = (frameIndex + 1) * sequence.hiddenSize
            for hiddenIndex in 0..<sequence.hiddenSize {
                prefix[destinationOffset + hiddenIndex] =
                    prefix[previousOffset + hiddenIndex] + sequence.values[sourceOffset + hiddenIndex]
            }
        }
        return prefix
    }

    private static func normalizedWindows(
        in sequence: EncoderFeatureSequence,
        prefix: [Float],
        frameCount: Int,
        stride: Int
    ) -> (values: [Float], ranges: [Range<Int>]) {
        let starts = Array(Swift.stride(from: 0, through: sequence.frameCount - frameCount, by: stride))
        var values = [Float](repeating: 0, count: starts.count * sequence.hiddenSize)
        var validRanges: [Range<Int>] = []
        validRanges.reserveCapacity(starts.count)
        var destinationIndex = 0

        for start in starts {
            let end = start + frameCount
            let startOffset = start * sequence.hiddenSize
            let endOffset = end * sequence.hiddenSize
            let destinationOffset = destinationIndex * sequence.hiddenSize
            var squaredNorm: Float = 0
            for hiddenIndex in 0..<sequence.hiddenSize {
                let sum = prefix[endOffset + hiddenIndex] - prefix[startOffset + hiddenIndex]
                values[destinationOffset + hiddenIndex] = sum
                squaredNorm += sum * sum
            }
            guard squaredNorm > 0, squaredNorm.isFinite else { continue }
            var inverseNorm = 1 / sqrt(squaredNorm)
            values.withUnsafeMutableBufferPointer { pointer in
                guard let baseAddress = pointer.baseAddress else { return }
                vDSP_vsmul(
                    baseAddress.advanced(by: destinationOffset),
                    1,
                    &inverseNorm,
                    baseAddress.advanced(by: destinationOffset),
                    1,
                    vDSP_Length(sequence.hiddenSize)
                )
            }
            validRanges.append(start..<end)
            destinationIndex += 1
        }

        if destinationIndex * sequence.hiddenSize < values.count {
            values.removeSubrange((destinationIndex * sequence.hiddenSize)..<values.count)
        }
        return (values, validRanges)
    }

    private static func normalize(_ values: inout [Float]) -> Bool {
        let squaredNorm = values.reduce(Float(0)) { $0 + $1 * $1 }
        guard squaredNorm > 0, squaredNorm.isFinite else { return false }
        let inverseNorm = 1 / sqrt(squaredNorm)
        for index in values.indices {
            values[index] *= inverseNorm
        }
        return true
    }
}

extension Array {
    fileprivate subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
