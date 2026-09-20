import Foundation

/// Immutable timestamp index for mapping many acoustic matches to one transcript.
///
/// Build once per transcript. Queries visit only the timestamp interval that can overlap
/// the match, preserving the extractor's majority-overlap rule and original word order.
public struct WordAudioOverlapIndex: Sendable {
    private let words: [TimedTranscriptWord]
    private let sortedIndices: [Int]
    private let maximumEnds: [TimeInterval]
    private let requiresLinearLookup: Bool

    public init(words: [TimedTranscriptWord]) {
        self.words = words
        requiresLinearLookup = words.contains { !$0.startTime.isFinite || !$0.endTime.isFinite }
        let indices =
            requiresLinearLookup
            ? []
            : words.indices.sorted {
                if words[$0].startTime != words[$1].startTime { return words[$0].startTime < words[$1].startTime }
                return $0 < $1
            }
        // Non-finite timestamps use the existing extractor, including its edge-case behavior.
        sortedIndices = indices
        var ends: [TimeInterval] = []
        ends.reserveCapacity(sortedIndices.count)
        var maximum = -TimeInterval.infinity
        for index in sortedIndices {
            maximum = max(maximum, words[index].endTime)
            ends.append(maximum)
        }
        maximumEnds = ends
    }

    /// Returns the same original word indices as `WordAudioChunkExtractor`, without
    /// scanning unrelated words before and after the acoustic match.
    public func substantiallyOverlappingWordIndices(
        startTime: TimeInterval,
        endTime: TimeInterval,
        minimumOverlapRatio: Double = PronunciationCustomizationDefaults.minimumWordOverlapRatio
    ) -> [Int] {
        guard endTime > startTime else { return [] }
        guard !requiresLinearLookup, startTime.isFinite, endTime.isFinite else {
            return WordAudioChunkExtractor.substantiallyOverlappingWordIndices(
                in: words, startTime: startTime, endTime: endTime, minimumOverlapRatio: minimumOverlapRatio)
        }
        var lower = 0
        var upper = sortedIndices.count
        // First prefix containing any word ending after the match begins.
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if maximumEnds[middle] <= startTime { lower = middle + 1 } else { upper = middle }
        }
        let first = lower
        upper = sortedIndices.count
        // First word starting at or after the match ends.
        while lower < upper {
            let middle = lower + (upper - lower) / 2
            if words[sortedIndices[middle]].startTime < endTime { lower = middle + 1 } else { upper = middle }
        }
        let requiredRatio = min(1, max(0, minimumOverlapRatio))
        var matches: [Int] = []
        for position in first..<lower {
            let index = sortedIndices[position]
            let word = words[index]
            let duration = word.endTime - word.startTime
            guard duration > 0 else { continue }
            let overlap = max(0, min(word.endTime, endTime) - max(word.startTime, startTime))
            if overlap / duration > requiredRatio { matches.append(index) }
        }
        return matches.sorted()
    }
}
