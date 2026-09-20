import Foundation

/// A match in recording-global encoder frames. The index refers to the session's immutable prototype array.
public struct PronunciationWindowMatch: Sendable {
    public let prototypeIndex: Int
    public let score: Float
    public let frameRange: Range<Int>

    public init(prototypeIndex: Int, score: Float, frameRange: Range<Int>) {
        self.prototypeIndex = prototypeIndex
        self.score = score
        self.frameRange = frameRange
    }
}

/// Stable chunks are retained; provisional tail results are replaced, never accumulated.
struct PronunciationWindowMatches: Sendable {
    private(set) var stable: [PronunciationWindowMatch] = []
    private(set) var tail: [PronunciationWindowMatch] = []

    mutating func finalize(_ matches: [PronunciationWindowMatch]) {
        stable.append(contentsOf: matches)
        tail = []
    }

    mutating func replaceTail(_ matches: [PronunciationWindowMatch]) { tail = matches }
    var all: [PronunciationWindowMatch] { stable + tail }
}
