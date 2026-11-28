#if arch(arm64)
import Foundation

struct RumbleWord {
    let tokens: [Int]
    let text: String
    let time: Int
    let key: String
    let startsWithSpace: Bool

    static func isSpace(_ byte: UInt8) -> Bool { byte == 0x20 || byte == 0x0A || byte == 0x0D || byte == 0x09 }

    static func key(of text: String) -> String {
        String(text.precomposedStringWithCanonicalMapping.lowercased().replacingOccurrences(of: "ς", with: "σ").unicodeScalars.filter { CharacterSet.alphanumerics.contains($0) }.map(Character.init))
    }
}

struct RumbleStitcher {
    static let attaching: Set<Character> = [",", ".", "!", "?", ";", ":", ")", "]", "}", "'", "’", "”", "%", "…", "、", "。", "，", "！", "？"]
    static let opening: Set<Character> = ["(", "[", "{", "\"", "“", "‘", "/", "-", "¿", "¡"]

    let spaceless: Bool
    let margin: Int
    private(set) var finalText = ""
    private(set) var finalTokens: [Int] = []
    private(set) var lastTime = Int.min
    private(set) var pending: [RumbleWord] = []
    private(set) var shared: Range<Int>?

    init(spaceless: Bool, margin: Int) {
        self.spaceless = spaceless
        self.margin = margin
    }

    mutating func add(_ piece: [RumbleWord], sharedWithNext next: Range<Int>?) {
        if let shared {
            pending = merged(pending, piece, over: shared)
        } else {
            pending += piece
        }
        shared = next
        let settled = next.map { $0.lowerBound - margin } ?? Int.max
        let cut = pending.firstIndex { $0.time >= settled } ?? pending.count
        finalize(Array(pending[..<cut]))
        pending.removeFirst(cut)
    }

    func text(with preview: [RumbleWord] = []) -> String {
        var text = finalText
        let heard = pending.last?.time ?? lastTime
        append(pending + preview.filter { $0.time > heard }, to: &text)
        while text.last?.isWhitespace == true { text.removeLast() }
        return text
    }

    var tokens: [Int] { finalTokens + pending.flatMap(\.tokens) }

    private func merged(_ left: [RumbleWord], _ right: [RumbleWord], over shared: Range<Int>) -> [RumbleWord] {
        let tailStart = left.firstIndex { $0.time >= shared.lowerBound - margin } ?? left.count
        let tail = Array(left[tailStart...])
        let head = Array(right.prefix { $0.time < shared.upperBound + margin })
        guard !tail.isEmpty, !head.isEmpty else { return left + right.filter { $0.time > (left.last?.time ?? Int.min) } }
        if let run = Self.longestRun(tail, head, within: margin * 3 / 2), run.length >= 2 {
            return Array(left[..<(tailStart + run.left)]) + Array(right[run.right...])
        }
        let middle = (shared.lowerBound + shared.upperBound) / 2
        return left.filter { $0.time < middle } + right.filter { $0.time >= middle }
    }

    static func longestRun(_ left: [RumbleWord], _ right: [RumbleWord], within skew: Int) -> (left: Int, right: Int, length: Int)? {
        let leftKeys = left.indices.filter { !left[$0].key.isEmpty }
        let rightKeys = right.indices.filter { !right[$0].key.isEmpty }
        guard !leftKeys.isEmpty, !rightKeys.isEmpty else { return nil }
        var best: (left: Int, right: Int, length: Int)?
        var previous = [Int](repeating: 0, count: rightKeys.count + 1)
        for i in 1...leftKeys.count {
            var current = [Int](repeating: 0, count: rightKeys.count + 1)
            let a = left[leftKeys[i - 1]]
            for j in 1...rightKeys.count {
                let b = right[rightKeys[j - 1]]
                guard a.key == b.key, abs(a.time - b.time) <= skew else { continue }
                current[j] = previous[j - 1] + 1
                if current[j] > (best?.length ?? 0) { best = (i - current[j], j - current[j], current[j]) }
            }
            previous = current
        }
        guard let best else { return nil }
        let half = best.length / 2
        return (leftKeys[best.left + half], rightKeys[best.right + half], best.length)
    }

    private mutating func finalize(_ words: [RumbleWord]) {
        append(words, to: &finalText)
        finalTokens += words.flatMap(\.tokens)
        if let last = words.last { lastTime = last.time }
    }

    private func append(_ words: [RumbleWord], to text: inout String) {
        for word in words {
            var piece = word.text
            if text.isEmpty {
                piece = String(piece.drop { $0.isWhitespace })
            } else if !spaceless, !word.startsWithSpace, let first = piece.first, !Self.attaching.contains(first),
                      let last = text.last, !last.isWhitespace, !Self.opening.contains(last) {
                piece = " " + piece
            }
            text += piece
        }
    }
}
#endif
