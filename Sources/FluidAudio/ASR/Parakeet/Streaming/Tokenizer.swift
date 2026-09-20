import Foundation

public class Tokenizer {
    private var vocab: [String: String] = [:]
    private var idToToken: [Int: String] = [:]

    public convenience init(modelDirectory: URL) throws {
        let jsonURL = modelDirectory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            try self.init(vocabPath: jsonURL)
            return
        }

        let modelURL = modelDirectory.appendingPathComponent("tokenizer.model")
        try self.init(sentencePiecePath: modelURL)
    }

    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: String] else {
            throw ASRError.processingFailed("Invalid tokenizer vocabulary at \(vocabPath.lastPathComponent)")
        }

        self.vocab = json
        for (key, value) in json {
            if let id = Int(key) {
                self.idToToken[id] = value
            }
        }
    }

    public init(sentencePiecePath: URL) throws {
        let data = try Data(contentsOf: sentencePiecePath)
        let pieces = try SentencePieceProto.parse(data)

        for (id, piece) in pieces.enumerated() {
            self.idToToken[id] = piece.piece
            self.vocab[String(id)] = piece.piece
        }
    }

    /// Return a raw vocabulary piece, preserving SentencePiece word boundaries.
    public func piece(forId id: Int) -> String? {
        idToToken[id]
    }

    public func decode(ids: [Int], skipSpecialTokens: Bool = false) -> String {
        var text = ""
        for id in ids {
            if let token = idToToken[id] {
                if skipSpecialTokens && token.hasPrefix("<") && token.hasSuffix(">") {
                    continue
                }
                text += token
            }
        }
        // Replace SentencePiece word boundary marker with space, then trim
        return text.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
