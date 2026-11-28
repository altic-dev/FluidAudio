#if arch(arm64)
import Accelerate
import CoreML
import CoreVideo
import Foundation

public final class RumbleAsrModel: @unchecked Sendable {
    public struct Manifest: Decodable, Sendable {
        public let version: String
        public let sampleRate: Int
        public let minimumSamples: Int
        public let hopLength: Int
        public let fftLength: Int
        public let melBins: Int
        public let windowMelFrames: Int
        public let windowAudioFrames: Int
        public let block: Int
        public let capacity: Int
        public let layers: Int
        public let kvHeads: Int
        public let headDim: Int
        public let hidden: Int
        public let ropeTheta: Double
        public let vocabulary: Int
        public let headRows: Int
        public let eos: [Int]
        public let prefix: [Int]
        public let afterAudio: [Int]
        public let languages: [String: [Int]]
        public let ctcBlank: Int
        public let maxAudioFrames: Int
        public let maxGeneratedTokens: Int
        public let spacelessLanguages: [String]
    }

    public enum Failure: LocalizedError {
        case missingOutput(String)
        case unsupportedLanguage(String)
        case unexpectedLayout(String)

        public var errorDescription: String? {
            switch self {
            case let .missingOutput(name): "The speech model did not return \(name)."
            case let .unsupportedLanguage(code): "The speech model does not support \(code)."
            case let .unexpectedLayout(name): "The speech model returned an unexpected \(name)."
            }
        }
    }

    public struct Transcript: Sendable {
        public let text: String
        public let tokens: [Int]
    }

    public static let fileNames = ["rumble.json", "tokens.bin", "ctc_vocab.bin", "mel_filters.bin"]
    public static let modelNames = ["Encoder", "Decoder"]

    public let manifest: Manifest
    private let encoder: MLModel
    private let decoder: MLModel
    private let table: PalettizedEmbedding
    private let pieces: [Data]
    private let ctcVocabulary: [Int32]
    private let mel: RumbleMelSpectrogram
    private let log: @Sendable (String) -> Void

    public static func validationIssue(in directory: URL) -> String? {
        let fileManager = FileManager.default
        for name in fileNames where !fileManager.fileExists(atPath: directory.appendingPathComponent(name).path) {
            return "missing-file:\(name)"
        }
        for name in modelNames {
            for file in ["coremldata.bin", "model.mil", "weights/weight.bin"] {
                let path = directory.appendingPathComponent("\(name).mlmodelc/\(file)").path
                if !fileManager.fileExists(atPath: path) { return "missing-model:\(name)/\(file)" }
            }
        }
        guard (try? loadManifest(in: directory)) != nil else { return "invalid-manifest" }
        return nil
    }

    public init(directory: URL, log: @escaping @Sendable (String) -> Void = { _ in }) throws {
        self.log = log
        let manifest = try Self.loadManifest(in: directory)
        self.manifest = manifest

        func load(_ name: String) throws -> MLModel {
            try Task.checkCancellation()
            let started = ProcessInfo.processInfo.systemUptime
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true), configuration: configuration)
            try Task.checkCancellation()
            log("rumble load step=\(name) ms=\(Int((ProcessInfo.processInfo.systemUptime - started) * 1000))")
            return model
        }

        encoder = try load("Encoder")
        decoder = try load("Decoder")
        table = try PalettizedEmbedding(tiedHeadOf: directory.appendingPathComponent("Decoder.mlmodelc", isDirectory: true), rows: manifest.headRows)
        ctcVocabulary = try Self.read(Int32.self, from: directory.appendingPathComponent("ctc_vocab.bin"))
        pieces = try Self.loadPieces(from: directory.appendingPathComponent("tokens.bin"))
        let filters = try Self.read(Float.self, from: directory.appendingPathComponent("mel_filters.bin"))
        mel = try RumbleMelSpectrogram(manifest: manifest, filters: filters)
        guard table.width == manifest.hidden, manifest.headRows >= manifest.vocabulary,
              pieces.count == manifest.vocabulary, ctcVocabulary.count == manifest.ctcBlank
        else { throw Failure.unexpectedLayout("model files") }
    }

    public func supports(language: String) -> Bool { manifest.languages[language] != nil }

    public func transcribe(samples: [Float], language: String) throws -> Transcript {
        guard let languageTokens = manifest.languages[language] else { throw Failure.unsupportedLanguage(language) }
        guard !samples.isEmpty else { return Transcript(text: "", tokens: []) }
        let spaceless = manifest.spacelessLanguages.contains(language)
        var stitcher = RumbleStitcher(spaceless: spaceless, margin: manifest.sampleRate)
        for span in Self.spans(of: samples, longest: longestPiece, sampleRate: manifest.sampleRate) {
            var windows: [EncodedWindow] = []
            let words = try heardWords(
                Array(samples[span.piece]), at: span.piece.lowerBound, languageTokens: languageTokens,
                spaceless: spaceless, cache: &windows
            )
            stitcher.add(words, sharedWithNext: span.shared)
        }
        return Transcript(text: stitcher.text(), tokens: stitcher.tokens)
    }

    public func makeLiveTranscription(language: String) throws -> LiveTranscription {
        guard let languageTokens = manifest.languages[language] else { throw Failure.unsupportedLanguage(language) }
        return LiveTranscription(
            model: self, languageTokens: languageTokens, spaceless: manifest.spacelessLanguages.contains(language)
        )
    }

    public struct LiveTranscription {
        private let model: RumbleAsrModel
        private let languageTokens: [Int]
        private let spaceless: Bool
        private var stitcher: RumbleStitcher
        private var windows: [EncodedWindow] = []
        public private(set) var committedSamples = 0

        fileprivate init(model: RumbleAsrModel, languageTokens: [Int], spaceless: Bool) {
            self.model = model
            self.languageTokens = languageTokens
            self.spaceless = spaceless
            stitcher = RumbleStitcher(spaceless: spaceless, margin: model.manifest.sampleRate)
        }

        public mutating func update(samples: [Float], startingAt offset: Int = 0) throws -> String {
            try commitCompletedPieces(samples, startingAt: offset)
            let start = committedSamples - offset
            let manifest = model.manifest
            let lookahead = max(0, manifest.fftLength / 2 - manifest.hopLength)
            let firstCut = model.longestPiece * 2 / 3 + manifest.sampleRate / 20
            let available = min(samples.count - start, firstCut)
            let count = max(0, (available - lookahead) / (manifest.windowMelFrames * manifest.hopLength))
            if count > windows.count {
                let features = model.mel.compute(Array(samples[start...]))
                try model.encode(features: features, windowCount: count, cache: &windows)
            }
            let runs = model.draftRuns(from: windows)
            let draft = Timed(tokens: runs.map(\.token), times: runs.map(\.sample), runs: runs, covered: runs.map { _ in true })
            let preview = model.words(of: draft, at: committedSamples, spaceless: spaceless)
            return stitcher.text(with: preview)
        }

        public mutating func commit(samples: [Float], startingAt offset: Int = 0) throws {
            try commitCompletedPieces(samples, startingAt: offset)
        }

        public mutating func finish(samples: [Float], startingAt offset: Int = 0) throws -> Transcript {
            try commitCompletedPieces(samples, startingAt: offset)
            let start = committedSamples - offset
            let words = samples.count > start
                ? try model.heardWords(
                    Array(samples[start...]), at: committedSamples, languageTokens: languageTokens,
                    spaceless: spaceless, cache: &windows
                )
                : []
            stitcher.add(words, sharedWithNext: nil)
            committedSamples = offset + samples.count
            windows.removeAll(keepingCapacity: true)
            return Transcript(text: stitcher.text(), tokens: stitcher.tokens)
        }

        private mutating func commitCompletedPieces(_ samples: [Float], startingAt offset: Int) throws {
            try Task.checkCancellation()
            guard offset >= 0, offset <= committedSamples, committedSamples - offset <= samples.count else {
                throw Failure.unexpectedLayout("live audio range")
            }
            var start = committedSamples - offset
            while samples.count - start > model.longestPiece {
                let span = RumbleAsrModel.nextSpan(
                    in: samples, start: start, longest: model.longestPiece, sampleRate: model.manifest.sampleRate
                )
                let words = try model.heardWords(
                    Array(samples[span.piece]), at: offset + span.piece.lowerBound, languageTokens: languageTokens,
                    spaceless: spaceless, cache: &windows
                )
                stitcher.add(words, sharedWithNext: span.shared.map { (offset + $0.lowerBound)..<(offset + $0.upperBound) })
                committedSamples = offset + span.next
                start = span.next
                windows.removeAll(keepingCapacity: true)
                try Task.checkCancellation()
            }
        }
    }

    private struct EncodedWindow {
        let melFrames: Int
        let input: [Float16]
        let audio: [[Float16]]
        let ctc: [Int]
    }

    private var longestPiece: Int {
        manifest.maxAudioFrames * manifest.hopLength * manifest.windowMelFrames / manifest.windowAudioFrames
    }

    struct Timed {
        let tokens: [Int]
        let times: [Int]
        let runs: [(token: Int, sample: Int)]
        let covered: [Bool]
    }

    struct Span {
        let piece: Range<Int>
        let next: Int

        var shared: Range<Int>? { next < piece.upperBound ? next..<piece.upperBound : nil }
    }

    private func heardWords(
        _ input: [Float], at offset: Int, languageTokens: [Int], spaceless: Bool, cache: inout [EncodedWindow]
    ) throws -> [RumbleWord] {
        let timed = try transcribePiece(input, languageTokens: languageTokens, cache: &cache)
        var words = self.words(of: timed, at: offset, spaceless: spaceless)
        let pad = manifest.sampleRate * 2 / 5
        let gaps = Self.uncovered(timed, sampleRate: manifest.sampleRate).filter { gap in
            let slice = input[max(0, gap.samples.lowerBound)..<min(input.count, gap.samples.upperBound)]
            let power = slice.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(max(1, slice.count))
            return power.squareRoot() >= 0.005
        }.sorted { $0.samples.count > $1.samples.count }
        for (gap, reachesEnd) in gaps.prefix(2) {
            let from = max(0, gap.lowerBound - pad)
            let to = min(input.count, gap.upperBound + pad)
            guard to - from >= manifest.sampleRate else { continue }
            var fresh: [EncodedWindow] = []
            let retry = try transcribePiece(Array(input[from..<to]), languageTokens: languageTokens, cache: &fresh)
            let inside = (offset + gap.lowerBound)..<(reachesEnd ? Int.max : offset + gap.upperBound)
            var found = self.words(of: retry, at: offset + from, spaceless: spaceless).filter { inside.contains($0.time) }
            let before = words.last { $0.time < inside.lowerBound }
            let after = words.first { $0.time >= inside.upperBound }
            if let first = found.first, first.key == before?.key, !first.key.isEmpty { found.removeFirst() }
            if let last = found.last, last.key == after?.key, !last.key.isEmpty { found.removeLast() }
            guard !found.isEmpty else { continue }
            log("rumble recovered words=\(found.count) from=\(gap.lowerBound) to=\(gap.upperBound)")
            words = (words.filter { !inside.contains($0.time) } + found).enumerated()
                .sorted { ($0.element.time, $0.offset) < ($1.element.time, $1.offset) }.map(\.element)
        }
        return words
    }

    static func uncovered(_ timed: Timed, sampleRate: Int) -> [(samples: Range<Int>, reachesEnd: Bool)] {
        var gaps: [(samples: Range<Int>, reachesEnd: Bool)] = []
        var index = 0
        let frame = sampleRate * 2 / 26
        while index < timed.runs.count {
            guard !timed.covered[index] else {
                index += 1
                continue
            }
            var end = index
            while end < timed.runs.count, !timed.covered[end] { end += 1 }
            let stretch = timed.runs[index..<end]
            let first = stretch.first!.sample
            let last = stretch.last!.sample
            let written = timed.times.filter { $0 >= first && $0 < last + frame }.count
            if stretch.count >= 6, Set(stretch.map(\.token)).count >= 3, last - first >= sampleRate * 6 / 5,
               written * 10 < stretch.count {
                gaps.append((first..<(last + frame), end == timed.runs.count))
            }
            index = end
        }
        return gaps
    }

    func words(of timed: Timed, at offset: Int, spaceless: Bool) -> [RumbleWord] {
        guard !timed.runs.isEmpty else { return [] }
        return spaceless ? characterWords(timed, at: offset) : spacedWords(timed, at: offset)
    }

    private func spacedWords(_ timed: Timed, at offset: Int) -> [RumbleWord] {
        var words: [RumbleWord] = []
        var tokens: [Int] = []
        var bytes = Data()
        var time = 0
        func close() {
            if !tokens.isEmpty {
                let text = String(decoding: bytes, as: UTF8.self)
                words.append(RumbleWord(
                    tokens: tokens, text: text, time: offset + time, key: RumbleWord.key(of: text),
                    startsWithSpace: bytes.first.map(RumbleWord.isSpace) ?? false
                ))
            }
            tokens = []
            bytes = Data()
        }
        for (index, token) in timed.tokens.enumerated() {
            let piece = pieces.indices.contains(token) ? pieces[token] : Data()
            if tokens.isEmpty || piece.first.map(RumbleWord.isSpace) == true {
                close()
                time = timed.times[index]
            }
            tokens.append(token)
            bytes.append(piece)
        }
        close()
        return words
    }

    private func characterWords(_ timed: Timed, at offset: Int) -> [RumbleWord] {
        var bytes = Data()
        var owners: [Int] = []
        for (index, token) in timed.tokens.enumerated() {
            let piece = pieces.indices.contains(token) ? pieces[token] : Data()
            bytes.append(piece)
            owners += [Int](repeating: index, count: piece.count)
        }
        var words: [RumbleWord] = []
        var position = 0
        var claimed = -1
        for character in String(decoding: bytes, as: UTF8.self) {
            let width = character.utf8.count
            let owner = owners.indices.contains(position) ? owners[position] : (owners.last ?? 0)
            let last = min(owners.count, position + width) - 1
            let through = last >= 0 && owners.indices.contains(last) ? owners[last] : owner
            let tokens = claimed < through ? Array(timed.tokens[(claimed + 1)...through]) : []
            claimed = max(claimed, through)
            position += width
            let text = String(character)
            let time = timed.times.indices.contains(owner) ? timed.times[owner] : 0
            if character.isASCII, character.isLetter || character.isNumber, let previous = words.last,
               previous.text.last.map({ $0.isASCII && ($0.isLetter || $0.isNumber) }) == true {
                words[words.count - 1] = RumbleWord(
                    tokens: previous.tokens + tokens, text: previous.text + text, time: previous.time,
                    key: RumbleWord.key(of: previous.text + text), startsWithSpace: false
                )
            } else {
                words.append(RumbleWord(
                    tokens: tokens, text: text, time: offset + time, key: RumbleWord.key(of: text),
                    startsWithSpace: character.isWhitespace
                ))
            }
        }
        return words
    }

    private func transcribePiece(
        _ input: [Float], languageTokens: [Int], cache: inout [EncodedWindow]
    ) throws -> Timed {
        try autoreleasepool {
            let started = ProcessInfo.processInfo.systemUptime
            var samples = input
            if samples.count < manifest.minimumSamples {
                samples += [Float](repeating: 0, count: manifest.minimumSamples - samples.count)
            }
            let features = mel.compute(samples)
            let count = (features.frames + manifest.windowMelFrames - 1) / manifest.windowMelFrames
            try encode(features: features, windowCount: count, cache: &cache)
            return try decodePiece(cache, samples: samples.count, languageTokens: languageTokens, started: started)
        }
    }

    private func decodePiece(
        _ windows: [EncodedWindow], samples: Int, languageTokens: [Int], started: TimeInterval
    ) throws -> Timed {
        let audio = windows.flatMap(\.audio)
        let draft = draft(from: windows)
        let encoded = ProcessInfo.processInfo.systemUptime

        let prompt = manifest.prefix.count + audio.count + manifest.afterAudio.count + languageTokens.count
        guard prompt + manifest.block < manifest.capacity else { throw Failure.unexpectedLayout("audio length") }
        var session = try DecoderSession(transcriber: self)
        var rows: [[Float16]] = manifest.prefix.map(embedding)
        rows += audio
        rows += (manifest.afterAudio + languageTokens).map(embedding)
        var last = 0
        for start in stride(from: 0, to: rows.count, by: manifest.block) {
            try Task.checkCancellation()
            last = try session.run(Array(rows[start..<min(start + manifest.block, rows.count)])).last ?? 0
        }
        let prefilled = ProcessInfo.processInfo.systemUptime

        var output = [last]
        var pointer = Self.follow(draft, from: 0, token: last)
        var accepted = 0
        let limit = min(manifest.maxGeneratedTokens, manifest.capacity - prompt - manifest.block)
        while let token = output.last, !manifest.eos.contains(token), output.count < limit {
            try Task.checkCancellation()
            let proposal = [token] + Array(draft[min(pointer, draft.count)..<min(pointer + manifest.block - 1, draft.count)])
            let predicted = try session.run(proposal.map(embedding))
            var agreed = 0
            while agreed + 1 < proposal.count, predicted[agreed] == proposal[agreed + 1] { agreed += 1 }
            output += proposal[1..<(1 + agreed)]
            pointer += agreed
            accepted += agreed
            session.rewind(proposal.count - 1 - agreed)
            output.append(predicted[agreed])
            pointer = Self.follow(draft, from: pointer, token: predicted[agreed])
        }
        let written = output.prefix { !manifest.eos.contains($0) }
        let text = decode(Array(written))
        let finished = ProcessInfo.processInfo.systemUptime
        log(
            "rumble piece samples=\(samples) encoderMs=\(Int((encoded - started) * 1000)) "
                + "prefillMs=\(Int((prefilled - encoded) * 1000)) decodeMs=\(Int((finished - prefilled) * 1000)) "
                + "draft=\(draft.count) accepted=\(accepted) calls=\(session.calls) tokens=\(written.count) chars=\(text.count)"
        )
        let runs = draftRuns(from: windows)
        let aligned = Self.align(Array(written), to: runs, length: samples, step: manifest.sampleRate * 3 / 20)
        return Timed(tokens: Array(written), times: aligned.times, runs: runs, covered: aligned.covered)
    }

    private func draftRuns(from windows: [EncodedWindow]) -> [(token: Int, sample: Int)] {
        let windowSamples = manifest.windowMelFrames * manifest.hopLength
        var runs: [(token: Int, sample: Int)] = []
        var previous = -1
        for (index, window) in windows.enumerated() {
            for (frame, id) in window.ctc.enumerated() {
                if id != previous, id != manifest.ctcBlank, ctcVocabulary.indices.contains(id) {
                    runs.append((Int(ctcVocabulary[id]), index * windowSamples + frame * windowSamples / manifest.windowAudioFrames))
                }
                previous = id
            }
        }
        return runs
    }

    static func align(
        _ tokens: [Int], to runs: [(token: Int, sample: Int)], length: Int, step: Int
    ) -> (times: [Int], covered: [Bool]) {
        let n = tokens.count
        let m = runs.count
        let band = max(16, abs(n - m) + 8)
        func matches(_ i: Int, _ j: Int) -> Bool {
            tokens[i] == runs[j].token && abs(i - j) <= band
        }
        var table = [[Int32]](repeating: [Int32](repeating: 0, count: m + 1), count: n + 1)
        if n > 0, m > 0 {
            for i in stride(from: n - 1, through: 0, by: -1) {
                for j in stride(from: m - 1, through: 0, by: -1) {
                    table[i][j] = matches(i, j) ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
                }
            }
        }
        var matched = [Int?](repeating: nil, count: n)
        var covered = [Bool](repeating: false, count: m)
        var i = 0
        var j = 0
        while i < n, j < m {
            if matches(i, j), table[i][j] == table[i + 1][j + 1] + 1 {
                matched[i] = runs[j].sample
                covered[j] = true
                i += 1
                j += 1
            } else if table[i + 1][j] >= table[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        var times = [Int](repeating: 0, count: n)
        var index = 0
        while index < n {
            if let time = matched[index] {
                times[index] = time
                index += 1
                continue
            }
            let before = index - 1
            var after = index
            while after < n, matched[after] == nil { after += 1 }
            for gap in index..<after {
                if before >= 0, after < n {
                    times[gap] = times[before] + (matched[after]! - times[before]) * (gap - before) / (after - before)
                } else if before >= 0 {
                    times[gap] = min(length, times[before] + (gap - before) * step)
                } else if after < n {
                    times[gap] = max(0, matched[after]! - (after - gap) * step)
                } else {
                    times[gap] = min(length, (runs.first?.sample ?? 0) + gap * step)
                }
            }
            index = after
        }
        for k in times.indices.dropFirst() where times[k] < times[k - 1] { times[k] = times[k - 1] }
        return (times, covered)
    }

    private func encode(
        features: (values: [Float], frames: Int), windowCount: Int, cache: inout [EncodedWindow]
    ) throws {
        let window = manifest.windowMelFrames
        if cache.count > windowCount { cache.removeLast(cache.count - windowCount) }
        let melInput = try Self.zeros([1, manifest.melBins, window], .float16)
        let maskInput = try Self.zeros([1, manifest.windowAudioFrames], .float16)
        let mels = melInput.dataPointer.assumingMemoryBound(to: Float16.self)
        let mask = maskInput.dataPointer.assumingMemoryBound(to: Float16.self)
        for index in 0..<windowCount {
            try Task.checkCancellation()
            let start = index * window
            let valid = min(window, features.frames - start)
            var input = [Float16](repeating: 0, count: manifest.melBins * window)
            for bin in 0..<manifest.melBins {
                for column in 0..<valid {
                    input[bin * window + column] = Float16(features.values[bin * features.frames + start + column])
                }
            }
            if index < cache.count, cache[index].melFrames == valid, cache[index].input == input { continue }
            input.withUnsafeBufferPointer { buffer in
                if let base = buffer.baseAddress { mels.update(from: base, count: buffer.count) }
            }
            let kept = Self.audioFrames(melFrames: valid)
            for row in 0..<manifest.windowAudioFrames { mask[row] = row < kept ? 1 : 0 }
            let encoded = try autoreleasepool {
                let result = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": melInput, "frame_mask": maskInput]))
                let states = try Self.output(result, "audio")
                let ids = try Self.output(result, "ctc_ids")
                guard states.dataType == .float16, ids.dataType == .int32 else { throw Failure.unexpectedLayout("encoder outputs") }
                let statePointer = states.dataPointer.assumingMemoryBound(to: Float16.self)
                let idPointer = ids.dataPointer.assumingMemoryBound(to: Int32.self)
                let rowStride = states.strides[1].intValue
                let idStride = ids.strides[1].intValue
                let audio = (0..<kept).map { row in
                    Array(UnsafeBufferPointer(start: statePointer + row * rowStride, count: manifest.hidden))
                }
                let ctc = (0..<kept).map { Int(idPointer[$0 * idStride]) }
                return EncodedWindow(melFrames: valid, input: input, audio: audio, ctc: ctc)
            }
            if index < cache.count { cache[index] = encoded } else { cache.append(encoded) }
            log("rumble encoder window=\(index) melFrames=\(valid)")
        }
    }

    private func draft(from windows: [EncodedWindow]) -> [Int] {
        var draft: [Int] = []
        var previous = -1
        for window in windows {
            for id in window.ctc {
                if id != previous, id != manifest.ctcBlank { draft.append(Int(ctcVocabulary[id])) }
                previous = id
            }
        }
        return draft
    }

    static func audioFrames(melFrames: Int) -> Int {
        let whole = melFrames / 100 * 13
        let rest = melFrames % 100
        guard rest > 0 else { return whole }
        let first = (rest - 1) / 2 + 1
        let second = (first - 1) / 2 + 1
        return whole + (second - 1) / 2 + 1
    }

    static func follow(_ draft: [Int], from pointer: Int, token: Int, reach: Int = 4) -> Int {
        for step in 0...reach where pointer + step < draft.count && draft[pointer + step] == token {
            return pointer + step + 1
        }
        return pointer
    }

    private func embedding(_ token: Int) -> [Float16] {
        var row = [Float16](repeating: 0, count: manifest.hidden)
        row.withUnsafeMutableBufferPointer { table.write(Int32(token), to: $0.baseAddress!) }
        return row
    }

    private func decode(_ tokens: [Int]) -> String {
        var bytes = Data()
        for token in tokens where pieces.indices.contains(token) { bytes.append(pieces[token]) }
        return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func spans(of samples: [Float], longest: Int, sampleRate: Int) -> [Span] {
        var spans: [Span] = []
        var start = 0
        while samples.count - start > longest {
            let span = nextSpan(in: samples, start: start, longest: longest, sampleRate: sampleRate)
            spans.append(span)
            start = span.next
        }
        spans.append(Span(piece: start..<samples.count, next: samples.count))
        return spans
    }

    static func nextSpan(in samples: [Float], start: Int, longest: Int, sampleRate: Int) -> Span {
        let frame = sampleRate / 10
        let cut = quietest(in: samples, from: start + longest * 2 / 3, to: start + longest, frame: frame, latest: true)
        let from = max(start + longest / 2, cut - 3 * sampleRate)
        let to = cut - sampleRate
        let next = to - from >= frame ? quietest(in: samples, from: from, to: to, frame: frame) : cut
        return Span(piece: start..<cut, next: next)
    }

    private static func quietest(in samples: [Float], from: Int, to: Int, frame: Int, latest: Bool = false) -> Int {
        var cut = to
        var quietest = Double.infinity
        var position = from
        let step = max(1, frame / 4)
        while position + frame <= to {
            var energy = 0.0
            for index in position..<(position + frame) {
                let sample = Double(samples[index])
                energy += sample * sample
            }
            if energy < quietest || latest && energy == quietest {
                quietest = energy
                cut = position + frame / 2
            }
            position += step
        }
        return cut
    }

    private struct DecoderSession {
        let transcriber: RumbleAsrModel
        let keys: MLMultiArray
        let values: MLMultiArray
        let embeds: MLMultiArray
        let cosines: MLMultiArray
        let sines: MLMultiArray
        let mask: MLMultiArray
        let inputs: MLDictionaryFeatureProvider
        var length = 0
        var calls = 0

        init(transcriber: RumbleAsrModel) throws {
            self.transcriber = transcriber
            let m = transcriber.manifest
            let cacheShape = [m.layers, m.kvHeads, m.capacity, m.headDim]
            keys = try RumbleAsrModel.sharedZeros(cacheShape)
            values = try RumbleAsrModel.sharedZeros(cacheShape)
            embeds = try RumbleAsrModel.zeros([1, m.block, m.hidden], .float16)
            cosines = try RumbleAsrModel.zeros([1, 1, m.block, m.headDim], .float16)
            sines = try RumbleAsrModel.zeros([1, 1, m.block, m.headDim], .float16)
            mask = try RumbleAsrModel.zeros([1, 1, m.block, m.capacity + m.block], .float16)
            inputs = try MLDictionaryFeatureProvider(dictionary: [
                "embeds": embeds, "cos": cosines, "sin": sines, "mask": mask, "k_cache": keys, "v_cache": values,
            ])
        }

        mutating func run(_ rows: [[Float16]]) throws -> [Int] {
            let m = transcriber.manifest
            guard !rows.isEmpty, rows.count <= m.block, length + rows.count <= m.capacity else {
                throw Failure.unexpectedLayout("decoder block")
            }
            let embedPointer = embeds.dataPointer.assumingMemoryBound(to: Float16.self)
            for row in 0..<m.block {
                for column in 0..<m.hidden {
                    embedPointer[row * m.hidden + column] = row < rows.count ? rows[row][column] : 0
                }
            }
            let cosPointer = cosines.dataPointer.assumingMemoryBound(to: Float16.self)
            let sinPointer = sines.dataPointer.assumingMemoryBound(to: Float16.self)
            let half = m.headDim / 2
            for row in 0..<m.block {
                let position = Double(length + row)
                for index in 0..<half {
                    let angle = position / pow(m.ropeTheta, Double(2 * index) / Double(m.headDim))
                    let c = Float16(Float(cos(angle)))
                    let s = Float16(Float(sin(angle)))
                    cosPointer[row * m.headDim + index] = c
                    cosPointer[row * m.headDim + index + half] = c
                    sinPointer[row * m.headDim + index] = s
                    sinPointer[row * m.headDim + index + half] = s
                }
            }
            let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
            let width = m.capacity + m.block
            for row in 0..<m.block {
                for column in 0..<width {
                    let open = column < length || (column >= m.capacity && column - m.capacity <= row)
                    maskPointer[row * width + column] = open ? 0 : -10_000
                }
            }
            let result = try transcriber.decoder.prediction(from: inputs)
            let ids = try RumbleAsrModel.output(result, "ids")
            try RumbleAsrModel.write(RumbleAsrModel.output(result, "new_k"), into: keys, at: length, rows: rows.count)
            try RumbleAsrModel.write(RumbleAsrModel.output(result, "new_v"), into: values, at: length, rows: rows.count)
            length += rows.count
            calls += 1
            guard ids.dataType == .int32 else { throw Failure.unexpectedLayout("decoder ids") }
            let idPointer = ids.dataPointer.assumingMemoryBound(to: Int32.self)
            let step = ids.strides.last?.intValue ?? 1
            return (0..<rows.count).map { Int(idPointer[$0 * step]) }
        }

        mutating func rewind(_ rows: Int) {
            length -= rows
        }
    }

    private static func loadManifest(in directory: URL) throws -> Manifest {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("rumble.json")))
        guard manifest.sampleRate == 16_000, manifest.block > 0, manifest.capacity > manifest.block, !manifest.eos.isEmpty,
              !manifest.languages.isEmpty, manifest.hidden > 0, manifest.windowMelFrames > 0, manifest.windowAudioFrames > 0
        else { throw Failure.unexpectedLayout("manifest") }
        return manifest
    }

    private static func loadPieces(from url: URL) throws -> [Data] {
        let data = try Data(contentsOf: url)
        var pieces: [Data] = []
        var offset = 0
        while offset + 2 <= data.count {
            let length = Int(data[offset]) | Int(data[offset + 1]) << 8
            offset += 2
            guard offset + length <= data.count else { throw Failure.unexpectedLayout("tokens") }
            pieces.append(data.subdata(in: offset..<(offset + length)))
            offset += length
        }
        return pieces
    }

    private static func read<T>(_ type: T.Type, from url: URL) throws -> [T] {
        let data = try Data(contentsOf: url)
        guard data.count % MemoryLayout<T>.stride == 0 else { throw Failure.unexpectedLayout(url.lastPathComponent) }
        return data.withUnsafeBytes { Array($0.bindMemory(to: T.self)) }
    }

    fileprivate static func zeros(_ shape: [Int], _ type: MLMultiArrayDataType) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
        let width = type == .float16 ? 2 : 4
        array.dataPointer.initializeMemory(as: UInt8.self, repeating: 0, count: array.count * width)
        return array
    }

    fileprivate static func sharedZeros(_ shape: [Int]) throws -> MLMultiArray {
        guard let width = shape.last else { throw Failure.unexpectedLayout("cache shape") }
        let height = shape.dropLast().reduce(1, *)
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent16Half, attributes, &buffer)
        guard let buffer else { return try zeros(shape, .float16) }
        CVPixelBufferLockBaseAddress(buffer, [])
        memset(CVPixelBufferGetBaseAddress(buffer), 0, CVPixelBufferGetBytesPerRow(buffer) * height)
        CVPixelBufferUnlockBaseAddress(buffer, [])
        return MLMultiArray(pixelBuffer: buffer, shape: shape.map(NSNumber.init))
    }

    fileprivate static func output(_ result: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
        guard let array = result.featureValue(for: name)?.multiArrayValue else { throw Failure.missingOutput(name) }
        return array
    }

    fileprivate static func write(_ source: MLMultiArray, into cache: MLMultiArray, at position: Int, rows: Int) throws {
        let shape = cache.shape.map(\.intValue)
        let sourceShape = source.shape.map(\.intValue)
        let strides = cache.strides.map(\.intValue)
        let sourceStrides = source.strides.map(\.intValue)
        guard shape.count == 4, sourceShape.count == 4, source.dataType == .float16, cache.dataType == .float16,
              sourceShape[0] == shape[0], sourceShape[1] == shape[1], sourceShape[3] == shape[3],
              rows <= sourceShape[2], position >= 0, position + rows <= shape[2], strides[3] == 1, sourceStrides[3] == 1
        else { throw Failure.unexpectedLayout("cache slice") }
        let width = MemoryLayout<Float16>.stride
        source.withUnsafeBytes { input in
            cache.withUnsafeMutableBytes { output, _ in
                guard let from = input.baseAddress, let to = output.baseAddress else { return }
                for layer in 0..<shape[0] {
                    for head in 0..<shape[1] {
                        for row in 0..<rows {
                            let read = layer * sourceStrides[0] + head * sourceStrides[1] + row * sourceStrides[2]
                            let store = layer * strides[0] + head * strides[1] + (position + row) * strides[2]
                            to.advanced(by: store * width).copyMemory(from: from.advanced(by: read * width), byteCount: shape[3] * width)
                        }
                    }
                }
            }
        }
    }
}
#endif
