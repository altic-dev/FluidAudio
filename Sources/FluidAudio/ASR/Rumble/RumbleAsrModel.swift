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
            let started = ProcessInfo.processInfo.systemUptime
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let model = try MLModel(contentsOf: directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true), configuration: configuration)
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
        let longest = manifest.maxAudioFrames * manifest.hopLength * manifest.windowMelFrames / manifest.windowAudioFrames
        var results: [Transcript] = []
        for range in Self.pieces(of: samples, longest: longest, sampleRate: manifest.sampleRate) {
            let result = try autoreleasepool {
                try transcribePiece(Array(samples[range]), languageTokens: languageTokens)
            }
            results.append(result)
        }
        if results.count == 1 { return results[0] }
        return Transcript(text: results.map(\.text).filter { !$0.isEmpty }.joined(separator: " "), tokens: results.flatMap(\.tokens))
    }

    private func transcribePiece(_ input: [Float], languageTokens: [Int]) throws -> Transcript {
        let started = ProcessInfo.processInfo.systemUptime
        var samples = input
        if samples.count < manifest.minimumSamples { samples += [Float](repeating: 0, count: manifest.minimumSamples - samples.count) }
        let features = mel.compute(samples)
        let (audio, draft) = try encode(features: features.values, frames: features.frames)
        let encoded = ProcessInfo.processInfo.systemUptime

        let prompt = manifest.prefix.count + audio.count + manifest.afterAudio.count + languageTokens.count
        guard prompt + manifest.block < manifest.capacity else { throw Failure.unexpectedLayout("audio length") }
        var session = try DecoderSession(transcriber: self)
        var rows: [[Float16]] = manifest.prefix.map(embedding)
        rows += audio
        rows += (manifest.afterAudio + languageTokens).map(embedding)
        var last = 0
        for start in stride(from: 0, to: rows.count, by: manifest.block) {
            last = try session.run(Array(rows[start..<min(start + manifest.block, rows.count)])).last ?? 0
        }
        let prefilled = ProcessInfo.processInfo.systemUptime

        var output = [last]
        var pointer = Self.follow(draft, from: 0, token: last)
        var accepted = 0
        let limit = min(manifest.maxGeneratedTokens, manifest.capacity - prompt - manifest.block)
        while let token = output.last, !manifest.eos.contains(token), output.count < limit {
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
            "rumble piece samples=\(input.count) encoderMs=\(Int((encoded - started) * 1000)) "
                + "prefillMs=\(Int((prefilled - encoded) * 1000)) decodeMs=\(Int((finished - prefilled) * 1000)) "
                + "draft=\(draft.count) accepted=\(accepted) calls=\(session.calls) tokens=\(written.count) chars=\(text.count)"
        )
        return Transcript(text: text, tokens: Array(written))
    }

    private func encode(features: [Float], frames: Int) throws -> (audio: [[Float16]], draft: [Int]) {
        let window = manifest.windowMelFrames
        let melInput = try Self.zeros([1, manifest.melBins, window], .float16)
        let maskInput = try Self.zeros([1, manifest.windowAudioFrames], .float16)
        let mels = melInput.dataPointer.assumingMemoryBound(to: Float16.self)
        let mask = maskInput.dataPointer.assumingMemoryBound(to: Float16.self)
        var audio: [[Float16]] = []
        var ctc: [Int] = []
        for start in stride(from: 0, to: frames, by: window) {
            let valid = min(window, frames - start)
            for bin in 0..<manifest.melBins {
                for column in 0..<window {
                    mels[bin * window + column] = column < valid ? Float16(features[bin * frames + start + column]) : 0
                }
            }
            let kept = Self.audioFrames(melFrames: valid)
            for index in 0..<manifest.windowAudioFrames { mask[index] = index < kept ? 1 : 0 }
            try autoreleasepool {
                let result = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mel": melInput, "frame_mask": maskInput]))
                let states = try Self.output(result, "audio")
                let ids = try Self.output(result, "ctc_ids")
                guard states.dataType == .float16, ids.dataType == .int32 else { throw Failure.unexpectedLayout("encoder outputs") }
                let statePointer = states.dataPointer.assumingMemoryBound(to: Float16.self)
                let idPointer = ids.dataPointer.assumingMemoryBound(to: Int32.self)
                let rowStride = states.strides[1].intValue
                for row in 0..<kept {
                    audio.append(Array(UnsafeBufferPointer(start: statePointer + row * rowStride, count: manifest.hidden)))
                    ctc.append(Int(idPointer[row * ids.strides[1].intValue]))
                }
            }
        }
        var draft: [Int] = []
        var previous = -1
        for id in ctc {
            if id != previous, id != manifest.ctcBlank { draft.append(Int(ctcVocabulary[id])) }
            previous = id
        }
        return (audio, draft)
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

    static func pieces(of samples: [Float], longest: Int, sampleRate: Int) -> [Range<Int>] {
        let frame = sampleRate / 10
        var ranges: [Range<Int>] = []
        var start = 0
        while samples.count - start > longest {
            var cut = start + longest
            var quietest = Float.infinity
            var position = start + longest * 2 / 3
            while position + frame <= start + longest {
                var energy: Float = 0
                samples.withUnsafeBufferPointer { vDSP_svesq($0.baseAddress! + position, 1, &energy, vDSP_Length(frame)) }
                if energy < quietest {
                    quietest = energy
                    cut = position + frame / 2
                }
                position += frame
            }
            ranges.append(start..<cut)
            start = cut
        }
        ranges.append(start..<samples.count)
        return ranges
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
