@preconcurrency import CoreML
import Foundation
import OSLog

private let turboCtcLogger = Logger(subsystem: "FluidAudio", category: "GraniteTurboCtcModels")

public enum GraniteTurboCtcError: LocalizedError {
    case manifestMissing(URL)
    case invalidManifest(URL, Error)
    case packageNotFound(String)
    case melFiltersMissing(URL)
    case melFiltersMalformed(expected: Int, actual: Int)
    case invalidAudio(String)
    case invalidOutput(String)

    public var errorDescription: String? {
        switch self {
        case let .manifestMissing(url):
            return "Granite TurboCTC manifest not found at \(url.path)"
        case let .invalidManifest(url, error):
            return "Failed to decode Granite TurboCTC manifest at \(url.path): \(error.localizedDescription)"
        case let .packageNotFound(name):
            return "Granite TurboCTC CoreML package not found: \(name)"
        case let .melFiltersMissing(url):
            return "Granite TurboCTC mel filterbank not found at \(url.path)"
        case let .melFiltersMalformed(expected, actual):
            return "Granite TurboCTC mel filterbank has \(actual) floats, expected \(expected)"
        case let .invalidAudio(message):
            return "Invalid Granite TurboCTC audio: \(message)"
        case let .invalidOutput(message):
            return "Invalid Granite TurboCTC model output: \(message)"
        }
    }
}

/// Fixed-window geometry emitted by `convert-coreml.py`.
public struct GraniteTurboCtcWindow: Codable, Sendable {
    public let seconds: Double
    public let samples: Int
    public let melFrames: Int
    public let stackedFrames: Int
    public let encoderFrames: Int

    private enum CodingKeys: String, CodingKey {
        case seconds
        case samples
        case melFrames = "mel_frames"
        case stackedFrames = "stacked_frames"
        case encoderFrames = "encoder_frames"
    }
}

public struct GraniteTurboCtcManifest: Codable, Sendable {
    public let modelID: String
    public let package: String
    public let melFilters: String
    public let tokenizer: String
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let nMels: Int
    public let stackFactor: Int
    public let deltaWinLength: Int
    public let logmelFloorDB: Float
    public let featureDim: Int
    public let encoderSubsample: Int
    public let blankTokenID: Int
    public let window: GraniteTurboCtcWindow

    private enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case package
        case melFilters = "mel_filters"
        case tokenizer
        case sampleRate = "sample_rate"
        case nFFT = "n_fft"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case nMels = "n_mels"
        case stackFactor = "stack_factor"
        case deltaWinLength = "delta_win_length"
        case logmelFloorDB = "logmel_floor_db"
        case featureDim = "feature_dim"
        case encoderSubsample = "encoder_subsample"
        case blankTokenID = "blank_token_id"
        case window
    }
}

@available(macOS 14, iOS 17, *)
public struct GraniteTurboCtcModels: @unchecked Sendable {
    // @unchecked: MLModel and the HF tokenizer are not formally Sendable, but both
    // are immutable after load and safe to share, matching how AsrModels treats
    // its MLModel stages.
    public static let manifestFile = "manifest.json"

    public let model: MLModel
    public let manifest: GraniteTurboCtcManifest
    public let tokenizer: GraniteTokenizer
    public let melFilterbank: [Float]
    public let modelDirectory: URL

    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> GraniteTurboCtcModels {
        let manifest = try loadManifest(from: directory)
        let tokenizer = try await GraniteTokenizer(
            modelDirectory: directory.appendingPathComponent(manifest.tokenizer, isDirectory: true)
        )
        let melFilterbank = try loadMelFilterbank(from: directory, manifest: manifest)
        let model = try await loadPackage(
            named: manifest.package,
            from: directory,
            computeUnits: computeUnits
        )

        let summary =
            "Loaded Granite TurboCTC from \(directory.path) "
            + "(window \(manifest.window.seconds)s, compute \(computeUnits.rawValue))"
        turboCtcLogger.info("\(summary, privacy: .public)")
        return GraniteTurboCtcModels(
            model: model,
            manifest: manifest,
            tokenizer: tokenizer,
            melFilterbank: melFilterbank,
            modelDirectory: directory
        )
    }

    public static func modelsExist(at directory: URL) -> Bool {
        (try? loadManifest(from: directory)) != nil
    }

    static func loadManifest(from directory: URL) throws -> GraniteTurboCtcManifest {
        let url = directory.appendingPathComponent(manifestFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraniteTurboCtcError.manifestMissing(url)
        }
        do {
            return try JSONDecoder().decode(GraniteTurboCtcManifest.self, from: Data(contentsOf: url))
        } catch {
            throw GraniteTurboCtcError.invalidManifest(url, error)
        }
    }

    /// The bank is stored as `(n_mels, n_freqs)` row-major float32 so it can feed
    /// `vDSP_mmul` directly against the power spectrum.
    static func loadMelFilterbank(
        from directory: URL,
        manifest: GraniteTurboCtcManifest
    ) throws -> [Float] {
        let url = directory.appendingPathComponent(manifest.melFilters)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw GraniteTurboCtcError.melFiltersMissing(url)
        }
        let data = try Data(contentsOf: url)
        let expected = manifest.nMels * (manifest.nFFT / 2 + 1)
        let actual = data.count / MemoryLayout<Float>.stride
        guard actual == expected else {
            throw GraniteTurboCtcError.melFiltersMalformed(expected: expected, actual: actual)
        }
        var filters = [Float](repeating: 0, count: expected)
        _ = filters.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return filters
    }

    private static func loadPackage(
        named packageName: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) async throws -> MLModel {
        let packageURL = directory.appendingPathComponent(packageName)
        let compiledSibling = directory.appendingPathComponent(
            packageURL.deletingPathExtension().lastPathComponent + ".mlmodelc",
            isDirectory: true
        )

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledSibling.path) {
            modelURL = compiledSibling
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            modelURL = try compiledModelURL(for: packageURL, computeUnits: computeUnits)
        } else {
            throw GraniteTurboCtcError.packageNotFound(packageName)
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true
        return try await MLModel.load(contentsOf: modelURL, configuration: configuration)
    }

    private static func compiledModelURL(
        for packageURL: URL,
        computeUnits: MLComputeUnits
    ) throws -> URL {
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? packageURL.deletingLastPathComponent()
        let root = cacheRoot
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("CompiledGraniteTurboCtc", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let attributes = try FileManager.default.attributesOfItem(atPath: packageURL.path)
        let modifiedAt = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let baseName = packageURL.deletingPathExtension().lastPathComponent
        let compiledName = "\(baseName)_\(computeUnits.rawValue)_\(Int64(modifiedAt * 1000)).mlmodelc"
        let compiledURL = root.appendingPathComponent(compiledName, isDirectory: true)

        if FileManager.default.fileExists(atPath: compiledURL.path) {
            return compiledURL
        }

        let temporaryURL = try MLModel.compileModel(at: packageURL)
        try? FileManager.default.removeItem(at: compiledURL)
        try FileManager.default.copyItem(at: temporaryURL, to: compiledURL)
        try? FileManager.default.removeItem(at: temporaryURL)
        return compiledURL
    }
}
