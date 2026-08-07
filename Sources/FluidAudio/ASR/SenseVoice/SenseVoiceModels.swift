@preconcurrency import CoreML
import Foundation

/// SenseVoice encoder weight precision. fp16 and int8 run on the Neural Engine;
/// fp32 is the non-ANE fallback.
public enum SenseVoiceEncoderPrecision: String, Sendable {
    case fp16
    case int8
    case fp32

    var modelName: String {
        switch self {
        case .fp16: return ModelNames.SenseVoice.encoder
        case .int8: return ModelNames.SenseVoice.encoderInt8
        case .fp32: return ModelNames.SenseVoice.encoderFp32
        }
    }

    var computeUnits: MLComputeUnits {
        self == .fp32 ? .all : .cpuAndNeuralEngine
    }
}

/// Loaded SenseVoiceSmall CoreML models plus vocabulary.
public struct SenseVoiceModels: Sendable {
    public let preprocessor: MLModel
    public let encoder: MLModel
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "SenseVoiceModels")

    public init(preprocessor: MLModel, encoder: MLModel, vocabulary: [Int: String]) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.vocabulary = vocabulary
    }

    public static func downloadAndLoad(
        precision: SenseVoiceEncoderPrecision = .fp16,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SenseVoiceModels {
        let directory = try await download(precision: precision, progressHandler: progressHandler)
        return try load(from: directory, precision: precision)
    }

    public static func download(
        precision: SenseVoiceEncoderPrecision = .fp16,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.senseVoiceSmall.folderName, isDirectory: true)

        if !force && modelsExist(at: targetDir, precision: precision) {
            logger.info("SenseVoice models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }

        logger.info("Downloading SenseVoice models from Hugging Face")
        try await DownloadUtils.downloadRepo(.senseVoiceSmall, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded SenseVoice models")
        return targetDir
    }

    public static func modelsExist(
        at directory: URL,
        precision: SenseVoiceEncoderPrecision = .fp16
    ) -> Bool {
        let required = [
            ModelNames.SenseVoice.preprocessorFile,
            precision.modelName + ".mlmodelc",
            ModelNames.SenseVoice.vocabularyFile,
        ]
        return required.allSatisfy {
            FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
    }

    public static func load(
        from directory: URL,
        precision: SenseVoiceEncoderPrecision = .fp16
    ) throws -> SenseVoiceModels {
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = precision.computeUnits

        let preprocessor = try loadModel(
            named: ModelNames.SenseVoice.preprocessor,
            from: directory,
            configuration: cpuConfig
        )
        let encoder = try loadModel(
            named: precision.modelName,
            from: directory,
            configuration: encoderConfig
        )
        let vocabulary = try loadVocabulary(from: directory)

        logger.info("Loaded SenseVoice (encoder: \(precision.rawValue), vocab: \(vocabulary.count))")
        return SenseVoiceModels(preprocessor: preprocessor, encoder: encoder, vocabulary: vocabulary)
    }

    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")
        let modelURL: URL

        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            modelURL = try MLModel.compileModel(at: packagePath)
        } else {
            throw ASRError.processingFailed("SenseVoice model not found: \(name)")
        }

        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let path = directory.appendingPathComponent(ModelNames.SenseVoice.vocabularyFile)
        let data = try Data(contentsOf: path)

        if let array = try? JSONSerialization.jsonObject(with: data) as? [String] {
            return Dictionary(uniqueKeysWithValues: array.enumerated().map { index, token in (index, token) })
        }

        if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            return Dictionary(uniqueKeysWithValues: dict.compactMap { key, token in
                guard let index = Int(key) else { return nil }
                return (index, token)
            })
        }

        throw ASRError.processingFailed("Failed to parse SenseVoice vocab.json.")
    }

    private static func modelsRootDirectory() -> URL {
        let fileManager = FileManager.default
        if let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            return appSupport
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return fileManager.temporaryDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
