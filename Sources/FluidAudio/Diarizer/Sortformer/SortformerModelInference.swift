@preconcurrency import CoreML
import Foundation
import OSLog

// MARK: - Models Container

/// Container for Sortformer CoreML models.
///
/// Sortformer uses three models:
/// - Preprocessor: Audio → Mel features
/// - PreEncoder: Mel features + State → Concatenated embeddings
/// - Head: Concatenated embeddings → Predictions + Chunk embeddings
public struct SortformerModels {
    /// Main Sortformer model for diarization (combined pipeline, deprecated)
    public let mainModel: MLModel

    /// Time taken to compile/load models
    public let compilationDuration: TimeInterval

    /// Tensor geometry and feature names read from `mainModel`'s own description.
    public let geometry: SortformerModelGeometry

    /// Cached buffers
    private let memoryOptimizer: ANEMemoryOptimizer
    private let chunkArray: MLMultiArray
    private let chunkLengthArray: MLMultiArray
    private let fifoArray: MLMultiArray
    private let fifoLengthArray: MLMultiArray
    private let spkcacheArray: MLMultiArray
    private let spkcacheLengthArray: MLMultiArray

    /// - Throws: `SortformerError.modelLoadFailed` if the model's inputs/outputs are unrecognized, or
    ///   `SortformerError.configurationError` if `config` does not match the model's declared shapes.
    public init(
        config: SortformerConfig,
        main: MLModel,
        compilationDuration: TimeInterval = 0
    ) throws {
        self.mainModel = main
        self.compilationDuration = compilationDuration

        try config.validateGeometry()
        let geometry = try SortformerModelGeometry.inspect(model: main)
        try geometry.validate(against: config)
        self.geometry = geometry

        self.memoryOptimizer = .init()
        self.chunkArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.chunkMelFrames), NSNumber(value: config.melFeatures)], dataType: .float32)
        self.fifoArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.fifoLen), NSNumber(value: config.preEncoderDims)], dataType: .float32)
        self.spkcacheArray = try memoryOptimizer.createAlignedArray(
            shape: [1, NSNumber(value: config.spkcacheLen), NSNumber(value: config.preEncoderDims)], dataType: .float32)
        self.chunkLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.fifoLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
        self.spkcacheLengthArray = try memoryOptimizer.createAlignedArray(shape: [1], dataType: .int32)
    }
}

// MARK: - Model Loading

extension SortformerModels {

    private static let logger = AppLogger(category: "SortformerModels")

    /// Load models from local file paths (combined pipeline mode).
    ///
    /// - Parameters:
    ///   - config: Sortformer configuration; validated against the model's declared shapes
    ///   - mainModelPath: Path to `Sortformer.mlpackage` (compiled on load) or to an already
    ///     compiled `Sortformer.mlmodelc` directory (loaded as-is)
    ///   - configuration: Optional MLModel configuration; used verbatim when supplied
    /// - Returns: Loaded SortformerModels
    /// - Throws: `CancellationError` if the enclosing task is cancelled before the model is returned.
    public static func load(
        config: SortformerConfig,
        mainModelPath: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from local paths (combined pipeline mode)")

        let startTime = Date()

        try Task.checkCancellation()

        // An `.mlmodelc` directory is already compiled; anything else (`.mlpackage`, `.mlmodel`)
        // goes through the compiler.
        let compiledMainModelURL: URL
        if mainModelPath.pathExtension.lowercased() == "mlmodelc" {
            logger.info("Using pre-compiled main model")
            compiledMainModelURL = mainModelPath
        } else {
            logger.info("Compiling main model...")
            compiledMainModelURL = try await MLModel.compileModel(at: mainModelPath)
        }

        try Task.checkCancellation()

        // Honor a caller-supplied configuration; otherwise keep the historical default where
        // .all lets CoreML pick optimal compute units.
        let mainConfig: MLModelConfiguration
        if let configuration {
            mainConfig = configuration
        } else {
            mainConfig = MLModelConfiguration()
            mainConfig.computeUnits = .all
        }
        let mainModel = try await MLModel.load(contentsOf: compiledMainModelURL, configuration: mainConfig)
        logger.info("Loaded main Sortformer model")

        // Loading can take tens of seconds; don't hand a model back to a caller that gave up.
        try Task.checkCancellation()

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Models loaded in \(String(format: "%.2f", duration))s")

        return try SortformerModels(
            config: config,
            main: mainModel,
            compilationDuration: duration
        )
    }

    /// Default MLModel configuration
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
        return config
    }

    /// Load Sortformer models from HuggingFace.
    ///
    /// Downloads models from FluidInference/diar-streaming-sortformer-coreml if not cached.
    ///
    /// - Parameters:
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: CoreML compute units to use (default: cpuOnly for consistency)
    /// - Returns: Loaded SortformerModels
    public static func loadFromHuggingFace(
        config: SortformerConfig,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .all,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> SortformerModels {
        logger.info("Loading Sortformer models from HuggingFace...")

        let startTime = Date()

        // Determine cache directory
        let directory: URL
        if let cache = cacheDirectory {
            directory = cache
        } else {
            directory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("FluidAudio/Models")
        }

        // Determine which file to retrieve
        guard let bundle = ModelNames.Sortformer.bundle(for: config) else {
            throw SortformerError.modelLoadFailed("Unsupported Sortformer configuration")
        }

        logger.info("Downloading Sortformer models from HuggingFace from bundle: \(bundle)...")

        // Download models if needed

        let models = try await DownloadUtils.loadModels(
            .sortformer,
            modelNames: [bundle],
            directory: directory,
            computeUnits: computeUnits,
            variant: bundle,
            progressHandler: progressHandler
        )

        guard let sortformer = models[bundle]
        else {
            throw SortformerError.modelLoadFailed("Failed to load Sortformer models from HuggingFace")
        }

        let duration = Date().timeIntervalSince(startTime)
        logger.info("Sortformer models loaded from HuggingFace in \(String(format: "%.2f", duration))s")

        return try SortformerModels(
            config: config,
            main: sortformer,
            compilationDuration: duration
        )
    }
}

// MARK: - Main Model Inference

extension SortformerModels {

    /// Main model output structure
    public struct MainModelOutput {
        /// Raw predictions (logits) [spkcache_len + fifo_len + chunk_len, num_speakers]
        public let predictions: [Float]

        /// Chunk embeddings [chunk_len, fc_d_model]
        public let chunkEmbeddings: [Float]

        /// Actual chunk embedding length
        public let chunkLength: Int
    }

    /// Run main Sortformer model.
    ///
    /// - Parameters:
    ///   - chunk: Feature chunk [T, 128] transposed from mel
    ///   - chunkLength: Actual chunk length
    ///   - spkcache: Speaker cache embeddings [spkcache_len, 512]
    ///   - spkcacheLength: Actual speaker cache length
    ///   - fifo: FIFO queue embeddings [fifo_len, 512]
    ///   - fifoLength: Actual FIFO length
    ///   - config: Sortformer configuration
    /// - Returns: MainModelOutput with predictions and embeddings
    public func runMainModel(
        chunk: [Float],
        chunkLength: Int,
        spkcache: [Float],
        spkcacheLength: Int,
        fifo: [Float],
        fifoLength: Int,
        config: SortformerConfig
    ) throws -> MainModelOutput {
        func exactCount(_ length: Int, width: Int) -> Int? {
            guard length >= 0 else { return nil }
            let (count, overflow) = length.multipliedReportingOverflow(by: width)
            return overflow ? nil : count
        }

        guard chunkLength >= 0,
            chunkLength <= config.chunkMelFrames,
            chunk.count <= chunkArray.count,
            chunk.count.isMultiple(of: config.melFeatures),
            chunkLength <= chunk.count / config.melFeatures
        else {
            throw SortformerError.invalidState(
                "Chunk features or reported length exceed the configured model input"
            )
        }
        guard let expectedFifoCount = exactCount(fifoLength, width: config.preEncoderDims),
            fifoLength <= config.fifoLen,
            fifo.count == expectedFifoCount,
            fifo.count <= fifoArray.count
        else {
            throw SortformerError.invalidState("FIFO buffer does not match its reported length")
        }
        guard let expectedSpkcacheCount = exactCount(spkcacheLength, width: config.preEncoderDims),
            spkcacheLength <= config.spkcacheLen,
            spkcache.count == expectedSpkcacheCount,
            spkcache.count <= spkcacheArray.count
        else {
            throw SortformerError.invalidState("Speaker-cache buffer does not match its reported length")
        }

        // Copy chunk features
        memoryOptimizer.optimizedCopy(
            from: chunk,
            to: chunkArray,
            pad: true
        )

        // Copy FIFO queue
        memoryOptimizer.optimizedCopy(
            from: fifo,
            to: fifoArray,
            pad: true
        )

        // Copy speaker cache
        memoryOptimizer.optimizedCopy(
            from: spkcache,
            to: spkcacheArray,
            pad: true
        )

        // Create chunk length input
        chunkLengthArray[0] = NSNumber(value: Int32(chunkLength))

        // Create FIFO length input
        fifoLengthArray[0] = NSNumber(value: Int32(fifoLength))

        // Create speaker cache length input
        spkcacheLengthArray[0] = NSNumber(value: Int32(spkcacheLength))

        // Run inference using the feature names resolved from the model's own description, so both
        // the shipped Sortformer exports and Nemotron-style exports work unchanged.
        let names = geometry.names
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            names.chunk: MLFeatureValue(multiArray: chunkArray),
            names.chunkLengths: MLFeatureValue(multiArray: chunkLengthArray),
            names.spkcache: MLFeatureValue(multiArray: spkcacheArray),
            names.spkcacheLengths: MLFeatureValue(multiArray: spkcacheLengthArray),
            names.fifo: MLFeatureValue(multiArray: fifoArray),
            names.fifoLengths: MLFeatureValue(multiArray: fifoLengthArray),
        ])

        let output = try mainModel.prediction(from: inputFeatures)

        // Predictions and embeddings may be Float16 (Nemotron, and fp16 head modules) or Float32.
        guard let predictions = SortformerModelGeometry.floatScalars(from: output, named: names.predictions)
        else {
            throw SortformerError.inferenceFailed("Missing or unreadable output '\(names.predictions)'")
        }

        guard
            let chunkEmbeddingsLength = SortformerModelGeometry.firstInt(
                from: output, named: names.chunkEmbeddingLengths)
        else {
            throw SortformerError.inferenceFailed("Missing or unreadable output '\(names.chunkEmbeddingLengths)'")
        }

        guard
            let chunkEmbeddings = SortformerModelGeometry.floatScalars(from: output, named: names.chunkEmbeddings)
        else {
            throw SortformerError.inferenceFailed("Missing or unreadable output '\(names.chunkEmbeddings)'")
        }

        return MainModelOutput(
            predictions: predictions,
            chunkEmbeddings: chunkEmbeddings,
            chunkLength: Int(chunkEmbeddingsLength)
        )
    }
}
