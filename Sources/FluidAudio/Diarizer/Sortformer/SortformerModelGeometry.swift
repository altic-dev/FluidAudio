@preconcurrency import CoreML
import Foundation

// MARK: - Feature Names

/// CoreML feature names resolved against a concrete Sortformer-family model.
///
/// Two export families are supported:
/// - the shipped Streaming Sortformer exports (`speaker_preds[_out]`, `chunk_pre_encoder_*`)
/// - locally converted Nemotron exports (`spkcache_fifo_chunk_preds`, `chunk_pre_encode_*`)
public struct SortformerFeatureNames: Sendable, Equatable {
    public let chunk: String
    public let chunkLengths: String
    public let spkcache: String
    public let spkcacheLengths: String
    public let fifo: String
    public let fifoLengths: String
    public let predictions: String
    public let chunkEmbeddings: String
    public let chunkEmbeddingLengths: String

    /// Accepted aliases, most-preferred first.
    enum Alias {
        static let chunk = ["chunk", "chunk_in", "audio_signal"]
        static let chunkLengths = ["chunk_lengths", "chunk_length", "audio_signal_lengths"]
        static let spkcache = ["spkcache", "spkcache_in", "speaker_cache"]
        static let spkcacheLengths = ["spkcache_lengths", "spkcache_length", "speaker_cache_lengths"]
        static let fifo = ["fifo", "fifo_in", "fifo_queue"]
        static let fifoLengths = ["fifo_lengths", "fifo_length", "fifo_queue_lengths"]

        static let predictions = [
            "speaker_preds_out",
            "speaker_preds",
            "spkcache_fifo_chunk_preds",
            "spkcache_fifo_chunk_preds_out",
            "preds",
        ]
        static let chunkEmbeddings = [
            "chunk_pre_encoder_embs_out",
            "chunk_pre_encoder_embs",
            "chunk_pre_encode_embs",
            "chunk_pre_encode_embs_out",
        ]
        static let chunkEmbeddingLengths = [
            "chunk_pre_encoder_lengths_out",
            "chunk_pre_encoder_lengths",
            "chunk_pre_encode_lengths",
            "chunk_pre_encode_lengths_out",
        ]
    }
}

// MARK: - Tensor Spec

/// One multi-array feature exactly as the model declares it.
///
/// The shape is stored verbatim so validation can distinguish "the model says 512" from "the model
/// left this axis flexible" from "the model is not even rank-3".
public struct SortformerTensorSpec: Sendable, Equatable {
    public let name: String
    /// Shape as declared by the model. A non-positive entry means the axis is unconstrained.
    public let shape: [Int]
    public let dataType: MLMultiArrayDataType

    public init(name: String, shape: [Int], dataType: MLMultiArrayDataType) {
        self.name = name
        self.shape = shape
        self.dataType = dataType
    }

    init(name: String, constraint: MLMultiArrayConstraint) {
        self.init(name: name, shape: constraint.shape.map(\.intValue), dataType: constraint.dataType)
    }

    /// `true` when the model declares `[1, time, features]`, the layout every input buffer uses.
    public var isBatchOneRank3: Bool {
        shape.count == 3 && shape[0] == 1
    }

    /// Time axis, or `nil` when the tensor is not a fixed rank-3 batch-1 shape.
    public var timeDimension: Int? { fixedAxis(1) }

    /// Feature axis, or `nil` when the tensor is not a fixed rank-3 batch-1 shape.
    public var featureDimension: Int? { fixedAxis(2) }

    /// `true` when the model declares exactly `[1]`, the layout of every length feature.
    public var isSingleElement: Bool {
        shape == [1]
    }

    public var shapeDescription: String {
        "[" + shape.map(String.init).joined(separator: ", ") + "]"
    }

    private func fixedAxis(_ axis: Int) -> Int? {
        guard isBatchOneRank3, shape[axis] > 0 else { return nil }
        return shape[axis]
    }
}

// MARK: - Model Geometry

/// Tensor geometry read from a model's `MLModelDescription`.
///
/// Nothing here is inferred from a file name or a README — every value comes from the compiled
/// model's own feature descriptions. Axes the model leaves unconstrained are preserved as declared
/// and are *rejected* by ``validate(against:)`` rather than skipped, because the runtime allocates
/// fixed-size input buffers and cannot adapt to a flexible axis.
public struct SortformerModelGeometry: Sendable {
    public let chunk: SortformerTensorSpec
    public let chunkLengths: SortformerTensorSpec
    public let spkcache: SortformerTensorSpec
    public let spkcacheLengths: SortformerTensorSpec
    public let fifo: SortformerTensorSpec
    public let fifoLengths: SortformerTensorSpec
    public let predictions: SortformerTensorSpec
    public let chunkEmbeddings: SortformerTensorSpec
    public let chunkEmbeddingLengths: SortformerTensorSpec

    public init(
        chunk: SortformerTensorSpec,
        chunkLengths: SortformerTensorSpec,
        spkcache: SortformerTensorSpec,
        spkcacheLengths: SortformerTensorSpec,
        fifo: SortformerTensorSpec,
        fifoLengths: SortformerTensorSpec,
        predictions: SortformerTensorSpec,
        chunkEmbeddings: SortformerTensorSpec,
        chunkEmbeddingLengths: SortformerTensorSpec
    ) {
        self.chunk = chunk
        self.chunkLengths = chunkLengths
        self.spkcache = spkcache
        self.spkcacheLengths = spkcacheLengths
        self.fifo = fifo
        self.fifoLengths = fifoLengths
        self.predictions = predictions
        self.chunkEmbeddings = chunkEmbeddings
        self.chunkEmbeddingLengths = chunkEmbeddingLengths
    }

    public var names: SortformerFeatureNames {
        SortformerFeatureNames(
            chunk: chunk.name,
            chunkLengths: chunkLengths.name,
            spkcache: spkcache.name,
            spkcacheLengths: spkcacheLengths.name,
            fifo: fifo.name,
            fifoLengths: fifoLengths.name,
            predictions: predictions.name,
            chunkEmbeddings: chunkEmbeddings.name,
            chunkEmbeddingLengths: chunkEmbeddingLengths.name
        )
    }

    /// The five `[1, time, features]` tensors the runtime binds fixed-size buffers to.
    var tensors: [SortformerTensorSpec] {
        [chunk, spkcache, fifo, predictions, chunkEmbeddings]
    }

    /// The four `Int32 [1]` length features.
    var lengthFeatures: [SortformerTensorSpec] {
        [chunkLengths, spkcacheLengths, fifoLengths, chunkEmbeddingLengths]
    }

    /// Mel frames per chunk, i.e. the `chunk` input's time dimension.
    public var chunkMelFrames: Int? { chunk.timeDimension }
    /// Mel filterbank count, i.e. the `chunk` input's feature dimension.
    public var melFeatures: Int? { chunk.featureDimension }
    /// Speaker cache capacity, i.e. the `spkcache` input's time dimension.
    public var spkcacheLen: Int? { spkcache.timeDimension }
    /// FIFO capacity, i.e. the `fifo` input's time dimension.
    public var fifoLen: Int? { fifo.timeDimension }
    /// Speaker slots, i.e. the prediction output's trailing dimension.
    public var numSpeakers: Int? { predictions.featureDimension }
    /// Prediction rows, i.e. the prediction output's time dimension.
    public var predictionFrames: Int? { predictions.timeDimension }
    /// Encoder frames emitted per chunk, i.e. the chunk embedding output's time dimension.
    public var chunkEncoderFrames: Int? { chunkEmbeddings.timeDimension }

    /// Pre-encoder embedding width, or `nil` when the model's own tensors disagree (or leave the
    /// axis flexible). A disagreement is never resolved by picking a side — ``validate(against:)``
    /// checks each tensor's width individually and reports every offender.
    public var preEncoderDims: Int? {
        let widths = Set([spkcache.featureDimension, fifo.featureDimension, chunkEmbeddings.featureDimension])
        guard widths.count == 1, let width = widths.first else { return nil }
        return width
    }

    /// Human-readable dump, used by the opt-in real-model smoke test and by load-failure logs.
    public var debugDescription: String {
        func line(_ label: String, _ spec: SortformerTensorSpec) -> String {
            "\(label)[\(spec.name)]: \(spec.shapeDescription) \(spec.dataType.fluidName)"
        }
        return [
            line("chunk", chunk),
            line("chunk_lengths", chunkLengths),
            line("spkcache", spkcache),
            line("spkcache_lengths", spkcacheLengths),
            line("fifo", fifo),
            line("fifo_lengths", fifoLengths),
            line("preds", predictions),
            line("embs", chunkEmbeddings),
            line("emb_lengths", chunkEmbeddingLengths),
        ].joined(separator: "\n")
    }
}

// MARK: - Inspection

extension SortformerModelGeometry {

    /// Read the geometry of a loaded Sortformer-family model from its own description.
    ///
    /// - Throws: `SortformerError.modelLoadFailed` when a required input or output is missing, or is
    ///   not a multi-array.
    public static func inspect(model: MLModel) throws -> SortformerModelGeometry {
        let description = model.modelDescription
        let inputs = description.inputDescriptionsByName
        let outputs = description.outputDescriptionsByName

        func input(_ aliases: [String]) throws -> SortformerTensorSpec {
            try spec(aliases, in: inputs, kind: "input")
        }
        func output(_ aliases: [String]) throws -> SortformerTensorSpec {
            try spec(aliases, in: outputs, kind: "output")
        }

        return SortformerModelGeometry(
            chunk: try input(SortformerFeatureNames.Alias.chunk),
            chunkLengths: try input(SortformerFeatureNames.Alias.chunkLengths),
            spkcache: try input(SortformerFeatureNames.Alias.spkcache),
            spkcacheLengths: try input(SortformerFeatureNames.Alias.spkcacheLengths),
            fifo: try input(SortformerFeatureNames.Alias.fifo),
            fifoLengths: try input(SortformerFeatureNames.Alias.fifoLengths),
            predictions: try output(SortformerFeatureNames.Alias.predictions),
            chunkEmbeddings: try output(SortformerFeatureNames.Alias.chunkEmbeddings),
            chunkEmbeddingLengths: try output(SortformerFeatureNames.Alias.chunkEmbeddingLengths)
        )
    }

    private static func spec(
        _ aliases: [String],
        in descriptions: [String: MLFeatureDescription],
        kind: String
    ) throws -> SortformerTensorSpec {
        for alias in aliases {
            guard let description = descriptions[alias] else { continue }
            guard let constraint = description.multiArrayConstraint else {
                throw SortformerError.modelLoadFailed("\(kind.capitalized) '\(alias)' is not a multi-array")
            }
            return SortformerTensorSpec(name: alias, constraint: constraint)
        }
        let available = descriptions.keys.sorted().joined(separator: ", ")
        throw SortformerError.modelLoadFailed(
            "Model has no \(kind) matching any of [\(aliases.joined(separator: ", "))]. Available: [\(available)]"
        )
    }
}

// MARK: - Validation

extension SortformerModelGeometry {

    /// Data types the runtime can read a float output from.
    ///
    /// `int32` is deliberately excluded: a quantized-to-integer prediction or embedding tensor would
    /// decode to garbage probabilities rather than fail loudly.
    static let supportedFloatOutputTypes: [MLMultiArrayDataType] = [.float16, .float32, .double]

    /// Check that a configuration can actually drive this model.
    ///
    /// Every mismatch is collected so a misconfigured run reports the full picture in one error
    /// instead of one dimension per attempt.
    ///
    /// - Throws: `SortformerError.configurationError` listing every mismatch.
    public func validate(against config: SortformerConfig) throws {
        try config.validateGeometry()
        var problems: [String] = []

        /// Require a fixed `[1, time, features]` shape and match it against the configuration.
        func checkTensor(
            _ label: String,
            _ spec: SortformerTensorSpec,
            time: Int,
            timeLabel: String,
            features: Int,
            featuresLabel: String
        ) {
            guard spec.isBatchOneRank3 else {
                problems.append(
                    "\(label) '\(spec.name)' must be rank-3 [1, \(time), \(features)] but the model declares "
                        + spec.shapeDescription
                )
                return
            }
            checkAxis(label, spec, timeLabel, spec.timeDimension, time)
            checkAxis(label, spec, featuresLabel, spec.featureDimension, features)
        }

        func checkAxis(
            _ label: String, _ spec: SortformerTensorSpec, _ axisLabel: String, _ actual: Int?, _ expected: Int
        ) {
            guard let actual else {
                problems.append(
                    "\(label) '\(spec.name)' \(axisLabel) is not a fixed dimension "
                        + "(model declares \(spec.shapeDescription), config expects \(expected))"
                )
                return
            }
            guard actual != expected else { return }
            problems.append("\(label) '\(spec.name)' \(axisLabel): model=\(actual) config=\(expected)")
        }

        checkTensor(
            "chunk input", chunk,
            time: config.chunkMelFrames, timeLabel: "mel frames",
            features: config.melFeatures, featuresLabel: "mel features")
        checkTensor(
            "spkcache input", spkcache,
            time: config.spkcacheLen, timeLabel: "cache length",
            features: config.preEncoderDims, featuresLabel: "pre-encoder dims")
        checkTensor(
            "fifo input", fifo,
            time: config.fifoLen, timeLabel: "fifo length",
            features: config.preEncoderDims, featuresLabel: "pre-encoder dims")
        checkTensor(
            "predictions output", predictions,
            time: config.predictionFrames, timeLabel: "prediction frames",
            features: config.numSpeakers, featuresLabel: "speaker count")
        checkTensor(
            "chunk embeddings output", chunkEmbeddings,
            time: config.chunkEncoderFrames, timeLabel: "encoder frames",
            features: config.preEncoderDims, featuresLabel: "pre-encoder dims")

        // The copy path into the model's input buffers is float32-only.
        for spec in [chunk, spkcache, fifo] where spec.dataType != .float32 {
            problems.append(
                "input '\(spec.name)' dtype \(spec.dataType.fluidName) is unsupported (expected float32)")
        }

        for spec in [predictions, chunkEmbeddings]
        where !Self.supportedFloatOutputTypes.contains(spec.dataType) {
            problems.append(
                "output '\(spec.name)' dtype \(spec.dataType.fluidName) is unsupported "
                    + "(expected float16, float32 or double)")
        }

        // Every length feature is bound as a single Int32 element in both directions.
        for spec in lengthFeatures {
            if !spec.isSingleElement {
                problems.append(
                    "length feature '\(spec.name)' must be shape [1] but the model declares "
                        + spec.shapeDescription)
            }
            if spec.dataType != .int32 {
                problems.append(
                    "length feature '\(spec.name)' dtype \(spec.dataType.fluidName) is unsupported "
                        + "(expected int32)")
            }
        }

        guard problems.isEmpty else {
            throw SortformerError.configurationError(
                "Sortformer model/config mismatch: " + problems.joined(separator: "; ")
                    + ". Model geometry:\n\(debugDescription)"
            )
        }
    }
}

// MARK: - Output Reading

extension SortformerModelGeometry {

    /// Read a float output regardless of whether the export emits Float32 or Float16.
    ///
    /// Nemotron emits Float16 predictions and embeddings; the shipped Sortformer exports emit Float32
    /// predictions and (depending on the conversion) Float16 embeddings.
    static func floatScalars(from provider: MLFeatureProvider, named name: String) -> [Float]? {
        guard let value = provider.featureValue(for: name) else { return nil }

        if let scalars = value.shapedArrayValue(of: Float32.self)?.scalars {
            return scalars
        }

        #if arch(arm64)
        if #available(macOS 15.0, iOS 18.0, *),
            let scalars = value.shapedArrayValue(of: Float16.self)?.scalars
        {
            return scalars.map { Float($0) }
        }
        #endif

        guard let array = value.multiArrayValue else { return nil }
        return floatScalars(from: array)
    }

    /// Element-wise fallback for data types without a `MLShapedArray` fast path.
    static func floatScalars(from array: MLMultiArray) -> [Float]? {
        guard array.dataType.isReadableAsFloat else { return nil }
        var scalars = [Float](repeating: 0, count: array.count)
        for index in 0..<array.count {
            scalars[index] = array[index].floatValue
        }
        return scalars
    }

    /// Read the first element of an integer output (chunk embedding lengths).
    static func firstInt(from provider: MLFeatureProvider, named name: String) -> Int? {
        guard let value = provider.featureValue(for: name) else { return nil }

        if let scalar = value.shapedArrayValue(of: Int32.self)?.scalars.first {
            return Int(scalar)
        }
        if let array = value.multiArrayValue, array.count > 0 {
            return array[0].intValue
        }
        return nil
    }
}

// MARK: - Data Type Helpers

extension MLMultiArrayDataType {
    /// Short name for diagnostics (`description` is not stable across SDKs).
    var fluidName: String {
        switch self {
        case .float32: return "float32"
        case .float16: return "float16"
        case .double: return "double"
        case .int32: return "int32"
        default: return "unknown(\(rawValue))"
        }
    }

    var isReadableAsFloat: Bool {
        switch self {
        case .float32, .float16, .double: return true
        default: return false
        }
    }
}
