@preconcurrency import CoreML
import Foundation

/// Runs a contiguous sequence of compiled encoder pieces on one actor, without a model subclass.
public actor ParakeetSplitEncoder {
    private let pieces: [MLModel]
    public nonisolated let inputFeatureNames: [String]
    public nonisolated let outputShape: [Int]

    private init(pieces: [MLModel]) {
        self.pieces = pieces
        self.inputFeatureNames = pieces[0].modelDescription.inputDescriptionsByName.keys.sorted()
        self.outputShape =
            pieces.last?.modelDescription.outputDescriptionsByName["encoder"]?
            .multiArrayConstraint?.shape.map(\.intValue) ?? []
    }

    /// Bounded discovery rejects gaps, incomplete bundles and more than sixteen pieces.
    /// A missing split encoder returns nil, allowing the ordinary monolithic path.
    public static func pieceURLs(in directory: URL) throws -> [URL]? {
        let manager = FileManager.default
        if manager.fileExists(atPath: directory.appendingPathComponent("Encoder.mlmodelc/coremldata.bin").path) {
            return nil
        }
        let names = (1...17).map { "Encoder-\($0).mlmodelc" }
        let present = names.filter { manager.fileExists(atPath: directory.appendingPathComponent($0).path) }
        guard !present.isEmpty else { return nil }
        guard (2...16).contains(present.count), present == Array(names.prefix(present.count)) else {
            throw AsrModelsError.loadingFailed("Split encoder requires contiguous Encoder-1…N pieces (2–16)")
        }
        let urls = present.map { directory.appendingPathComponent($0, isDirectory: true) }
        for url in urls {
            guard manager.fileExists(atPath: url.appendingPathComponent("coremldata.bin").path) else {
                throw AsrModelsError.modelNotFound("coremldata.bin", url)
            }
        }
        return urls
    }

    static func load(
        urls: [URL], configuration: MLModelConfiguration, hiddenSize: Int
    ) async throws -> ParakeetSplitEncoder {
        var pieces: [MLModel] = []
        for url in urls {
            try Task.checkCancellation()
            pieces.append(try await MLModel.load(contentsOf: url, configuration: configuration))
        }
        try Task.checkCancellation()
        guard let first = pieces.first, let last = pieces.last,
            first.modelDescription.inputDescriptionsByName["mel"] != nil,
            first.modelDescription.inputDescriptionsByName["mel_length"] != nil,
            last.modelDescription.outputDescriptionsByName["encoder_length"] != nil,
            let shape = last.modelDescription.outputDescriptionsByName["encoder"]?.multiArrayConstraint?.shape,
            shape.count == 3, shape[1].intValue == hiddenSize
        else { throw AsrModelsError.loadingFailed("Split encoder does not match the selected Parakeet contract") }
        for (index, piece) in pieces.enumerated() {
            if index > 0, piece.modelDescription.inputDescriptionsByName["hidden_in"] == nil {
                throw AsrModelsError.loadingFailed("Split encoder piece is missing hidden_in")
            }
            if index < pieces.count - 1, piece.modelDescription.outputDescriptionsByName["hidden_out"] == nil {
                throw AsrModelsError.loadingFailed("Split encoder piece is missing hidden_out")
            }
        }
        return ParakeetSplitEncoder(pieces: pieces)
    }

    /// Synchronous Core ML calls inside the actor prevent overlapping piece chains.
    /// Caller-provided backings apply only to the final output, never intermediate shapes.
    func prediction(from input: MLFeatureProvider, options: MLPredictionOptions) throws -> MLFeatureProvider {
        try Task.checkCancellation()
        guard let melLength = input.featureValue(for: "mel_length") else {
            throw AsrModelsError.loadingFailed("Split encoder input is missing mel_length")
        }
        var nextInput = input
        var output: MLFeatureProvider = input
        for (index, piece) in pieces.enumerated() {
            try Task.checkCancellation()
            let pieceOptions = index == pieces.count - 1 ? options : MLPredictionOptions()
            output = try piece.prediction(from: nextInput, options: pieceOptions)
            if index < pieces.count - 1 {
                guard let hidden = output.featureValue(for: "hidden_out") else {
                    throw AsrModelsError.loadingFailed("Split encoder output is missing hidden_out")
                }
                nextInput = try MLDictionaryFeatureProvider(dictionary: ["hidden_in": hidden, "mel_length": melLength])
            }
        }
        try Task.checkCancellation()
        // Copy while actor isolation still holds: a shared encoder cannot overwrite a retained window.
        return try AsrModels.snapshotFeatureProvider(output)
    }
}
