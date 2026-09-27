@preconcurrency import CoreML
import Foundation

/// Factory for creating true streaming ASR engines from model variants.
///
/// Only creates engines with native streaming architectures (EOU, Nemotron, Unified).
/// For Parakeet TDT (sliding-window pseudo-streaming), use `SlidingWindowAsrManager` directly.
///
/// Usage:
/// ```swift
/// let engine = StreamingAsrEngineFactory.create(.parakeetEou160ms)
/// try await engine.loadModels()
/// try engine.appendAudio(buffer)
/// try await engine.processBufferedAudio()
/// let text = try await engine.finish()
/// ```
public enum StreamingAsrEngineFactory {

    /// Create a streaming ASR engine for the given model variant.
    ///
    /// The returned engine is not yet loaded — call `loadModels()` before use.
    ///
    /// - Parameters:
    ///   - variant: The streaming model variant to use.
    ///   - configuration: Optional `MLModelConfiguration` override.
    /// - Returns: A streaming ASR engine conforming to `StreamingAsrEngine`.
    public static func create(
        _ variant: StreamingModelVariant,
        configuration: MLModelConfiguration? = nil
    ) -> any StreamingAsrEngine {
        switch variant.engineFamily {
        case .parakeetEou:
            return createEouEngine(variant: variant, configuration: configuration)

        case .nemotron:
            return createNemotronEngine(variant: variant, configuration: configuration)

        case .parakeetUnified:
            let mlConfig = configuration?.copy() as? MLModelConfiguration ?? AsrModels.defaultConfiguration()
            return StreamingUnifiedAsrManager(
                configuration: mlConfig, config: variant.unifiedConfig ?? UnifiedConfig()
            )
        }
    }

    // MARK: - Private Factory Methods

    private static func createEouEngine(
        variant: StreamingModelVariant,
        configuration: MLModelConfiguration?
    ) -> StreamingEouAsrManager {
        let chunkSize = variant.eouChunkSize ?? .ms160
        let mlConfig = configuration ?? MLModelConfiguration()
        return StreamingEouAsrManager(configuration: mlConfig, chunkSize: chunkSize)
    }

    private static func createNemotronEngine(
        variant: StreamingModelVariant,
        configuration: MLModelConfiguration?
    ) -> NemotronStreamingAsrManager {
        let mlConfig = configuration ?? NemotronStreamingAsrManager.defaultModelConfiguration()
        let chunkSize = variant.nemotronChunkSize ?? .ms1120
        return NemotronStreamingAsrManager(configuration: mlConfig, requestedChunkSize: chunkSize)
    }
}
