import Foundation

/// Catalogues all available true streaming ASR model variants with native streaming encoders.
///
/// These are models with native streaming architectures with cache-aware or chunked-attention encoders. This does **not** include Parakeet TDT, which uses an offline encoder
/// in a sliding-window pseudo-streaming mode (use `AsrModelVersion` + `SlidingWindowAsrManager`
/// directly for TDT).
///
/// Use with `StreamingAsrEngineFactory.create(_:)` to instantiate the appropriate engine.
///
/// Following the `CtcModelVariant` pattern for consistency.
public enum StreamingModelVariant: String, CaseIterable, Sendable {
    // MARK: - Parakeet EOU (cache-aware streaming encoder, 120M params)

    /// Parakeet EOU 120M with 160ms chunks (lowest latency)
    case parakeetEou160ms = "parakeet-eou-160ms"
    /// Parakeet EOU 120M with 320ms chunks (balanced)
    case parakeetEou320ms = "parakeet-eou-320ms"
    /// Parakeet EOU 120M with 1280ms chunks (highest throughput)
    case parakeetEou1280ms = "parakeet-eou-1280ms"

    // MARK: - Nemotron Speech Streaming (cache-aware streaming, 0.6B params)

    /// Nemotron 0.6B with 560ms chunks (balanced)
    case nemotron560ms = "nemotron-560ms"
    /// Nemotron 0.6B with 1120ms chunks (best accuracy)
    case nemotron1120ms = "nemotron-1120ms"

    // MARK: - Parakeet Unified (chunked-attention streaming, 0.6B params)

    /// Parakeet Unified with 320 ms of chunk and right context.
    case parakeetUnified320ms = "parakeet-unified-320ms"
    /// Parakeet Unified with 640 ms of chunk and right context.
    case parakeetUnified640ms = "parakeet-unified-640ms"
    /// Parakeet Unified with 1120 ms of chunk and right context.
    case parakeetUnified1120ms = "parakeet-unified-1120ms"
    /// Parakeet Unified with 2080 ms of chunk and right context.
    case parakeetUnified2080ms = "parakeet-unified-2080ms"

    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .parakeetEou160ms: return "Parakeet EOU 120M (160ms)"
        case .parakeetEou320ms: return "Parakeet EOU 120M (320ms)"
        case .parakeetEou1280ms: return "Parakeet EOU 120M (1280ms)"
        case .nemotron560ms: return "Nemotron 0.6B (560ms)"
        case .nemotron1120ms: return "Nemotron 0.6B (1120ms)"
        case .parakeetUnified320ms: return "Parakeet Unified 0.6B (320ms)"
        case .parakeetUnified640ms: return "Parakeet Unified 0.6B (640ms)"
        case .parakeetUnified1120ms: return "Parakeet Unified 0.6B (1120ms)"
        case .parakeetUnified2080ms: return "Parakeet Unified 0.6B (2080ms)"
        }
    }

    /// The HuggingFace repo for this variant's CoreML models
    public var repo: Repo {
        switch self {
        case .parakeetEou160ms: return .parakeetEou160
        case .parakeetEou320ms: return .parakeetEou320
        case .parakeetEou1280ms: return .parakeetEou1280
        case .nemotron560ms: return .nemotronStreaming560
        case .nemotron1120ms: return .nemotronStreaming1120
        case .parakeetUnified320ms, .parakeetUnified640ms, .parakeetUnified1120ms, .parakeetUnified2080ms:
            return .parakeetUnified
        }
    }

    /// Engine family grouping for factory dispatch
    public var engineFamily: EngineFamily {
        switch self {
        case .parakeetEou160ms, .parakeetEou320ms, .parakeetEou1280ms:
            return .parakeetEou
        case .nemotron560ms, .nemotron1120ms:
            return .nemotron
        case .parakeetUnified320ms, .parakeetUnified640ms, .parakeetUnified1120ms, .parakeetUnified2080ms:
            return .parakeetUnified
        }
    }

    /// The streaming chunk size for EOU variants (nil for non-EOU)
    public var eouChunkSize: StreamingChunkSize? {
        switch self {
        case .parakeetEou160ms: return .ms160
        case .parakeetEou320ms: return .ms320
        case .parakeetEou1280ms: return .ms1280
        default: return nil
        }
    }

    /// The streaming chunk size for Nemotron variants (nil for non-Nemotron)
    public var nemotronChunkSize: NemotronChunkSize? {
        switch self {
        case .nemotron560ms: return .ms560
        case .nemotron1120ms: return .ms1120
        default: return nil
        }
    }

    /// Attention context matching the selected Unified encoder, or nil for other families.
    public var unifiedConfig: UnifiedConfig? {
        switch self {
        case .parakeetUnified320ms: return UnifiedConfig(chunkFrames: 2, rightFrames: 2)
        case .parakeetUnified640ms: return UnifiedConfig(chunkFrames: 7, rightFrames: 1)
        case .parakeetUnified1120ms: return UnifiedConfig(chunkFrames: 7, rightFrames: 7)
        case .parakeetUnified2080ms: return UnifiedConfig()
        default: return nil
        }
    }

    /// Engine family types for true streaming models
    public enum EngineFamily: String, Sendable {
        /// Parakeet EOU: cache-aware streaming with end-of-utterance detection
        case parakeetEou = "parakeet-eou"
        /// Nemotron: cache-aware streaming with encoder cache states
        case nemotron = "nemotron"
        /// Parakeet Unified: stateless chunked-attention encoder and persistent RNNT decoder.
        case parakeetUnified = "parakeet-unified"
    }
}
