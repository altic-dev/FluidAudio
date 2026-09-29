import Foundation

/// Configuration constants for the SenseVoiceSmall CoreML pipeline.
///
/// SenseVoiceSmall is non-autoregressive: a SANM encoder + single CTC head.
/// The CoreML export is a 3-stage pipeline: preprocessor, encoder+CTC, then
/// host-side greedy decode.
public enum SenseVoiceConfig {
    /// LFR feature dimension (80-bin fbank * LFR m=7).
    public static let featureDim = 560

    /// Enumerated encoder sequence-length buckets after LFR framing.
    public static let buckets = [128, 256, 512, 1024, 1800]

    /// Query tokens prepended by the encoder: language, emotion, event, text norm.
    public static let numQueryTokens = 4

    /// SenseVoice uses `<unk>` = 0 as the CTC blank.
    public static let blankId = 0

    /// Auto-detect language.
    public static let defaultLanguage: Int32 = 0

    /// English language embedding index in FunASR/SenseVoice's `lid_dict`.
    public static let englishLanguage: Int32 = 4

    /// `woitn`: no inverse text normalization.
    public static let defaultTextNorm: Int32 = 15

    public static let sampleRate = 16_000

    /// Kaldi feeds waveforms in int16 range; AudioConverter yields [-1, 1].
    public static let waveformScale: Float = 32_768.0

    /// Largest supported feature length, about 108 seconds of audio.
    public static var maxFrames: Int { buckets.last ?? 1800 }

    public static func pickBucket(forFrames frames: Int) -> Int {
        for bucket in buckets where bucket >= frames { return bucket }
        return buckets.last ?? 1800
    }
}
