import Accelerate
import Foundation

/// Log-mel + delta front-end for Granite Speech 5.0 TurboCTC.
///
/// Mirrors `GraniteSpeech5FeatureExtractor` from `transformers` exactly:
///
/// 1. power mel spectrogram (`n_fft` 512, `win_length` 400, `hop_length` 160, 80 mels)
/// 2. `log10`, floored at `max - logmel_floor_db` **over the whole clip**, then `/ 4 + 1`
/// 3. first-order deltas over the normalised values, replicate-padded at the edges
/// 4. concatenate `[mel, delta]` per frame, then stack pairs of adjacent frames
///
/// The result is one 320-wide row per two mel frames, which is what the encoder
/// consumes. The floor is clip-global, so the whole utterance is featurised in one
/// pass rather than per window -- a per-window maximum would drift from the
/// reference implementation.
public final class GraniteTurboCtcFeatureExtractor {
    public struct Features {
        /// Row-major `(stackedFrames, featureDim)`.
        public let values: [Float]
        public let stackedFrames: Int
        public let featureDim: Int
    }

    private let manifest: GraniteTurboCtcManifest
    private let melFilterbank: [Float]
    private let hannWindow: [Float]
    private let numFreqBins: Int
    private let windowOffset: Int
    private var fftSetup: vDSP_DFT_Setup?

    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var powerSpectrum: [Float]
    private var imagSquared: [Float]
    private var frame: [Float]
    private var melFrame: [Float]
    private var paddedAudio: [Float]

    public init(manifest: GraniteTurboCtcManifest, melFilterbank: [Float]) throws {
        self.manifest = manifest
        self.melFilterbank = melFilterbank
        numFreqBins = manifest.nFFT / 2 + 1
        windowOffset = (manifest.nFFT - manifest.winLength) / 2
        hannWindow = Self.periodicHannWindow(length: manifest.winLength)

        guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(manifest.nFFT), .FORWARD) else {
            throw GraniteTurboCtcError.invalidAudio("Failed to create FFT setup")
        }
        fftSetup = setup

        realIn = [Float](repeating: 0, count: manifest.nFFT)
        imagIn = [Float](repeating: 0, count: manifest.nFFT)
        realOut = [Float](repeating: 0, count: manifest.nFFT)
        imagOut = [Float](repeating: 0, count: manifest.nFFT)
        powerSpectrum = [Float](repeating: 0, count: numFreqBins)
        imagSquared = [Float](repeating: 0, count: numFreqBins)
        frame = [Float](repeating: 0, count: manifest.nFFT)
        melFrame = [Float](repeating: 0, count: manifest.nMels)
        paddedAudio = []
    }

    deinit {
        if let fftSetup {
            vDSP_DFT_DestroySetup(fftSetup)
        }
    }

    public func extract(audio: [Float]) throws -> Features {
        guard !audio.isEmpty else {
            throw GraniteTurboCtcError.invalidAudio("Audio is empty")
        }

        let stackFactor = manifest.stackFactor
        let melFrames = audio.count / manifest.hopLength
        // Round up so the trailing stacking pair is filled rather than dropped.
        let frameCount = stackFactor * ((melFrames + stackFactor - 1) / stackFactor)
        guard frameCount > 0 else {
            throw GraniteTurboCtcError.invalidAudio("Audio is shorter than one mel frame")
        }

        let neededSamples = (frameCount - 1) * manifest.hopLength + 1
        var source = audio
        if source.count < neededSamples {
            source.append(contentsOf: [Float](repeating: 0, count: neededSamples - source.count))
        }

        let logMel = computeLogMel(audio: source, frameCount: frameCount)
        let normalized = normalize(logMel)
        let deltas = computeDeltas(normalized, frameCount: frameCount)
        let values = stack(mel: normalized, deltas: deltas, frameCount: frameCount)

        return Features(
            values: values,
            stackedFrames: frameCount / stackFactor,
            featureDim: manifest.featureDim
        )
    }

    // MARK: - Stages

    private func computeLogMel(audio: [Float], frameCount: Int) -> [Float] {
        let padCount = makeReflectPaddedAudio(audio)
        var logMel = [Float](repeating: 0, count: frameCount * manifest.nMels)

        for frameIndex in 0..<frameCount {
            computeMelFrame(frameIndex: frameIndex, paddedAudioCount: padCount)
            let base = frameIndex * manifest.nMels
            for melIndex in 0..<manifest.nMels {
                logMel[base + melIndex] = log10f(max(melFrame[melIndex], 1e-10))
            }
        }
        return logMel
    }

    private func normalize(_ logMel: [Float]) -> [Float] {
        var maximum: Float = -.greatestFiniteMagnitude
        vDSP_maxv(logMel, 1, &maximum, vDSP_Length(logMel.count))
        let floorValue = maximum - manifest.logmelFloorDB

        var normalized = logMel
        for index in 0..<normalized.count {
            normalized[index] = (max(normalized[index], floorValue) / 4.0) + 1.0
        }
        return normalized
    }

    /// `torchaudio.functional.compute_deltas` with `win_length == 3` reduces to
    /// `(x[t + 1] - x[t - 1]) / 2`, with the sequence replicate-padded at both ends.
    private func computeDeltas(_ values: [Float], frameCount: Int) -> [Float] {
        let mels = manifest.nMels
        var deltas = [Float](repeating: 0, count: values.count)

        for frameIndex in 0..<frameCount {
            let previous = max(frameIndex - 1, 0) * mels
            let next = min(frameIndex + 1, frameCount - 1) * mels
            let destination = frameIndex * mels
            for melIndex in 0..<mels {
                deltas[destination + melIndex] = (values[next + melIndex] - values[previous + melIndex]) / 2.0
            }
        }
        return deltas
    }

    /// Row layout per stacked frame: `[mel(t), delta(t), mel(t+1), delta(t+1)]`.
    private func stack(mel: [Float], deltas: [Float], frameCount: Int) -> [Float] {
        let mels = manifest.nMels
        let stackFactor = manifest.stackFactor
        let stackedFrames = frameCount / stackFactor
        let stride = manifest.featureDim
        var output = [Float](repeating: 0, count: stackedFrames * stride)

        for stackedIndex in 0..<stackedFrames {
            let destination = stackedIndex * stride
            for offset in 0..<stackFactor {
                let source = (stackedIndex * stackFactor + offset) * mels
                let melDestination = destination + offset * mels * 2
                let deltaDestination = melDestination + mels
                for melIndex in 0..<mels {
                    output[melDestination + melIndex] = mel[source + melIndex]
                    output[deltaDestination + melIndex] = deltas[source + melIndex]
                }
            }
        }
        return output
    }

    // MARK: - STFT

    private func makeReflectPaddedAudio(_ audio: [Float]) -> Int {
        let pad = manifest.nFFT / 2
        let paddedCount = audio.count + 2 * pad
        if paddedAudio.count < paddedCount {
            paddedAudio = [Float](repeating: 0, count: paddedCount)
        }
        for paddedIndex in 0..<paddedCount {
            paddedAudio[paddedIndex] = audio[reflectIndex(paddedIndex - pad, count: audio.count)]
        }
        return paddedCount
    }

    private func reflectIndex(_ index: Int, count: Int) -> Int {
        guard count > 1 else { return 0 }
        var reflected = index
        while reflected < 0 || reflected >= count {
            reflected = reflected < 0 ? -reflected : 2 * count - reflected - 2
        }
        return reflected
    }

    private func computeMelFrame(frameIndex: Int, paddedAudioCount: Int) {
        vDSP_vclr(&frame, 1, vDSP_Length(manifest.nFFT))
        let audioStart = frameIndex * manifest.hopLength
        let available = min(manifest.winLength, paddedAudioCount - audioStart)

        if available > 0 {
            paddedAudio.withUnsafeBufferPointer { audioPtr in
                hannWindow.withUnsafeBufferPointer { windowPtr in
                    frame.withUnsafeMutableBufferPointer { framePtr in
                        vDSP_vmul(
                            audioPtr.baseAddress! + audioStart, 1,
                            windowPtr.baseAddress!, 1,
                            framePtr.baseAddress! + windowOffset, 1,
                            vDSP_Length(available)
                        )
                    }
                }
            }
        }

        computePowerSpectrum()
        melFilterbank.withUnsafeBufferPointer { filterPtr in
            powerSpectrum.withUnsafeBufferPointer { spectrumPtr in
                melFrame.withUnsafeMutableBufferPointer { outputPtr in
                    vDSP_mmul(
                        filterPtr.baseAddress!, 1,
                        spectrumPtr.baseAddress!, 1,
                        outputPtr.baseAddress!, 1,
                        vDSP_Length(manifest.nMels),
                        vDSP_Length(1),
                        vDSP_Length(numFreqBins)
                    )
                }
            }
        }
    }

    private func computePowerSpectrum() {
        guard let fftSetup else { return }

        frame.withUnsafeBufferPointer { source in
            realIn.withUnsafeMutableBufferPointer { destination in
                _ = memcpy(
                    destination.baseAddress!,
                    source.baseAddress!,
                    manifest.nFFT * MemoryLayout<Float>.size
                )
            }
        }
        vDSP_vclr(&imagIn, 1, vDSP_Length(manifest.nFFT))
        vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)

        vDSP_vsq(realOut, 1, &powerSpectrum, 1, vDSP_Length(numFreqBins))
        vDSP_vsq(imagOut, 1, &imagSquared, 1, vDSP_Length(numFreqBins))
        vDSP_vadd(powerSpectrum, 1, imagSquared, 1, &powerSpectrum, 1, vDSP_Length(numFreqBins))
    }

    private static func periodicHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        for index in 0..<length {
            window[index] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(index) / Float(length)))
        }
        return window
    }
}
