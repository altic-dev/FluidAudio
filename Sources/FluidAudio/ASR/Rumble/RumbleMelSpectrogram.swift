#if arch(arm64)
import Accelerate
import Foundation

struct RumbleMelSpectrogram {
    private let fftLength: Int
    private let hop: Int
    private let bins: Int
    private let frequencies: Int
    private let filters: [Float]
    private let window: [Float]
    private let cosines: [Float]
    private let sines: [Float]

    init(manifest: RumbleAsrModel.Manifest, filters: [Float]) throws {
        fftLength = manifest.fftLength
        hop = manifest.hopLength
        bins = manifest.melBins
        frequencies = manifest.fftLength / 2 + 1
        guard filters.count == frequencies * bins else { throw RumbleAsrModel.Failure.unexpectedLayout("mel filters") }
        self.filters = filters
        window = (0..<fftLength).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(manifest.fftLength)) }
        var cosines = [Float](repeating: 0, count: fftLength * frequencies)
        var sines = [Float](repeating: 0, count: fftLength * frequencies)
        for n in 0..<fftLength {
            for k in 0..<frequencies {
                let angle = 2 * Double.pi * Double(k * n % manifest.fftLength) / Double(manifest.fftLength)
                cosines[n * frequencies + k] = Float(cos(angle))
                sines[n * frequencies + k] = Float(sin(angle))
            }
        }
        self.cosines = cosines
        self.sines = sines
    }

    func compute(_ samples: [Float]) -> (values: [Float], frames: Int) {
        let frames = (samples.count + hop - 1) / hop
        let half = fftLength / 2
        var padded = [Float](repeating: 0, count: half + samples.count + fftLength + hop)
        for index in 0..<half where index + 1 < samples.count { padded[half - 1 - index] = samples[index + 1] }
        padded.replaceSubrange(half..<(half + samples.count), with: samples)
        var windowed = [Float](repeating: 0, count: frames * fftLength)
        for frame in 0..<frames {
            for n in 0..<fftLength { windowed[frame * fftLength + n] = padded[frame * hop + n] * window[n] }
        }
        var real = [Float](repeating: 0, count: frames * frequencies)
        var imaginary = [Float](repeating: 0, count: frames * frequencies)
        vDSP_mmul(windowed, 1, cosines, 1, &real, 1, vDSP_Length(frames), vDSP_Length(frequencies), vDSP_Length(fftLength))
        vDSP_mmul(windowed, 1, sines, 1, &imaginary, 1, vDSP_Length(frames), vDSP_Length(frequencies), vDSP_Length(fftLength))
        var power = [Float](repeating: 0, count: frames * frequencies)
        vDSP_vsq(real, 1, &power, 1, vDSP_Length(power.count))
        vDSP_vma(imaginary, 1, imaginary, 1, power, 1, &power, 1, vDSP_Length(power.count))
        var melFrames = [Float](repeating: 0, count: frames * bins)
        vDSP_mmul(power, 1, filters, 1, &melFrames, 1, vDSP_Length(frames), vDSP_Length(bins), vDSP_Length(frequencies))
        var floor: Float = 1e-10
        var ceiling = Float.greatestFiniteMagnitude
        vDSP_vclip(melFrames, 1, &floor, &ceiling, &melFrames, 1, vDSP_Length(melFrames.count))
        var count = Int32(melFrames.count)
        vvlog10f(&melFrames, melFrames, &count)
        var peak: Float = 0
        vDSP_maxv(melFrames, 1, &peak, vDSP_Length(melFrames.count))
        var lowest = peak - 8
        vDSP_vclip(melFrames, 1, &lowest, &ceiling, &melFrames, 1, vDSP_Length(melFrames.count))
        var values = [Float](repeating: 0, count: bins * frames)
        for frame in 0..<frames {
            for bin in 0..<bins { values[bin * frames + frame] = (melFrames[frame * bins + bin] + 4) / 4 }
        }
        return (values, frames)
    }
}
#endif
