import AVFoundation
import XCTest

@testable import FluidAudio

final class StreamingAudioSourceFactoryTests: XCTestCase {
    func testStereoConversionIncludesRightChannel() throws {
        let samples = try self.convertFixture(channelCount: 2) { channels, frame in
            channels[0][frame] = 0
            channels[1][frame] = 1
        }

        XCTAssertEqual(samples.count, 16_000, accuracy: 2)
        let mean = samples.reduce(0, +) / Float(samples.count)
        XCTAssertEqual(mean, 0.5, accuracy: 0.01)
        XCTAssertGreaterThan(samples.map(abs).max() ?? 0, 0.4)
    }

    func testSixChannelConversionIncludesEveryChannel() throws {
        let samples = try self.convertFixture(channelCount: 6) { channels, frame in
            for channel in 0..<6 {
                channels[channel][frame] = channel == 5 ? 1 : 0
            }
        }

        let mean = samples.reduce(0, +) / Float(samples.count)
        XCTAssertGreaterThan(mean, 0.1)
        XCTAssertGreaterThan(samples.map(abs).max() ?? 0, 0.1)
    }

    private func convertFixture(
        channelCount: AVAudioChannelCount,
        fill: (UnsafePointer<UnsafeMutablePointer<Float>>, Int) -> Void
    ) throws -> [Float] {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fluid-audio-stereo-\(UUID().uuidString).caf")
        defer { try? FileManager.default.removeItem(at: url) }

        let format: AVAudioFormat
        if channelCount > 2 {
            let layout = try XCTUnwrap(AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_MPEG_5_1_A))
            format = try XCTUnwrap(
                AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: 48_000,
                    interleaved: false,
                    channelLayout: layout
                ))
        } else {
            format = try XCTUnwrap(
                AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: 48_000,
                    channels: channelCount,
                    interleaved: false
                ))
        }
        let frameCount: AVAudioFrameCount = 48_000
        let buffer = try XCTUnwrap(
            AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: frameCount
            ))
        buffer.frameLength = frameCount
        let channels = try XCTUnwrap(buffer.floatChannelData)
        for frame in 0..<Int(frameCount) {
            fill(channels, frame)
        }

        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
        }

        let result = try StreamingAudioSourceFactory().makeDiskBackedSource(
            from: url,
            targetSampleRate: 16_000
        )
        defer { result.source.cleanup() }

        var samples = Array(repeating: Float.zero, count: result.source.sampleCount)
        try samples.withUnsafeMutableBufferPointer { pointer in
            try result.source.copySamples(
                into: try XCTUnwrap(pointer.baseAddress),
                offset: 0,
                count: pointer.count
            )
        }

        return samples
    }
}
