#if arch(arm64)
import Foundation

public final class PalettizedEmbedding: @unchecked Sendable {
    public enum Failure: Error {
        case unexpectedLayout
    }

    public let width: Int
    private let weights: Data
    private let indicesOffset: Int
    private let lutOffset: Int
    private let bits: Int

    public init(tiedHeadOf model: URL, rows: Int) throws {
        let program = try String(contentsOf: model.appendingPathComponent("model.mil"), encoding: .utf8)
        let blob = #"BLOBFILE\(path = string\("@model_path/weights/weight.bin"\), offset = uint64\((\d+)\)\)"#
        let pattern = try NSRegularExpression(
            pattern: "tensor<uint(\\d+), \\[\(rows), (\\d+)\\]>\\(\(blob)\\), lut = tensor<fp16, \\[\(rows / 16), 1, \\d+, 1\\]>\\(\(blob)\\)"
        )
        let capture = { (match: NSTextCheckingResult, index: Int) in Int(program[Range(match.range(at: index), in: program)!]) }
        guard let match = pattern.firstMatch(in: program, range: NSRange(program.startIndex..., in: program)),
              let bits = capture(match, 1), (1 ... 8).contains(bits), let width = capture(match, 2),
              let indexHeader = capture(match, 3), let lutHeader = capture(match, 4)
        else { throw Failure.unexpectedLayout }
        let mapped = try Data(contentsOf: model.appendingPathComponent("weights/weight.bin"), options: .alwaysMapped)
        let blobDataOffsetField = 16
        let dataOffset = { (header: Int) in
            mapped.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: header + blobDataOffsetField, as: UInt64.self)) }
        }
        weights = mapped
        indicesOffset = dataOffset(indexHeader)
        lutOffset = dataOffset(lutHeader)
        self.width = width
        self.bits = bits
        guard width * bits % 8 == 0, indicesOffset + rows * width * bits / 8 <= mapped.count,
              lutOffset + rows / 16 * (2 << bits) <= mapped.count
        else { throw Failure.unexpectedLayout }
    }

    public func write(_ token: Int32, to destination: UnsafeMutablePointer<Float16>) {
        let row = Int(token)
        let rowBytes = width * bits / 8
        let mask = UInt32(1 << bits - 1)
        weights.withUnsafeBytes { raw in
            let lut = (raw.baseAddress! + lutOffset + (row / 16) * (2 << bits)).assumingMemoryBound(to: Float16.self)
            let packed = (raw.baseAddress! + indicesOffset + row * rowBytes).assumingMemoryBound(to: UInt8.self)
            for column in 0 ..< width {
                let bit = column * bits, byte = bit / 8
                let word = UInt32(packed[byte]) | (byte + 1 < rowBytes ? UInt32(packed[byte + 1]) << 8 : 0)
                destination[column] = lut[Int((word >> UInt32(bit % 8)) & mask)]
            }
        }
    }
}
#endif
