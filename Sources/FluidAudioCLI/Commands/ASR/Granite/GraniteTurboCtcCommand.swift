#if os(macOS)
@preconcurrency import CoreML
import FluidAudio
import Foundation

private struct GraniteTurboCtcOptions {
    let audioFile: String
    var modelDir: String?
    var referenceJSON: String?
    var computeUnits: MLComputeUnits = .cpuAndGPU
    /// Route through AsrModels/AsrManager — the exact path FluidVoice uses.
    var unified = false
    /// Use the non-commercial checkpoint's cache (research/benchmarking only).
    var nonCommercial = false
}

private struct GraniteTurboCtcReference: Decodable {
    let samples: Int
    let tokenIDs: [Int]

    private enum CodingKeys: String, CodingKey {
        case samples
        case tokenIDs = "token_ids"
    }
}

enum GraniteTurboCtcCommand {
    private static let logger = AppLogger(category: "GraniteTurboCtc")

    static func run(arguments: [String]) async {
        guard let options = parseOptions(arguments) else { return }
        await transcribe(options: options)
    }

    private static func parseOptions(_ arguments: [String]) -> GraniteTurboCtcOptions? {
        if arguments.isEmpty || arguments.first == "--help" || arguments.first == "-h" {
            printUsage()
            return nil
        }

        var options = GraniteTurboCtcOptions(audioFile: arguments[0])
        var index = 1
        while index < arguments.count {
            switch arguments[index] {
            case "--model-dir":
                options.modelDir = nextValue(arguments, at: &index, option: "--model-dir")
            case "--reference-json":
                options.referenceJSON = nextValue(arguments, at: &index, option: "--reference-json")
            case "--unified":
                options.unified = true
            case "--nc":
                options.nonCommercial = true
            case "--compute-units":
                guard let value = nextValue(arguments, at: &index, option: "--compute-units"),
                    let parsed = parseComputeUnits(value)
                else {
                    logger.error("Unknown compute units")
                    exit(1)
                }
                options.computeUnits = parsed
            default:
                logger.warning("Unknown option: \(arguments[index])")
            }
            index += 1
        }

        guard options.modelDir != nil || options.unified else {
            logger.error("--model-dir is required (no hosted bundle yet), or pass --unified to use the cache")
            return nil
        }
        return options
    }

    private static func nextValue(_ arguments: [String], at index: inout Int, option: String) -> String? {
        guard index + 1 < arguments.count else {
            logger.error("Missing value for \(option)")
            return nil
        }
        index += 1
        return arguments[index]
    }

    private static func transcribe(options: GraniteTurboCtcOptions) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Granite TurboCTC requires macOS 14 or later")
            return
        }

        if options.unified {
            await transcribeUnified(options: options)
            return
        }

        do {
            let directory = URL(fileURLWithPath: options.modelDir!)
            let manager = GraniteTurboCtcManager()
            try await manager.loadModels(from: directory, computeUnits: options.computeUnits)

            var samples = try AudioConverter().resampleAudioFile(path: options.audioFile)
            let reference = try loadReference(options.referenceJSON)
            if let reference, samples.count > reference.samples {
                // The fixture covers a prefix of the file; match it exactly.
                samples = Array(samples[0..<reference.samples])
            }

            let result = try await manager.transcribeDetailed(audioSamples: samples)

            logger.info(String(repeating: "=", count: 50))
            logger.info("GRANITE TURBOCTC")
            logger.info(String(repeating: "=", count: 50))
            print(result.text)
            logger.info("")
            print("  Audio duration: \(String(format: "%.2f", result.durationSeconds))s")
            print("  Processing time: \(String(format: "%.3f", result.elapsedSeconds))s")
            print("  RTFx: \(String(format: "%.1f", result.realTimeFactorX))x")
            print("  Windows: \(result.windowCount)")
            print("  Tokens: \(result.tokenIDs.count)")

            if let reference {
                printParity(hypothesis: result.tokenIDs, reference: reference.tokenIDs)
            }
        } catch {
            logger.error("Granite TurboCTC failed: \(error)")
        }
    }

    /// Drives the same API surface FluidVoice's FluidAudioProvider uses:
    /// AsrModels.downloadAndLoad(version:) -> AsrManager.initialize(models:) -> transcribe().
    private static func transcribeUnified(options: GraniteTurboCtcOptions) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Granite TurboCTC requires macOS 14 or later")
            return
        }
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = options.computeUnits
            let version: AsrModelVersion = options.nonCommercial ? .graniteTurboCtcNc : .graniteTurboCtc
            let models: AsrModels
            if let modelDir = options.modelDir {
                models = try await AsrModels.load(
                    from: URL(fileURLWithPath: modelDir),
                    configuration: configuration,
                    version: version
                )
            } else {
                models = try await AsrModels.downloadAndLoad(
                    configuration: configuration,
                    version: version
                )
            }

            let manager = AsrManager(config: .default)
            try await manager.initialize(models: models)

            let samples = try AudioConverter().resampleAudioFile(path: options.audioFile)
            let result = try await manager.transcribe(samples, source: .microphone)

            logger.info(String(repeating: "=", count: 50))
            logger.info("GRANITE TURBOCTC (unified AsrManager path)")
            logger.info(String(repeating: "=", count: 50))
            print(result.text)
            print("  Confidence: \(String(format: "%.3f", result.confidence))")
            print("  Audio duration: \(String(format: "%.2f", result.duration))s")
            print("  Processing time: \(String(format: "%.3f", result.processingTime))s")
            print("  RTFx: \(String(format: "%.1f", result.rtfx))x")
            if let timings = result.tokenTimings, let first = timings.first, let last = timings.last {
                print("  Tokens: \(timings.count)")
                let firstText = "\(first.token) @ \(String(format: "%.2f", first.startTime))s"
                let lastText = "\(last.token) @ \(String(format: "%.2f", last.startTime))s"
                print("  First/last timing: \(firstText) ... \(lastText)")
            }
        } catch {
            logger.error("Unified Granite TurboCTC failed: \(error)")
        }
    }

    private static func loadReference(_ path: String?) throws -> GraniteTurboCtcReference? {
        guard let path else { return nil }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try JSONDecoder().decode(GraniteTurboCtcReference.self, from: data)
    }

    private static func printParity(hypothesis: [Int], reference: [Int]) {
        // The fixture stores raw per-frame argmax; collapse it the same way.
        var collapsed: [Int] = []
        var previous = -1
        for token in reference {
            if token != previous, token != 0 {
                collapsed.append(token)
            }
            previous = token
        }

        logger.info("")
        logger.info("Parity vs PyTorch fixture:")
        logger.info("  reference tokens: \(collapsed.count)")
        logger.info("  swift tokens:     \(hypothesis.count)")
        if hypothesis == collapsed {
            print("  MATCH: token sequences are identical")
        } else {
            let shared = zip(hypothesis, collapsed).prefix { $0 == $1 }.count
            print("  MISMATCH: diverges at token \(shared)")
            let hypSlice = hypothesis.dropFirst(max(0, shared - 3)).prefix(8)
            let refSlice = collapsed.dropFirst(max(0, shared - 3)).prefix(8)
            logger.error("    swift:     \(Array(hypSlice))")
            logger.error("    reference: \(Array(refSlice))")
        }
    }

    private static func parseComputeUnits(_ value: String) -> MLComputeUnits? {
        switch value.lowercased() {
        case "cpu", "cpu-only", "cpu_only": return .cpuOnly
        case "gpu", "cpu-and-gpu", "cpu_and_gpu": return .cpuAndGPU
        case "ane", "ne", "cpu-and-neural-engine": return .cpuAndNeuralEngine
        case "all": return .all
        default: return nil
        }
    }

    private static func printUsage() {
        print(
            """

            Granite TurboCTC Transcribe Command

            Usage: fluidaudio granite-turboctc <audio_file> --model-dir <path> [options]

            Options:
                --help, -h                  Show this help message
                --model-dir <path>          Directory holding manifest.json, the .mlpackage,
                                            mel_filters.bin and processor/ (required)
                --reference-json <path>     reference.json from export-assets.py; checks
                                            Swift token output against the PyTorch fixture
                --compute-units <cpu|gpu|ane|all>
                                            Default: gpu

            Example:
                fluidaudio granite-turboctc audio.wav \\
                    --model-dir Scripts/granite-turboctc/coreml/build \\
                    --reference-json Scripts/granite-turboctc/coreml/build/reference.json
            """
        )
    }
}
#endif
