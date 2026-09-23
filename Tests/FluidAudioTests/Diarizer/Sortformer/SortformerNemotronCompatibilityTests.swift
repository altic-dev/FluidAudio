import CoreML
import Foundation
import XCTest
@testable import FluidAudio

final class SortformerNemotronCompatibilityTests: XCTestCase {
    private func configuration() throws -> SortformerConfig {
        // Explicit cache parameters for contract testing, not a claim of NeMo parity.
        try .nemotron(spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: 3, predScoreThreshold: 0.25)
    }

    private func geometry(
        chunkShape: [Int] = [1, 3040, 128],
        fifoShape: [Int] = [1, 40, 512],
        lengthShape: [Int] = [1],
        lengthType: MLMultiArrayDataType = .int32,
        predictionType: MLMultiArrayDataType = .float16
    ) -> SortformerModelGeometry {
        SortformerModelGeometry(
            chunk: .init(name: "chunk", shape: chunkShape, dataType: .float32),
            chunkLengths: .init(name: "chunk_lengths", shape: lengthShape, dataType: lengthType),
            spkcache: .init(name: "spkcache", shape: [1, 264, 512], dataType: .float32),
            spkcacheLengths: .init(name: "spkcache_lengths", shape: [1], dataType: .int32),
            fifo: .init(name: "fifo", shape: fifoShape, dataType: .float32),
            fifoLengths: .init(name: "fifo_lengths", shape: [1], dataType: .int32),
            predictions: .init(name: "spkcache_fifo_chunk_preds", shape: [1, 684, 8], dataType: predictionType),
            chunkEmbeddings: .init(name: "chunk_pre_encode_embs", shape: [1, 380, 512], dataType: .float16),
            chunkEmbeddingLengths: .init(name: "chunk_pre_encode_lengths", shape: [1], dataType: .int32)
        )
    }

    /// Runs one first-chunk update whose frames are all silent, which overflows the FIFO and pops
    /// frames through the silence-profile update.
    private func silentFirstUpdate(config: SortformerConfig) throws -> SortformerStreamingState {
        var state = SortformerStreamingState(config: config)
        _ = try SortformerStateUpdater(config: config).streamingUpdate(
            state: &state,
            chunk: [Float](repeating: 1, count: config.chunkEncoderFrames * config.preEncoderDims),
            preds: [Float](repeating: 0, count: config.chunkEncoderFrames * config.numSpeakers),
            leftContext: config.chunkLeftContext,
            rightContext: config.chunkRightContext
        )
        return state
    }

    func testLearnedSilenceEmbeddingIsUsedAndNeverUpdated() throws {
        let learned = (0..<512).map { Float($0) / 512 }
        let config = try SortformerConfig.nemotron(
            spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: 1, predScoreThreshold: 0.25,
            learnedSilenceEmbedding: learned
        )
        XCTAssertEqual(SortformerStreamingState(config: config).meanSilenceEmbedding, learned)

        let state = try self.silentFirstUpdate(config: config)

        XCTAssertEqual(state.meanSilenceEmbedding, learned)
        XCTAssertEqual(state.silenceFrameCount, 0)
    }

    func testRunningMeanSilenceStillLearnsWithoutLearnedEmbedding() throws {
        let config = try SortformerConfig.nemotron(
            spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: 1, predScoreThreshold: 0.25
        )
        XCTAssertNil(config.learnedSilenceEmbedding)

        let state = try self.silentFirstUpdate(config: config)

        XCTAssertGreaterThan(state.silenceFrameCount, 0)
        XCTAssertEqual(state.meanSilenceEmbedding, [Float](repeating: 1, count: 512))
    }

    func testLearnedSilenceEmbeddingMustMatchEmbeddingWidth() {
        XCTAssertThrowsError(
            try SortformerConfig.nemotron(
                spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: 1, predScoreThreshold: 0.25,
                learnedSilenceEmbedding: [0, 0, 0]
            )
        )
        XCTAssertThrowsError(
            try SortformerConfig.nemotron(
                spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: 1, predScoreThreshold: 0.25,
                learnedSilenceEmbedding: [Float](repeating: .nan, count: 512)
            )
        )
    }

    func testExplicitNemotronCadenceAndLegacyDefaults() throws {
        let config = try configuration()
        XCTAssertEqual(config.melFrontend.family, .nemotron3Diarization)
        XCTAssertEqual(SortformerConfig.default.melFrontend.family, .streamingSortformer)
        XCTAssertEqual(config.numSpeakers, 8)
        XCTAssertEqual(config.chunkMelFrames, 3040)
        XCTAssertEqual(config.predictionFrames, 684)
        XCTAssertEqual(config.spkcacheUpdatePeriod, 300)
        XCTAssertNil(config.modelVariant)
        XCTAssertEqual(SortformerConfig.default.numSpeakers, 4)
        XCTAssertEqual(SortformerConfig(chunkLen: 340, spkcacheUpdatePeriod: 300).spkcacheUpdatePeriod, 340)
        XCTAssertThrowsError(try geometry().validate(against: .default))
        XCTAssertNoThrow(try geometry().validate(against: config))
    }

    func testNemotronFrontendContractIsFrozenToExtractedNeMoConfiguration() throws {
        let frontend = try configuration().melFrontend
        XCTAssertEqual(frontend.sampleRate, 16_000)
        XCTAssertEqual(frontend.featureCount, 128)
        XCTAssertEqual(frontend.fftLength, 512)
        XCTAssertEqual(frontend.windowLength, 400)
        XCTAssertEqual(frontend.hopLength, 160)
        XCTAssertEqual(frontend.preemphasis, 0.97)
        XCTAssertEqual(frontend.dither, 0)
        XCTAssertEqual(frontend.padding, .centeredConstant)
        XCTAssertEqual(frontend.window, .symmetricHann)
        XCTAssertEqual(frontend.melScale, .slaneyNormalized)
        XCTAssertEqual(frontend.magnitudePower, 2)
        XCTAssertEqual(frontend.logGuard, .additive)
        XCTAssertEqual(frontend.logGuardValue, powf(2, -24))
        XCTAssertEqual(frontend.normalization, .none)
        XCTAssertEqual(frontend.padTo, 0)
        XCTAssertEqual(frontend.padValue, 0)
        XCTAssertEqual(frontend.validFrameCount(sampleCount: 16_000), 100)
        XCTAssertEqual(frontend.storageFrameCount(sampleCount: 16_000), 101)
        // The older `(L + nFFT - windowLength) / hop` approximation returns 102 here.
        XCTAssertEqual(frontend.validFrameCount(sampleCount: 16_100), 100)
        XCTAssertEqual(frontend.storageFrameCount(sampleCount: 16_100), 101)
    }

    func testOfflineFeatureLoaderEmitsRightContextAgainAsFinalCoreTail() throws {
        let config = try configuration()
        // Exactly one fixed model window: 3040 valid 10 ms mel frames. NeMo chunks by the
        // 2720-frame core, so the final 320 frames must be emitted a second time as core after
        // serving as the first call's right context.
        let audio = [Float](repeating: 0.125, count: config.chunkMelFrames * config.melStride)
        var loader = SortformerFeatureLoader(config: config, audio: audio)

        XCTAssertEqual(loader.numChunks, 2)
        let first = try XCTUnwrap(loader.next())
        XCTAssertEqual(first.chunkLength, config.chunkMelFrames)
        XCTAssertEqual(first.leftOffset, 0)
        XCTAssertEqual(first.rightOffset, config.chunkRightContext * config.subsamplingFactor)
        XCTAssertEqual(first.chunkFeatures.count, config.chunkMelFrames * config.melFeatures)

        let tail = try XCTUnwrap(loader.next())
        XCTAssertEqual(tail.chunkLength, config.chunkRightContext * config.subsamplingFactor)
        XCTAssertEqual(tail.leftOffset, 0)
        XCTAssertEqual(tail.rightOffset, 0)
        XCTAssertEqual(tail.chunkFeatures.count, tail.chunkLength * config.melFeatures)
        XCTAssertNil(loader.next())
    }

    func testNemotronStateCadenceLeavesExactlyFifoCapacity() throws {
        let config = try configuration()
        let updater = SortformerStateUpdater(config: config)
        var state = SortformerStreamingState(config: config)
        let chunk = [Float](repeating: 0.25, count: config.chunkEncoderFrames * config.preEncoderDims)
        let predictions = [Float](repeating: 0.1, count: config.predictionFrames * config.numSpeakers)

        let first = try updater.streamingUpdate(
            state: &state,
            chunk: chunk,
            preds: predictions,
            leftContext: config.chunkLeftContext,
            rightContext: config.chunkRightContext
        )
        XCTAssertEqual(first.confirmedFrameCount, config.chunkLen)
        XCTAssertEqual(first.tentativeFrameCount, config.chunkRightContext)
        XCTAssertEqual(state.spkcacheLength, config.spkcacheLen)
        XCTAssertEqual(state.fifoLength, config.fifoLen)

        _ = try updater.streamingUpdate(
            state: &state,
            chunk: chunk,
            preds: predictions,
            leftContext: config.chunkLeftContext,
            rightContext: config.chunkRightContext
        )
        XCTAssertEqual(state.spkcacheLength, config.spkcacheLen)
        XCTAssertEqual(state.fifoLength, config.fifoLen)
        XCTAssertEqual(state.spkcache.count, config.spkcacheLen * config.preEncoderDims)
        XCTAssertEqual(state.fifo.count, config.fifoLen * config.preEncoderDims)
    }

    func testPartialEncodedTailUsesReportedLengthNotFixedCapacity() throws {
        let config = try configuration()
        let updater = SortformerStateUpdater(config: config)
        var state = SortformerStreamingState(config: config)
        let partialFrames = 5

        let result = try updater.streamingUpdate(
            state: &state,
            chunk: [Float](repeating: 1, count: partialFrames * config.preEncoderDims),
            preds: [Float](repeating: 0.2, count: partialFrames * config.numSpeakers),
            leftContext: 0,
            rightContext: 0
        )

        XCTAssertEqual(result.confirmedFrameCount, partialFrames)
        XCTAssertEqual(result.tentativeFrameCount, 0)
        XCTAssertEqual(state.fifoLength, partialFrames)
        XCTAssertEqual(state.fifo.count, partialFrames * config.preEncoderDims)
    }

    func testModelGeometryRejectsInvalidRanksAndDisagreeingWidths() throws {
        let config = try configuration()
        for shape in [[3040, 128], [2, 3040, 128], [1, -1, 128], [1, 3040, 64]] {
            XCTAssertThrowsError(try geometry(chunkShape: shape).validate(against: config))
        }
        XCTAssertThrowsError(try geometry(fifoShape: [1, 40, 256]).validate(against: config))
        XCTAssertThrowsError(try geometry(lengthShape: [1, 1]).validate(against: config))
        XCTAssertThrowsError(try geometry(lengthType: .float32).validate(against: config))
        XCTAssertThrowsError(try geometry(predictionType: .int32).validate(against: config))
    }

    func testExtremeGeometryThrowsInsteadOfOverflowing() throws {
        var config = try configuration()
        config.chunkLeftContext = Int.max
        config.chunkRightContext = Int.max
        XCTAssertThrowsError(try config.validateGeometry())
        XCTAssertThrowsError(try geometry().validate(against: config))
        config = try configuration()
        config.spkcacheSilFramesPerSpk = Int.min
        XCTAssertThrowsError(try config.validateGeometry())
        for silenceFrames in [Int.min, -1, 33, Int.max] {
            XCTAssertThrowsError(
                try SortformerConfig.nemotron(
                    spkcacheUpdatePeriod: 300, spkcacheSilFramesPerSpk: silenceFrames, predScoreThreshold: 0.25
                ))
        }
    }

    func testFloat16PredictionValuesAreReadWithoutAnotherSigmoid() throws {
        let values = try MLMultiArray(shape: [1, 3], dataType: .float16)
        values[0] = 0
        values[1] = 0.25
        values[2] = 1
        let provider = try MLDictionaryFeatureProvider(dictionary: ["preds": values])
        let decoded = try XCTUnwrap(SortformerModelGeometry.floatScalars(from: provider, named: "preds"))
        XCTAssertEqual(decoded, [0, 0.25, 1])
        let integers = try MLMultiArray(shape: [1], dataType: .int32)
        XCTAssertNil(SortformerModelGeometry.floatScalars(from: integers))
    }

    func testRealNemotronArtifactLoadsWithSuppliedComputeConfiguration() async throws {
        guard let path = ProcessInfo.processInfo.environment["NEMOTRON_DIARIZATION_MODEL_PATH"] else {
            throw XCTSkip("Set NEMOTRON_DIARIZATION_MODEL_PATH for real-artifact metadata validation")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let models = try await SortformerModels.load(
            config: self.configuration(), mainModelPath: URL(fileURLWithPath: path), configuration: configuration
        )
        XCTAssertEqual(models.geometry.numSpeakers, 8)
        XCTAssertEqual(models.geometry.predictions.dataType, .float16)
        XCTAssertEqual(models.geometry.names.predictions, "spkcache_fifo_chunk_preds")
        XCTAssertEqual(models.mainModel.configuration.computeUnits, .cpuOnly)
        XCTAssertThrowsError(try models.geometry.validate(against: .default))
    }

    func testRealNemotronArtifactProcessesFixedWindowAndCoreTail() async throws {
        guard let path = ProcessInfo.processInfo.environment["NEMOTRON_DIARIZATION_MODEL_PATH"],
            let audioPath = ProcessInfo.processInfo.environment["NEMOTRON_DIARIZATION_AUDIO_PATH"]
        else {
            throw XCTSkip(
                "Set NEMOTRON_DIARIZATION_MODEL_PATH and NEMOTRON_DIARIZATION_AUDIO_PATH "
                    + "for real-artifact inference validation"
            )
        }
        let config = try self.configuration()
        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = .cpuOnly
        let models = try await SortformerModels.load(
            config: config,
            mainModelPath: URL(fileURLWithPath: path),
            configuration: modelConfiguration
        )
        let diarizer = SortformerDiarizer(config: config)
        diarizer.initialize(models: models)

        // Use a real speech recording; synthetic tones cannot validate a diarization model. One
        // fixed 3040-frame model window is 380 encoder frames. The first call confirms 340 and
        // uses 40 as right context; the loader must issue those 40 again as the final core tail.
        let source = try AudioConverter(sampleRate: Double(config.sampleRate)).resampleAudioFile(
            URL(fileURLWithPath: audioPath)
        )
        let requiredSamples = config.chunkMelFrames * config.melStride
        guard source.count >= requiredSamples else {
            throw XCTSkip("NEMOTRON_DIARIZATION_AUDIO_PATH must contain at least one model window of speech")
        }
        let samples = Array(source.prefix(requiredSamples))
        let timeline = try diarizer.processComplete(samples)

        XCTAssertEqual(timeline.numFinalizedFrames, config.chunkLen + config.chunkRightContext)
        XCTAssertEqual(timeline.numTentativeFrames, 0)
        XCTAssertEqual(diarizer.state.spkcacheLength, config.spkcacheLen)
        XCTAssertEqual(diarizer.state.fifoLength, 0)
    }
}
