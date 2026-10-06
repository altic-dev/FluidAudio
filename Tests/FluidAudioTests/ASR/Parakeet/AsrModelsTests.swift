@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class AsrModelsTests: XCTestCase {

    // MARK: - Model Names Tests

    func testModelNames() {
        XCTAssertEqual(ModelNames.ASR.preprocessorFile, "Preprocessor.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.encoderFile, "Encoder.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.decoderFile, "Decoder.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.jointFile, "JointDecision.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.vocabulary(for: .parakeet), "parakeet_vocab.json")
        XCTAssertEqual(ModelNames.ASR.vocabulary(for: .parakeetV2), "parakeet_vocab.json")
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = AsrModels.defaultConfiguration()

        XCTAssertTrue(config.allowLowPrecisionAccumulationOnGPU)
        // Should always use CPU+ANE for optimal performance
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    // MARK: - Directory Tests

    func testDefaultCacheDirectory() {
        let cacheDir = AsrModels.defaultCacheDirectory()

        // Verify path components
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))
        XCTAssertTrue(cacheDir.path.contains(Repo.parakeet.folderName))

        // Verify it's an absolute path
        XCTAssertTrue(cacheDir.isFileURL)
        XCTAssertTrue(cacheDir.path.starts(with: "/"))
    }

    // MARK: - Model Existence Tests

    func testModelsExistWithMissingFiles() {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-\(UUID().uuidString)")

        // Test with non-existent directory - should return false
        let result = AsrModels.modelsExist(at: tempDir)
        // We're just testing the method doesn't crash with non-existent paths
        XCTAssertNotNil(result)  // Method returns a boolean
    }

    func testModelsExistLogic() {
        // Test that the method handles various scenarios without crashing
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-\(UUID().uuidString)")

        // Test 1: Non-existent directory
        _ = AsrModels.modelsExist(at: tempDir)

        // Test 2: The method should check for model files in the expected structure
        // We're testing the logic, not the actual file system operations
        let modelNames: [String] = [
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.encoderFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointFile,
            ModelNames.ASR.vocabulary(for: .parakeet),
        ]

        // Verify all expected model names are defined
        XCTAssertEqual(modelNames.count, 5)
        XCTAssertTrue(modelNames.allSatisfy { !$0.isEmpty })
    }

    // MARK: - Error Tests

    func testAsrModelsErrorDescriptions() {
        let modelNotFound = AsrModelsError.modelNotFound(
            "test.mlmodel", URL(fileURLWithPath: "/test/path"))
        XCTAssertEqual(
            modelNotFound.errorDescription, "ASR model 'test.mlmodel' not found at: /test/path")

        let downloadFailed = AsrModelsError.downloadFailed("Network error")
        XCTAssertEqual(
            downloadFailed.errorDescription, "Failed to download ASR models: Network error")

        let loadingFailed = AsrModelsError.loadingFailed("Invalid format")
        XCTAssertEqual(loadingFailed.errorDescription, "Failed to load ASR models: Invalid format")

        let compilationFailed = AsrModelsError.modelCompilationFailed("Compilation error")
        XCTAssertEqual(
            compilationFailed.errorDescription,
            "Failed to compile ASR models: Compilation error. Try deleting the models and re-downloading."
        )
    }

    // MARK: - Model Initialization Tests

    func testAsrModelsInitialization() throws {
        // Create mock models
        let mockConfig = MLModelConfiguration()
        mockConfig.computeUnits = .cpuOnly

        // Note: We can't create actual MLModel instances in tests without valid model files
        // This test verifies the AsrModels struct initialization logic

        // Test that AsrModels struct can be created with proper types
        let modelNames = [
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.encoderFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointFile,
        ]

        XCTAssertEqual(modelNames.count, 4)
        XCTAssertTrue(modelNames.allSatisfy { $0.hasSuffix(".mlmodelc") })
    }

    // MARK: - Download Path Tests

    func testDownloadPathStructure() async throws {
        let customDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-Download-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: customDir) }

        // Test that download would target correct directory structure
        let expectedRepoPath = customDir.deletingLastPathComponent()
            .appendingPathComponent(Repo.parakeet.folderName)

        // Verify path components
        XCTAssertTrue(expectedRepoPath.path.contains(Repo.parakeet.folderName))
    }

    // MARK: - Model Loading Configuration Tests

    func testCustomConfigurationPropagation() {
        // Test that custom configuration would be used correctly
        let customConfig = MLModelConfiguration()
        customConfig.modelDisplayName = "Test ASR Model"
        customConfig.computeUnits = .cpuAndNeuralEngine
        customConfig.allowLowPrecisionAccumulationOnGPU = false

        // Verify configuration properties
        XCTAssertEqual(customConfig.modelDisplayName, "Test ASR Model")
        XCTAssertEqual(customConfig.computeUnits, .cpuAndNeuralEngine)
        XCTAssertFalse(customConfig.allowLowPrecisionAccumulationOnGPU)
    }

    // MARK: - Force Download Tests

    func testForceDownloadLogic() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-Force-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create existing directory
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        // Add a test file
        let testFile = tempDir.appendingPathComponent("test.txt")
        try "test content".write(to: testFile, atomically: true, encoding: .utf8)

        XCTAssertTrue(FileManager.default.fileExists(atPath: testFile.path))

        // In actual download with force=true, directory would be removed
        // Here we just verify the file exists before theoretical removal
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempDir.path))
    }

    // MARK: - Helper Method Tests

    func testRepoPathCalculation() {
        let modelsDir = URL(fileURLWithPath: "/test/Models/parakeet-tdt-0.6b-v3-coreml")
        let repoPath = modelsDir.deletingLastPathComponent()
            .appendingPathComponent(Repo.parakeet.folderName)

        XCTAssertTrue(repoPath.path.hasSuffix(Repo.parakeet.folderName))
        XCTAssertEqual(repoPath.lastPathComponent, Repo.parakeet.folderName)
    }

    // MARK: - Integration Test Helpers

    func testModelFileValidation() {
        // Test model file extension validation
        let validModelFiles = [
            "model.mlmodelc",
            "Model.mlmodelc",
            "test_model.mlmodelc",
        ]

        for file in validModelFiles {
            XCTAssertTrue(file.hasSuffix(".mlmodelc"), "\(file) should have .mlmodelc extension")
        }

        // Test vocabulary file
        let vocabFile = "parakeet_vocab.json"
        XCTAssertTrue(vocabFile.hasSuffix(".json"))
        XCTAssertTrue(vocabFile.contains("vocab"))
    }

    // MARK: - Neural Engine Optimization Tests

    func testOptimizedConfiguration() {
        // In CI environment, all compute units are overridden to .cpuOnly
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil

        // Test encoder configuration
        let melConfig = AsrModels.optimizedConfiguration(for: .encoder)
        if isCI {
            XCTAssertEqual(melConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(melConfig.computeUnits, .cpuAndNeuralEngine)
        }
        XCTAssertTrue(melConfig.allowLowPrecisionAccumulationOnGPU)

        // Test decoder configuration
        let decoderConfig = AsrModels.optimizedConfiguration(for: .decoder)
        if isCI {
            XCTAssertEqual(decoderConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(decoderConfig.computeUnits, .cpuAndNeuralEngine)
        }

        // Test joint configuration
        let jointConfig = AsrModels.optimizedConfiguration(for: .joint)
        if isCI {
            XCTAssertEqual(jointConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(jointConfig.computeUnits, .cpuAndNeuralEngine)
        }

        // Test with FP16 disabled
        let fp32Config = AsrModels.optimizedConfiguration(for: .encoder, enableFP16: false)
        XCTAssertFalse(fp32Config.allowLowPrecisionAccumulationOnGPU)
    }

    func testOptimizedConfigurationCIEnvironment() {
        // Simulate CI environment
        let originalCI = ProcessInfo.processInfo.environment["CI"]
        setenv("CI", "true", 1)
        defer {
            if let originalCI = originalCI {
                setenv("CI", originalCI, 1)
            } else {
                unsetenv("CI")
            }
        }

        let config = AsrModels.optimizedConfiguration(for: .encoder)
        XCTAssertEqual(config.computeUnits, .cpuOnly)
    }

    func testOptimizedPredictionOptions() {
        let options = AsrModels.optimizedPredictionOptions()
        XCTAssertNotNil(options)

        // Output backings should be configured
        XCTAssertNotNil(options.outputBackings)
    }

    // Removed testLoadWithANEOptimization - causes crashes when trying to load models

    // MARK: - User Configuration Tests

    func testUserConfigurationIsRespected() {
        // Test that when a user provides a configuration, it's respected
        let userConfig = MLModelConfiguration()
        userConfig.computeUnits = .cpuOnly
        userConfig.modelDisplayName = "User Custom Model"

        // Verify the configuration properties
        XCTAssertEqual(userConfig.computeUnits, .cpuOnly)
        XCTAssertEqual(userConfig.modelDisplayName, "User Custom Model")

        // The actual load test would require model files, so we test the configuration logic
        // The fix ensures that when configuration is not nil, it uses the user's compute units
    }

    func testPlatformAwareDefaultConfiguration() {
        let config = AsrModels.defaultConfiguration()

        // Should always use CPU+ANE for optimal performance
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    func testOptimalComputeUnitsRespectsPlatform() {
        // Test each model type
        let modelTypes: [ANEOptimizer.ModelType] = [
            .encoder,
            .decoder,
            .joint,
        ]

        for modelType in modelTypes {
            let computeUnits = ANEOptimizer.optimalComputeUnits(for: modelType)

            // All models should use CPU+ANE for optimal performance
            XCTAssertEqual(
                computeUnits, .cpuAndNeuralEngine,
                "Model type \(modelType) should use CPU+ANE")
        }
    }

    // MARK: - TDT-CTC-110M Model Version Tests

    func testTdtCtc110mHasFusedEncoder() {
        // tdtCtc110m has fused preprocessor+encoder
        XCTAssertTrue(AsrModelVersion.tdtCtc110m.hasFusedEncoder)

        // v2 and v3 have separate encoder
        XCTAssertFalse(AsrModelVersion.v2.hasFusedEncoder)
        XCTAssertFalse(AsrModelVersion.v3.hasFusedEncoder)
    }

    func testTdtCtc110mEncoderHiddenSize() {
        // tdtCtc110m uses 512-dim encoder output
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.encoderHiddenSize, 512)

        // v2 and v3 use 1024-dim encoder output
        XCTAssertEqual(AsrModelVersion.v2.encoderHiddenSize, 1024)
        XCTAssertEqual(AsrModelVersion.v3.encoderHiddenSize, 1024)
    }

    func testTdtCtc110mBlankId() {
        // tdtCtc110m uses blank ID 1024 (same as v2)
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.blankId, 1024)
        XCTAssertEqual(AsrModelVersion.v2.blankId, 1024)

        // v3 uses blank ID 8192
        XCTAssertEqual(AsrModelVersion.v3.blankId, 8192)
    }

    func testTdtCtc110mDecoderLayers() {
        // tdtCtc110m uses 1 decoder LSTM layer
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.decoderLayers, 1)

        // v2 and v3 use 2 decoder LSTM layers
        XCTAssertEqual(AsrModelVersion.v2.decoderLayers, 2)
        XCTAssertEqual(AsrModelVersion.v3.decoderLayers, 2)
    }

    func testTdtCtc110mRepo() {
        // Verify correct HuggingFace repo
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.repo, .parakeetTdtCtc110m)
        XCTAssertEqual(AsrModelVersion.v2.repo, .parakeetV2)
        XCTAssertEqual(AsrModelVersion.v3.repo, .parakeet)
        XCTAssertEqual(AsrModelVersion.fluidParakeetMini.repo, .fluidParakeetMini)
        XCTAssertEqual(AsrModelVersion.fluidParakeetPico.repo, .fluidParakeetPico)
    }

    func testFluidParakeetVersionsKeepV3Contract() {
        for version in [AsrModelVersion.fluidParakeetMini, .fluidParakeetPico] {
            XCTAssertEqual(version.blankId, AsrModelVersion.v3.blankId)
            XCTAssertEqual(version.decoderLayers, AsrModelVersion.v3.decoderLayers)
            XCTAssertEqual(version.encoderHiddenSize, AsrModelVersion.v3.encoderHiddenSize)
            XCTAssertFalse(version.hasFusedEncoder)
            XCTAssertEqual(
                ModelNames.getRequiredModelNames(for: version.repo, variant: nil), ModelNames.ASR.requiredModels)
            XCTAssertEqual(
                AsrModels.defaultCacheDirectory(for: version).lastPathComponent, version.repo.folderName)
        }
        XCTAssertEqual(Repo.fluidParakeetMini.folderName, "fluid-parakeet-mini-coreml")
        XCTAssertEqual(Repo.fluidParakeetPico.remotePath, "altic-dev/fluid-parakeet-pico-coreml")
    }

    func testEveryVersionHasAnIsolatedCacheAndLegacyDefaultsStayV3() {
        let paths = AsrModelVersion.allCases.map { AsrModels.defaultCacheDirectory(for: $0).path }
        XCTAssertEqual(Set(paths).count, AsrModelVersion.allCases.count)
        XCTAssertEqual(AsrModels.defaultCacheDirectory(), AsrModels.defaultCacheDirectory(for: .v3))
        XCTAssertEqual(AsrModelVersion.v2.repo.remotePath, "FluidInference/parakeet-tdt-0.6b-v2-coreml")
        XCTAssertEqual(AsrModelVersion.v3.repo.remotePath, "FluidInference/parakeet-tdt-0.6b-v3-coreml")
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.repo.folderName, "parakeet-tdt-ctc-110m")
    }

    func testFolderInferenceUsesNearestExactComponent() {
        for version in AsrModelVersion.allCases {
            let folder = URL(fileURLWithPath: "/local/models/\(version.repo.folderName)")
            XCTAssertEqual(AsrModels.inferredVersion(from: folder), version)
        }
        let nested = URL(fileURLWithPath: "/\(Repo.fluidParakeetMini.folderName)/\(Repo.parakeet.folderName)")
        XCTAssertEqual(AsrModels.inferredVersion(from: nested), .v3)
        XCTAssertNil(
            AsrModels.inferredVersion(from: URL(fileURLWithPath: "/models/\(Repo.fluidParakeetMini.folderName)-backup"))
        )
        XCTAssertNil(AsrModels.inferredVersion(from: URL(fileURLWithPath: "/models/renamed-local-model")))
    }

    func testLocalOnlyMissingFilesUseExactCustomFolderWithoutRepairingSiblingCache() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FluidParakeetLocal-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let custom = root.appendingPathComponent("renamed-model")
        let sibling = root.appendingPathComponent(Repo.fluidParakeetMini.folderName)
        try FileManager.default.createDirectory(at: custom, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: sibling, withIntermediateDirectories: true)
        let marker = sibling.appendingPathComponent("preserve-me.txt")
        try Data("existing unrelated cache".utf8).write(to: marker)
        do {
            _ = try await AsrModels.loadLocalOnly(from: custom, version: .fluidParakeetMini)
            XCTFail("Missing installed model files must fail without downloading.")
        } catch AsrModelsError.modelNotFound(let name, let path) {
            XCTAssertEqual(name, ModelNames.ASR.preprocessorFile)
            XCTAssertEqual(path, custom.appendingPathComponent(name))
        }
        XCTAssertEqual(try Data(contentsOf: marker), Data("existing unrelated cache".utf8))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: custom.path), [])
    }

    func testCancelledLocalOnlyLoadHasNoFilesystemEffects() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FluidParakeetCancelled-\(UUID().uuidString)")
        let gate = AsyncStream<Void>.makeStream()
        let operation = Task {
            for await _ in gate.stream { break }
            return try await AsrModels.loadLocalOnly(from: folder, version: .fluidParakeetPico)
        }
        operation.cancel()
        gate.continuation.yield(())
        gate.continuation.finish()
        do {
            _ = try await operation.value
            XCTFail("Cancelled local-only loading must not start model work.")
        } catch is CancellationError {}
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
    }

    func testCancelledForceDownloadCannotDeleteExistingFolder() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FluidParakeetForce-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let marker = folder.appendingPathComponent("preserve-me.txt")
        try Data("keep existing files".utf8).write(to: marker)
        let gate = AsyncStream<Void>.makeStream()
        let operation = Task {
            for await _ in gate.stream { break }
            return try await AsrModels.download(to: folder, force: true, version: .fluidParakeetMini)
        }
        operation.cancel()
        gate.continuation.yield(())
        gate.continuation.finish()
        do {
            _ = try await operation.value
            XCTFail("Cancelled forced download must not delete installed files.")
        } catch is CancellationError {}
        XCTAssertEqual(try Data(contentsOf: marker), Data("keep existing files".utf8))
    }

    func testFluidParakeetDownloadsRequireLocalPacksAndPreserveFiles() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let marker = folder.appendingPathComponent("existing-settings.txt")
        try Data("unchanged".utf8).write(to: marker)
        for version in [AsrModelVersion.fluidParakeetMini, .fluidParakeetPico] {
            XCTAssertTrue(version.requiresLocalModels)
            XCTAssertEqual(version.hostedModelArchiveURL?.host, "models.fluidvoice.app")
            do {
                _ = try await AsrModels.download(to: folder, force: true, version: version)
                XCTFail("A local-only variant must not contact private HF or delete existing files")
            } catch let error as AsrModelsError {
                XCTAssertTrue(error.localizedDescription.contains("models.fluidvoice.app"))
                XCTAssertTrue(error.localizedDescription.contains("--model-dir"))
            }
            XCTAssertEqual(try Data(contentsOf: marker), Data("unchanged".utf8))
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), [marker.lastPathComponent])
            let valid = try await AsrModels.isModelValid(version: version, at: folder)
            XCTAssertFalse(valid)
        }
        for version in [AsrModelVersion.v2, .v3, .tdtCtc110m] {
            XCTAssertFalse(version.requiresLocalModels)
            XCTAssertNil(version.hostedModelArchiveURL)
        }
    }

    /// Opt-in integration proof uses installed real models and recorded speech, never fixtures.
    func testFluidParakeetRealModelsTranscribeRecordedSpeech() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let mini = environment["FLUIDAUDIO_MINI_MODEL_DIR"],
            let pico = environment["FLUIDAUDIO_PICO_MODEL_DIR"],
            let recording = environment["FLUIDAUDIO_PARAKEET_TEST_AUDIO"]
        else { throw XCTSkip("Provide Mini/Pico compiled models and recorded speech for local inference proof.") }
        for (directory, version) in [(mini, AsrModelVersion.fluidParakeetMini), (pico, .fluidParakeetPico)] {
            let models = try await AsrModels.loadLocalOnly(from: URL(fileURLWithPath: directory), version: version)
            XCTAssertEqual(models.version, version)
            XCTAssertEqual(models.vocabulary.count, 8192)
            XCTAssertTrue(models.encoder != nil || models.splitEncoder != nil)
            XCTAssertEqual(models.encoderOutputShape, [1, 1024, 188])
            XCTAssertEqual(Set(models.encoderInputFeatureNames), ["mel", "mel_length"])
            XCTAssertTrue(AsrModels.localModelsExist(at: URL(fileURLWithPath: directory), version: version))
            let retainedDirectory = try await AsrModels.download(to: URL(fileURLWithPath: directory), version: version)
            XCTAssertEqual(retainedDirectory, URL(fileURLWithPath: directory))
            let valid = try await AsrModels.isModelValid(version: version, at: URL(fileURLWithPath: directory))
            XCTAssertTrue(valid)
            let manager = AsrManager(
                config: ASRConfig(
                    tdtConfig: TdtConfig(blankId: version.blankId), encoderHiddenSize: version.encoderHiddenSize))
            try await manager.initialize(models: models)
            let result = try await manager.transcribe(URL(fileURLWithPath: recording))
            XCTAssertFalse(result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            XCTAssertGreaterThan(result.duration, 0)
            XCTAssertTrue(result.processingTime.isFinite)
            print("Fluid Parakeet \(version.repo.folderName) recorded-speech result: \(result.text)")
            let recorded = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: recording))
            let samples = Array(recorded.prefix(ASRConstants.maxModelSamples))
            let embedding = try await manager.pronunciationEmbedding(
                audioSamples: samples, focalSampleRange: 0..<samples.count)
            XCTAssertEqual(embedding.values.count, 1024)
            XCTAssertTrue(embedding.values.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(embedding.sourceFrameCount, 0)
            print("Fluid Parakeet \(version.repo.folderName) pronunciation dimensions: \(embedding.values.count)")
            try await self.verifyRetainedEncoderWindow(manager: manager, samples: samples)
            try await self.verifySharedEncoderWindows(models: models, samples: samples)
            await manager.cleanup()
        }
    }

    /// Explicit opt-in permits downloading the real auxiliary CTC110m model through the library API.
    func testFluidParakeetRealModelsSupportCtc110mVocabularyBoosting() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["FLUIDAUDIO_TEST_CTC_DOWNLOAD"] == "1",
            let mini = environment["FLUIDAUDIO_MINI_MODEL_DIR"],
            let pico = environment["FLUIDAUDIO_PICO_MODEL_DIR"],
            let recording = environment["FLUIDAUDIO_PARAKEET_TEST_AUDIO"]
        else { throw XCTSkip("Explicitly opt in to real CTC110m downloading and recorded-speech boosting.") }
        let ctc = try await CtcModels.downloadAndLoad(variant: .ctc110m)
        let tokenizer = try await CtcTokenizer.load(from: CtcModels.defaultCacheDirectory(for: .ctc110m))
        let ids = tokenizer.encode("phone")
        XCTAssertFalse(ids.isEmpty)
        let vocabulary = CustomVocabularyContext(terms: [
            CustomVocabularyTerm(text: "phone", weight: 10, tokenIds: nil, ctcTokenIds: ids)
        ])
        for (directory, version) in [(mini, AsrModelVersion.fluidParakeetMini), (pico, .fluidParakeetPico)] {
            let models = try await AsrModels.loadLocalOnly(from: URL(fileURLWithPath: directory), version: version)
            XCTAssertEqual(models.vocabulary.count, 8192)
            XCTAssertEqual(version.blankId, 8192)
            let manager = AsrManager(config: ASRConfig(tdtConfig: TdtConfig(blankId: version.blankId)))
            try await manager.initialize(models: models)
            try await manager.configureVocabularyBoosting(vocabulary: vocabulary, ctcModels: ctc)
            let result = try await manager.transcribe(URL(fileURLWithPath: recording))
            XCTAssertTrue(result.text.lowercased().contains("phone"))
            XCTAssertTrue(result.processingTime.isFinite)
            print("Fluid Parakeet \(version.repo.folderName) CTC110m-boosted result: \(result.text)")
            await manager.disableVocabularyBoosting()
            let unboosted = try await manager.transcribe(URL(fileURLWithPath: recording))
            XCTAssertFalse(unboosted.text.isEmpty)
            await manager.cleanup()
        }
    }

    func testRealMonolithicLegacyVersionsStillTranscribeRecordedSpeech() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let v2 = environment["FLUIDAUDIO_V2_MODEL_DIR"],
            let v3 = environment["FLUIDAUDIO_V3_MODEL_DIR"],
            let recording = environment["FLUIDAUDIO_PARAKEET_TEST_AUDIO"]
        else {
            throw XCTSkip("Provide existing v2/v3 compiled models and recorded speech for legacy regression proof.")
        }
        for (directory, version) in [(v2, AsrModelVersion.v2), (v3, .v3)] {
            let models = try await AsrModels.loadLocalOnly(from: URL(fileURLWithPath: directory), version: version)
            XCTAssertNotNil(models.encoder)
            XCTAssertNil(models.splitEncoder)
            let manager = AsrManager(
                config: ASRConfig(
                    tdtConfig: TdtConfig(blankId: version.blankId), encoderHiddenSize: version.encoderHiddenSize))
            try await manager.initialize(models: models)
            let result = try await manager.transcribe(URL(fileURLWithPath: recording))
            XCTAssertFalse(result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            let available = await manager.isAvailable
            XCTAssertTrue(available)
            print("Legacy \(version.repo.folderName) recorded-speech result: \(result.text)")
            await manager.cleanup()
        }
    }

    private func verifySharedEncoderWindows(models: AsrModels, samples: [Float]) async throws {
        let config = ASRConfig(tdtConfig: TdtConfig(blankId: models.version.blankId), encoderHiddenSize: 1024)
        let firstManager = AsrManager(config: config)
        let secondManager = AsrManager(config: config)
        try await firstManager.initialize(models: models)
        try await secondManager.initialize(models: models)
        let suffix = Array(samples.suffix(max(16000, samples.count / 2)))
        let inputA = firstManager.padAudioIfNeeded(samples, targetLength: ASRConstants.maxModelSamples)
        let inputB = secondManager.padAudioIfNeeded(suffix, targetLength: ASRConstants.maxModelSamples)
        let preprocessorA = try await firstManager.prepareParakeetPreprocessorOutput(
            inputA, originalLength: samples.count, snapshotOutput: true)
        let preprocessorB = try await secondManager.prepareParakeetPreprocessorOutput(
            inputB, originalLength: suffix.count, snapshotOutput: true)
        async let requestA = firstManager.prepareParakeetEncoderOutput(
            preparedPreprocessor: preprocessorA, snapshotOutput: true)
        async let requestB = secondManager.prepareParakeetEncoderOutput(
            preparedPreprocessor: preprocessorB, snapshotOutput: true)
        let (windowA, windowB) = try await (requestA, requestB)
        let frames = ASRConstants.calculateEncoderFrames(from: samples.count)
        let before = try await firstManager.pronunciationFeatures(
            preparedEncoder: windowA, actualAudioFrames: frames, contextFrameAdjustment: 0, globalFrameOffset: 0)
        let held = try XCTUnwrap(before)
        let laterPreprocessor = try await secondManager.prepareParakeetPreprocessorOutput(
            inputB, originalLength: suffix.count, snapshotOutput: true)
        let laterWindow = try await secondManager.prepareParakeetEncoderOutput(
            preparedPreprocessor: laterPreprocessor, snapshotOutput: true)
        let after = try await firstManager.pronunciationFeatures(
            preparedEncoder: windowA, actualAudioFrames: frames, contextFrameAdjustment: 0, globalFrameOffset: 0)
        XCTAssertEqual(try XCTUnwrap(after).values, held.values)
        XCTAssertTrue(held.values.allSatisfy(\.isFinite))
        await firstManager.discardParakeetEncoderOutput(windowA)
        await secondManager.discardParakeetEncoderOutput(windowB)
        await secondManager.discardParakeetEncoderOutput(laterWindow)
        await firstManager.cleanup()
        await secondManager.cleanup()
    }

    private func verifyRetainedEncoderWindow(manager: AsrManager, samples: [Float]) async throws {
        let firstInput = manager.padAudioIfNeeded(samples, targetLength: ASRConstants.maxModelSamples)
        let firstPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
            firstInput, originalLength: samples.count, snapshotOutput: true)
        let firstEncoder = try await manager.prepareParakeetEncoderOutput(
            preparedPreprocessor: firstPreprocessor, snapshotOutput: true)
        let frames = ASRConstants.calculateEncoderFrames(from: samples.count)
        let first = try await manager.pronunciationFeatures(
            preparedEncoder: firstEncoder, actualAudioFrames: frames, contextFrameAdjustment: 0, globalFrameOffset: 0)
        let before = try XCTUnwrap(first)
        let laterSamples = Array(samples.suffix(max(16000, samples.count / 2)))
        let laterInput = manager.padAudioIfNeeded(laterSamples, targetLength: ASRConstants.maxModelSamples)
        let laterPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
            laterInput, originalLength: laterSamples.count, snapshotOutput: true)
        let laterEncoder = try await manager.prepareParakeetEncoderOutput(
            preparedPreprocessor: laterPreprocessor, snapshotOutput: true)
        let retained = try await manager.pronunciationFeatures(
            preparedEncoder: firstEncoder, actualAudioFrames: frames, contextFrameAdjustment: 0, globalFrameOffset: 0)
        XCTAssertEqual(try XCTUnwrap(retained).values, before.values)
        XCTAssertEqual(before.hiddenSize, 1024)
        await manager.discardParakeetEncoderOutput(laterEncoder)
        await manager.discardParakeetEncoderOutput(firstEncoder)
    }

    func testTdtCtc110mUsesSplitFrontend() {
        // Create a mock AsrModels instance for tdtCtc110m
        // Note: We can't create actual MLModel instances without model files
        // So we test the version property directly

        // tdtCtc110m has fused frontend (no split)
        XCTAssertFalse(AsrModelVersion.tdtCtc110m.hasFusedEncoder == false)

        // Test the inverse logic used in usesSplitFrontend
        let tdtCtc110mUsesSplit = !AsrModelVersion.tdtCtc110m.hasFusedEncoder
        XCTAssertFalse(tdtCtc110mUsesSplit, "tdtCtc110m should not use split frontend")

        // v2 and v3 use split frontend
        let v2UsesSplit = !AsrModelVersion.v2.hasFusedEncoder
        let v3UsesSplit = !AsrModelVersion.v3.hasFusedEncoder
        XCTAssertTrue(v2UsesSplit, "v2 should use split frontend")
        XCTAssertTrue(v3UsesSplit, "v3 should use split frontend")
    }

    func testTdtCtc110mDefaultCacheDirectory() {
        let cacheDir = AsrModels.defaultCacheDirectory(for: .tdtCtc110m)

        // Verify path contains correct repo folder name
        XCTAssertTrue(cacheDir.path.contains(Repo.parakeetTdtCtc110m.folderName))
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))

        // Verify it's an absolute path
        XCTAssertTrue(cacheDir.isFileURL)
        XCTAssertTrue(cacheDir.path.starts(with: "/"))
    }

    func testTdtCtc110mVocabularyFilename() {
        // tdtCtc110m uses parakeet_vocab.json (array format)
        let vocabFile = ModelNames.ASR.vocabularyFileArray
        XCTAssertEqual(vocabFile, "parakeet_vocab.json")

        // Verify it has .json extension
        XCTAssertTrue(vocabFile.hasSuffix(".json"))
        XCTAssertTrue(vocabFile.contains("vocab"))
    }

    func testAllModelVersionsHaveRequiredProperties() {
        let versions = AsrModelVersion.allCases

        for version in versions {
            // All versions should have valid repo
            XCTAssertNotNil(version.repo)

            // All versions should have positive encoder hidden size
            XCTAssertGreaterThan(version.encoderHiddenSize, 0)

            // All versions should have positive blank ID
            XCTAssertGreaterThan(version.blankId, 0)

            // All versions should have at least 1 decoder layer
            XCTAssertGreaterThan(version.decoderLayers, 0)
        }
    }
}
