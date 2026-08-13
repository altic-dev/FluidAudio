import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    static let automaticPipelineMinimumPhysicalMemoryBytes: UInt64 = 24 * 1_024 * 1_024 * 1_024
    static let automaticPipelineMinimumAvailableMemoryBytes: UInt64 = 8 * 1_024 * 1_024 * 1_024

    let sampleSource: StreamingAudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    private typealias TokenWindow = AsrChunkTokenMerger.TokenWindow
    private struct TaskResult: Sendable {
        let index: Int
        let tokens: [TokenWindow]
        let workerIndex: Int
        let chunkEnd: Int
    }
    private let layout = ParakeetChunkLayout.standard
    private var sampleRate: Int { layout.sampleRate }
    private var overlapSeconds: Double { layout.overlapSeconds }
    private var strideSamples: Int { layout.strideSamples }

    /// Initialize with a streaming audio sample source for memory-efficient processing.
    init(sampleSource: StreamingAudioSampleSource) {
        self.sampleSource = sampleSource
        self.totalSamples = sampleSource.sampleCount
    }

    /// Convenience initializer for in-memory audio samples.
    init(audioSamples: [Float]) {
        self.init(sampleSource: ArrayAudioSampleSource(samples: audioSamples))
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil
    ) async throws -> ASRResult {
        let pipelineMode = ProcessInfo.processInfo.environment["FLUIDAUDIO_FRONTEND_PIPELINE_MODE"]
        let supportsPipelining = await manager.supportsParakeetFrontendPipelining()
        let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
        let availableMemoryBytes = SystemInfo.availableMemoryBytes()
        let pipeliningEnabled = Self.shouldPipelineFrontend(
            environmentMode: pipelineMode,
            supportsPipelining: supportsPipelining,
            physicalMemoryBytes: physicalMemoryBytes,
            availableMemoryBytes: availableMemoryBytes
        )
        let workerConcurrency = Self.effectiveWorkerConcurrency(
            configuredConcurrency: await manager.parallelChunkConcurrency,
            environmentMode: pipelineMode,
            physicalMemoryBytes: physicalMemoryBytes,
            availableMemoryBytes: availableMemoryBytes
        )
        let hybridPipeliningEnabled =
            pipelineMode?.lowercased() == "hybrid" && pipeliningEnabled && workerConcurrency > 1
        let chunkOutputs = try await processChunks(
            using: manager,
            pipeliningEnabled: pipeliningEnabled,
            hybridPipeliningEnabled: hybridPipeliningEnabled,
            workerConcurrency: workerConcurrency,
            progressHandler: progressHandler
        )

        guard var mergedTokens = chunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime),
                audioSampleCount: totalSamples
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = AsrChunkTokenMerger.mergePair(
                    mergedTokens, chunk, sampleRate: sampleRate, overlapSeconds: overlapSeconds)
            }
        }

        if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }
        let allDurations = mergedTokens.map { $0.duration }

        return await manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            tokenDurations: allDurations,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime),
            audioSampleCount: totalSamples
        )
    }

    static func shouldPipelineFrontend(
        environmentMode: String?,
        supportsPipelining: Bool,
        physicalMemoryBytes: UInt64,
        availableMemoryBytes: UInt64?
    ) -> Bool {
        guard supportsPipelining else { return false }

        switch environmentMode?.lowercased() {
        case "pipeline", "hybrid":
            return true
        case "serial", "workers":
            return false
        default:
            let hasPhysicalHeadroom =
                physicalMemoryBytes >= automaticPipelineMinimumPhysicalMemoryBytes
            let hasAvailableHeadroom = availableMemoryBytes.map {
                $0 >= automaticPipelineMinimumAvailableMemoryBytes
            } ?? true
            return hasPhysicalHeadroom && hasAvailableHeadroom
        }
    }

    static func effectiveWorkerConcurrency(
        configuredConcurrency: Int,
        environmentMode: String?,
        physicalMemoryBytes: UInt64,
        availableMemoryBytes: UInt64?
    ) -> Int {
        let configured = max(1, configuredConcurrency)
        switch environmentMode?.lowercased() {
        case "serial":
            return 1
        case "workers", "hybrid":
            return configured
        default:
            break
        }

        let gibibyte = UInt64(1_024 * 1_024 * 1_024)
        if let availableMemoryBytes, availableMemoryBytes < 4 * gibibyte {
            return 1
        }
        if physicalMemoryBytes <= 8 * gibibyte
            || availableMemoryBytes.map({ $0 < 6 * gibibyte }) == true
        {
            return min(configured, 2)
        }
        if physicalMemoryBytes <= 16 * gibibyte
            || availableMemoryBytes.map({ $0 < 8 * gibibyte }) == true
        {
            return min(configured, 3)
        }
        return configured
    }

    private typealias ChunkWork = ParakeetChunkWork

    private func processChunks(
        using manager: AsrManager,
        pipeliningEnabled: Bool,
        hybridPipeliningEnabled: Bool,
        workerConcurrency: Int,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        if hybridPipeliningEnabled,
            let workers = await makeWorkerPool(using: manager, count: workerConcurrency)
        {
            return try await processChunksHybridPipelined(
                using: workers,
                progressHandler: progressHandler
            )
        }

        if pipeliningEnabled {
            return try await processChunksPipelined(
                using: manager,
                progressHandler: progressHandler
            )
        }

        if workerConcurrency > 1,
            let workers = await makeWorkerPool(using: manager, count: workerConcurrency)
        {
            return try await processChunksParallel(
                using: workers,
                progressHandler: progressHandler
            )
        }

        return try await processChunksSerial(
            using: manager,
            progressHandler: progressHandler
        )
    }

    private func processChunksHybridPipelined(
        using workers: [AsrManager],
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        let chunkCount = totalSamples > 0 ? ((totalSamples - 1) / strideSamples) + 1 : 0
        guard chunkCount > 0 else { return [] }

        var chunkOutputs = Array<[TokenWindow]?>(repeating: nil, count: chunkCount)
        var completedThroughChunkEnd = 0

        try await withThrowingTaskGroup(of: [TaskResult].self) { group in
            for workerIndex in workers.indices where workerIndex < chunkCount {
                let worker = workers[workerIndex]
                group.addTask {
                    try await processChunkLanePipelined(
                        using: worker,
                        workerIndex: workerIndex,
                        firstChunkIndex: workerIndex,
                        chunkIndexStride: workers.count
                    )
                }
            }

            for try await laneResults in group {
                for result in laneResults {
                    chunkOutputs[result.index] = result.tokens
                    completedThroughChunkEnd = max(completedThroughChunkEnd, result.chunkEnd)
                }
                if let progressHandler, completedThroughChunkEnd < totalSamples {
                    await progressHandler(
                        min(1.0, max(0.0, Double(completedThroughChunkEnd) / Double(totalSamples)))
                    )
                }
            }
        }

        return try chunkOutputs.enumerated().map { index, output in
            guard let output else {
                throw ASRError.processingFailed("Missing hybrid Parakeet output for chunk \(index)")
            }
            return output
        }
    }

    private func processChunkLanePipelined(
        using manager: AsrManager,
        workerIndex: Int,
        firstChunkIndex: Int,
        chunkIndexStride: Int
    ) async throws -> [TaskResult] {
        func work(at chunkIndex: Int) throws -> ChunkWork? {
            try makeChunkWork(
                chunkStart: chunkIndex * strideSamples,
                chunkIndex: chunkIndex
            )
        }

        guard var currentWork = try work(at: firstChunkIndex) else { return [] }

        var results: [TaskResult] = []
        var currentChunkIndex = firstChunkIndex
        var decoderState = TdtDecoderState.make(decoderLayers: await manager.getDecoderLayers())
        var currentEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?
        var lookaheadWork: ChunkWork?
        var lookaheadChunkIndex: Int?
        var lookaheadPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?

        while true {
            var currentEncoder: PreparedParakeetEncoderHandle?
            var initialPreprocessor: PreparedParakeetPreprocessorHandle?
            var lookaheadPreprocessor: PreparedParakeetPreprocessorHandle?
            var followingWork: ChunkWork?
            var followingChunkIndex: Int?
            var followingPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?
            var nextEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?

            do {
                try Task.checkCancellation()
                if let currentEncoderTask {
                    currentEncoder = try await currentEncoderTask.value
                } else {
                    initialPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
                        currentWork.paddedSamples,
                        originalLength: currentWork.samples.count,
                        snapshotOutput: true
                    )
                    let nextChunkIndex = currentChunkIndex + chunkIndexStride
                    lookaheadWork = try work(at: nextChunkIndex)
                    lookaheadChunkIndex = lookaheadWork == nil ? nil : nextChunkIndex
                    lookaheadPreprocessorTask =
                        lookaheadWork.map { makePreprocessorTask(for: $0, using: manager) }
                    guard let initialPreprocessor else {
                        throw ASRError.processingFailed("Initial hybrid preprocessor output is unavailable")
                    }
                    currentEncoder = try await manager.prepareParakeetEncoderOutput(
                        preparedPreprocessor: initialPreprocessor,
                        snapshotOutput: true
                    )
                }

                if lookaheadWork != nil, let lookaheadChunkIndex {
                    guard let lookaheadPreprocessorTask else {
                        throw ASRError.processingFailed("Hybrid lookahead preprocessor task is unavailable")
                    }
                    lookaheadPreprocessor = try await lookaheadPreprocessorTask.value
                    let nextChunkIndex = lookaheadChunkIndex + chunkIndexStride
                    followingWork = try work(at: nextChunkIndex)
                    followingChunkIndex = followingWork == nil ? nil : nextChunkIndex
                    followingPreprocessorTask =
                        followingWork.map { makePreprocessorTask(for: $0, using: manager) }
                    guard let lookaheadPreprocessor else {
                        throw ASRError.processingFailed("Hybrid lookahead preprocessor output is unavailable")
                    }
                    nextEncoderTask = makeEncoderTask(for: lookaheadPreprocessor, using: manager)
                }

                guard let currentEncoder else {
                    throw ASRError.processingFailed("Hybrid encoder output is unavailable")
                }
                decoderState.reset()
                let output = try await transcribeChunk(
                    work: currentWork,
                    preparedEncoder: currentEncoder,
                    using: manager,
                    decoderState: &decoderState
                )
                results.append(
                    TaskResult(
                        index: currentChunkIndex,
                        tokens: try ParakeetChunkInference.makeTokenWindow(from: output),
                        workerIndex: workerIndex,
                        chunkEnd: currentWork.chunkEnd
                    )
                )
            } catch {
                if let initialPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(initialPreprocessor)
                }
                if let lookaheadPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(lookaheadPreprocessor)
                }
                if let currentEncoder {
                    await manager.discardParakeetEncoderOutput(currentEncoder)
                }
                await discardPreprocessorTask(lookaheadPreprocessorTask, using: manager)
                await discardPreprocessorTask(followingPreprocessorTask, using: manager)
                await discardEncoderTask(currentEncoderTask, using: manager)
                await discardEncoderTask(nextEncoderTask, using: manager)
                throw error
            }

            guard let nextWork = lookaheadWork, let nextChunkIndex = lookaheadChunkIndex else {
                break
            }
            currentWork = nextWork
            currentChunkIndex = nextChunkIndex
            currentEncoderTask = nextEncoderTask
            lookaheadWork = followingWork
            lookaheadChunkIndex = followingChunkIndex
            lookaheadPreprocessorTask = followingPreprocessorTask
        }

        return results
    }

    private func processChunksParallel(
        using workers: [AsrManager],
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        guard var currentWork = try makeChunkWork(chunkStart: 0, chunkIndex: 0) else {
            return []
        }

        let decoderLayers = await workers[0].getDecoderLayers()
        var chunkOutputs: [[TokenWindow]?] = []
        var availableWorkers = Array(workers.indices)
        var inFlight = 0
        var chunkIndex = 0
        var completedThroughChunkEnd = 0

        func collectNextResult(
            _ group: inout ThrowingTaskGroup<TaskResult, Error>
        ) async throws {
            guard inFlight > 0, let finished = try await group.next() else { return }
            chunkOutputs[finished.index] = finished.tokens
            availableWorkers.append(finished.workerIndex)
            inFlight -= 1
            completedThroughChunkEnd = max(completedThroughChunkEnd, finished.chunkEnd)

            if let progressHandler, completedThroughChunkEnd < totalSamples {
                let progress = min(
                    1.0,
                    max(0.0, Double(completedThroughChunkEnd) / Double(totalSamples))
                )
                await progressHandler(progress)
            }
        }

        try await withThrowingTaskGroup(of: TaskResult.self) { group in
            while true {
                try Task.checkCancellation()
                if availableWorkers.isEmpty {
                    try await collectNextResult(&group)
                }

                guard !availableWorkers.isEmpty else {
                    throw ASRError.processingFailed("No Parakeet chunk worker is available")
                }

                let workerIndex = availableWorkers.removeFirst()
                let worker = workers[workerIndex]
                let work = currentWork
                let index = chunkIndex
                chunkOutputs.append(nil)

                group.addTask {
                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()
                    let tokens = try await ParakeetChunkInference.transcribe(
                        work: work,
                        using: worker,
                        decoderState: &decoderState
                    )
                    return TaskResult(
                        index: index,
                        tokens: tokens,
                        workerIndex: workerIndex,
                        chunkEnd: work.chunkEnd
                    )
                }
                inFlight += 1

                if work.isLastChunk {
                    break
                }

                chunkIndex += 1
                guard
                    let nextWork = try makeChunkWork(
                        chunkStart: work.chunkStart + strideSamples,
                        chunkIndex: chunkIndex
                    )
                else {
                    break
                }
                currentWork = nextWork
            }

            while inFlight > 0 {
                try Task.checkCancellation()
                try await collectNextResult(&group)
            }
        }

        return try chunkOutputs.enumerated().map { index, output in
            guard let output else {
                throw ASRError.processingFailed("Missing Parakeet output for chunk \(index)")
            }
            return output
        }
    }

    private func makeWorkerPool(using manager: AsrManager, count: Int) async -> [AsrManager]? {
        guard count > 0 else { return nil }
        var workers = [manager]
        guard count > 1 else { return workers }

        for _ in 1..<count {
            guard let worker = await manager.makeWorkerClone() else { return nil }
            workers.append(worker)
        }
        logger.debug("ChunkProcessor using worker pool of size \(workers.count)")
        return workers
    }

    private func processChunksSerial(
        using manager: AsrManager,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        guard var currentWork = try makeChunkWork(chunkStart: 0, chunkIndex: 0) else {
            return []
        }

        var chunkOutputs: [[TokenWindow]] = []
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )

        while true {
            var preparedPreprocessor: PreparedParakeetPreprocessorHandle?
            var preparedEncoder: PreparedParakeetEncoderHandle?
            do {
                try Task.checkCancellation()
                preparedPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
                    currentWork.paddedSamples,
                    originalLength: currentWork.samples.count
                )
                guard let preprocessorToEncode = preparedPreprocessor else {
                    throw ASRError.processingFailed("Preprocessor output was not prepared")
                }
                preparedEncoder = try await manager.prepareParakeetEncoderOutput(
                    preparedPreprocessor: preprocessorToEncode
                )
                preparedPreprocessor = nil
                guard let encoderToDecode = preparedEncoder else {
                    throw ASRError.processingFailed("Encoder output was not prepared")
                }
                chunkDecoderState.reset()
                let output = try await transcribeChunk(
                    work: currentWork,
                    preparedEncoder: encoderToDecode,
                    using: manager,
                    decoderState: &chunkDecoderState
                )
                preparedEncoder = nil
                chunkOutputs.append(try ParakeetChunkInference.makeTokenWindow(from: output))
            } catch {
                if let preparedPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(preparedPreprocessor)
                }
                if let preparedEncoder {
                    await manager.discardParakeetEncoderOutput(preparedEncoder)
                }
                throw error
            }

            if currentWork.isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(currentWork.chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            guard
                let nextWork = try makeChunkWork(
                    chunkStart: currentWork.chunkStart + strideSamples,
                    chunkIndex: chunkIndex + 1
                )
            else {
                break
            }
            currentWork = nextWork
            chunkIndex += 1
        }

        return chunkOutputs
    }

    private func processChunksPipelined(
        using manager: AsrManager,
        progressHandler: ((Double) async -> Void)?
    ) async throws -> [[TokenWindow]] {
        guard var currentWork = try makeChunkWork(chunkStart: 0, chunkIndex: 0) else {
            return []
        }

        var chunkOutputs: [[TokenWindow]] = []
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.getDecoderLayers()
        )
        var currentEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?
        var lookaheadWork: ChunkWork?
        var lookaheadPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?

        while true {
            var currentEncoder: PreparedParakeetEncoderHandle?
            var initialPreprocessor: PreparedParakeetPreprocessorHandle?
            var lookaheadPreprocessor: PreparedParakeetPreprocessorHandle?
            var followingWork: ChunkWork?
            var followingPreprocessorTask: Task<PreparedParakeetPreprocessorHandle, Error>?
            var nextEncoderTask: Task<PreparedParakeetEncoderHandle, Error>?

            do {
                try Task.checkCancellation()
                if let currentEncoderTask {
                    currentEncoder = try await currentEncoderTask.value
                } else {
                    initialPreprocessor = try await manager.prepareParakeetPreprocessorOutput(
                        currentWork.paddedSamples,
                        originalLength: currentWork.samples.count,
                        snapshotOutput: true
                    )
                    if !currentWork.isLastChunk {
                        lookaheadWork = try makeChunkWork(
                            chunkStart: currentWork.chunkStart + strideSamples,
                            chunkIndex: chunkIndex + 1
                        )
                        lookaheadPreprocessorTask =
                            lookaheadWork.map { makePreprocessorTask(for: $0, using: manager) }
                    }
                    guard let initialPreprocessorHandle = initialPreprocessor else {
                        throw ASRError.processingFailed("Initial preprocessor output was not prepared")
                    }
                    currentEncoder = try await manager.prepareParakeetEncoderOutput(
                        preparedPreprocessor: initialPreprocessorHandle,
                        snapshotOutput: true
                    )
                    initialPreprocessor = nil
                }

                if let lookaheadWork {
                    guard let lookaheadPreprocessorTask else {
                        throw ASRError.processingFailed("Lookahead preprocessor task is unavailable")
                    }
                    lookaheadPreprocessor = try await lookaheadPreprocessorTask.value

                    if !lookaheadWork.isLastChunk {
                        followingWork = try makeChunkWork(
                            chunkStart: lookaheadWork.chunkStart + strideSamples,
                            chunkIndex: chunkIndex + 2
                        )
                        followingPreprocessorTask =
                            followingWork.map { makePreprocessorTask(for: $0, using: manager) }
                    }

                    guard let lookaheadPreprocessorHandle = lookaheadPreprocessor else {
                        throw ASRError.processingFailed("Lookahead preprocessor output was not prepared")
                    }
                    nextEncoderTask = makeEncoderTask(
                        for: lookaheadPreprocessorHandle,
                        using: manager
                    )
                    lookaheadPreprocessor = nil
                }

                guard let encoderToDecode = currentEncoder else {
                    throw ASRError.processingFailed("Encoder output was not prepared")
                }
                chunkDecoderState.reset()
                let output = try await transcribeChunk(
                    work: currentWork,
                    preparedEncoder: encoderToDecode,
                    using: manager,
                    decoderState: &chunkDecoderState
                )
                currentEncoder = nil
                chunkOutputs.append(try ParakeetChunkInference.makeTokenWindow(from: output))
            } catch {
                if let initialPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(initialPreprocessor)
                }
                if let lookaheadPreprocessor {
                    await manager.discardParakeetPreprocessorOutput(lookaheadPreprocessor)
                }
                if let currentEncoder {
                    await manager.discardParakeetEncoderOutput(currentEncoder)
                }
                await discardPreprocessorTask(lookaheadPreprocessorTask, using: manager)
                await discardPreprocessorTask(followingPreprocessorTask, using: manager)
                await discardEncoderTask(currentEncoderTask, using: manager)
                await discardEncoderTask(nextEncoderTask, using: manager)
                throw error
            }

            if currentWork.isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(currentWork.chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            guard let nextWork = lookaheadWork else {
                break
            }
            currentWork = nextWork
            currentEncoderTask = nextEncoderTask
            lookaheadWork = followingWork
            lookaheadPreprocessorTask = followingPreprocessorTask
            chunkIndex += 1
        }

        return chunkOutputs
    }

    private func makeChunkWork(chunkStart: Int, chunkIndex: Int) throws -> ChunkWork? {
        try layout.makeWork(
            totalSamples: totalSamples,
            chunkStart: chunkStart,
            chunkIndex: chunkIndex,
            readSamples: { offset, count in
                try readSamples(offset: offset, count: count)
            }
        )
    }

    private func makePreprocessorTask(
        for work: ChunkWork,
        using manager: AsrManager
    ) -> Task<PreparedParakeetPreprocessorHandle, Error> {
        Task {
            try await manager.prepareParakeetPreprocessorOutput(
                work.paddedSamples,
                originalLength: work.samples.count,
                snapshotOutput: true
            )
        }
    }

    private func discardPreprocessorTask(
        _ task: Task<PreparedParakeetPreprocessorHandle, Error>?,
        using manager: AsrManager
    ) async {
        guard let task else { return }
        task.cancel()
        if let handle = try? await task.value {
            await manager.discardParakeetPreprocessorOutput(handle)
        }
    }

    private func makeEncoderTask(
        for preparedPreprocessor: PreparedParakeetPreprocessorHandle,
        using manager: AsrManager
    ) -> Task<PreparedParakeetEncoderHandle, Error> {
        Task {
            await Task.yield()
            return try await manager.prepareParakeetEncoderOutput(
                preparedPreprocessor: preparedPreprocessor,
                snapshotOutput: true
            )
        }
    }

    private func discardEncoderTask(
        _ task: Task<PreparedParakeetEncoderHandle, Error>?,
        using manager: AsrManager
    ) async {
        guard let task else { return }
        task.cancel()
        if let handle = try? await task.value {
            await manager.discardParakeetEncoderOutput(handle)
        }
    }

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    private func transcribeChunk(
        work: ChunkWork,
        preparedEncoder: PreparedParakeetEncoderHandle,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        try await ParakeetChunkInference.transcribe(
            work: work,
            preparedEncoder: preparedEncoder,
            using: manager,
            decoderState: &decoderState
        )
    }

}
