@preconcurrency import CoreML
import Foundation
import OSLog

struct PreparedParakeetPreprocessorHandle: Hashable, Sendable {
    let id: UUID
}

struct PreparedParakeetEncoderHandle: Hashable, Sendable {
    let id: UUID
}

struct ParakeetPreprocessorOutput {
    let input: MLFeatureProvider
    let output: MLFeatureProvider
    let audioArray: MLMultiArray?
}

struct ParakeetEncoderOutput {
    let encoderOutput: MLFeatureProvider
    let ownedBackings: [MLMultiArray]
}

extension AsrManager {
    func supportsParakeetFrontendPipelining() -> Bool {
        encoderModel != nil
    }

    internal func transcribeWithState(
        _ audioSamples: [Float],
        source: AudioSource,
        applyVocabularyBoosting: Bool = true,
        isolatedDecoderState: Bool = false
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Incremental previews reprocess the same short utterance from sample zero, so they must
        // not inherit decoder context from an earlier preview or another session.
        var decoderState: TdtDecoderState
        if isolatedDecoderState {
            decoderState = TdtDecoderState.make(decoderLayers: getDecoderLayers())
        } else {
            switch source {
            case .microphone:
                decoderState = microphoneDecoderState
            case .system:
                decoderState = systemDecoderState
            }
        }

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= ASRConstants.maxModelSamples {
            let originalLength = audioSamples.count
            let frameAlignedCandidate =
                ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                    / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
            let frameAlignedLength: Int
            let alignedSamples: [Float]
            if frameAlignedCandidate > originalLength && frameAlignedCandidate <= ASRConstants.maxModelSamples {
                frameAlignedLength = frameAlignedCandidate
                alignedSamples = audioSamples + Array(repeating: 0, count: frameAlignedLength - originalLength)
            } else {
                frameAlignedLength = originalLength
                alignedSamples = audioSamples
            }
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState
            )

            var result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            // Auto-apply vocabulary rescoring when configured
            if applyVocabularyBoosting, vocabBoostingEnabled {
                result = await applyVocabularyRescoring(result: result, audioSamples: audioSamples)
            }

            if !isolatedDecoderState {
                switch source {
                case .microphone:
                    microphoneDecoderState = decoderState
                case .system:
                    systemDecoderState = decoderState
                }
            }

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples)
        var result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            }
        )

        // Auto-apply vocabulary rescoring when configured
        if applyVocabularyBoosting, vocabBoostingEnabled {
            result = await applyVocabularyRescoring(result: result, audioSamples: audioSamples)
        }

        if !isolatedDecoderState {
            switch source {
            case .microphone:
                microphoneDecoderState = decoderState
            case .system:
                systemDecoderState = decoderState
            }
        }

        return result
    }

    func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        let preparedPreprocessor = try await prepareParakeetPreprocessorOutput(
            paddedAudio,
            originalLength: originalLength
        )
        let preparedEncoder = try await prepareParakeetEncoderOutput(
            preparedPreprocessor: preparedPreprocessor
        )
        return try await executeMLInferenceWithTimings(
            preparedEncoder: preparedEncoder,
            paddedAudio: paddedAudio,
            originalLength: originalLength,
            actualAudioFrames: actualAudioFrames,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )
    }

    func prepareParakeetPreprocessorOutput(
        _ paddedAudio: [Float],
        originalLength: Int?,
        snapshotOutput: Bool = false
    ) async throws -> PreparedParakeetPreprocessorHandle {
        let preprocessorInput = try await preparePreprocessorInput(
            paddedAudio, actualLength: originalLength
        )

        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        do {
            guard let preprocessorModel = preprocessorModel else {
                throw ASRError.notInitialized
            }

            try Task.checkCancellation()
            let preprocessorOutput = try await preprocessorModel.compatPrediction(
                from: preprocessorInput,
                options: predictionOptions
            )
            let storedOutput =
                snapshotOutput
                ? try Self.snapshotFeatureProvider(preprocessorOutput)
                : preprocessorOutput

            let handle = PreparedParakeetPreprocessorHandle(id: UUID())
            preparedParakeetPreprocessorOutputs[handle.id] = ParakeetPreprocessorOutput(
                input: preprocessorInput,
                output: storedOutput,
                audioArray: preprocessorAudioArray
            )
            return handle
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    func prepareParakeetEncoderOutput(
        preparedPreprocessor handle: PreparedParakeetPreprocessorHandle,
        snapshotOutput: Bool = false
    ) async throws -> PreparedParakeetEncoderHandle {
        guard let preparedOutput = preparedParakeetPreprocessorOutputs.removeValue(forKey: handle.id) else {
            throw ASRError.processingFailed("Prepared preprocessor output is unavailable")
        }

        var ownedBackings: [MLMultiArray] = []
        do {
            let encoderOutputProvider: MLFeatureProvider
            if let encoderModel {
                let encoderInput = try prepareEncoderInput(
                    encoder: encoderModel,
                    preprocessorOutput: preparedOutput.output,
                    originalInput: preparedOutput.input
                )

                try Task.checkCancellation()
                let encoderPredictionOptions: MLPredictionOptions
                if snapshotOutput {
                    let backings = try await makeOutputBackings(for: encoderModel)
                    let options = AsrModels.optimizedPredictionOptions()
                    options.outputBackings = backings
                    encoderPredictionOptions = options
                    ownedBackings = Array(backings.values)
                } else {
                    encoderPredictionOptions = predictionOptions
                }
                encoderOutputProvider = try await encoderModel.compatPrediction(
                    from: encoderInput,
                    options: encoderPredictionOptions
                )
            } else {
                encoderOutputProvider = preparedOutput.output
            }

            if let preprocessorAudioArray = preparedOutput.audioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }

            let encoderHandle = PreparedParakeetEncoderHandle(id: UUID())
            preparedParakeetEncoderOutputs[encoderHandle.id] = ParakeetEncoderOutput(
                encoderOutput: encoderOutputProvider,
                ownedBackings: ownedBackings
            )
            return encoderHandle
        } catch {
            await returnEncoderOutputBackings(ownedBackings)
            if let preprocessorAudioArray = preparedOutput.audioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    func executeMLInferenceWithTimings(
        preparedEncoder handle: PreparedParakeetEncoderHandle,
        paddedAudio: [Float],
        originalLength: Int?,
        actualAudioFrames: Int?,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        guard let preparedOutput = preparedParakeetEncoderOutputs.removeValue(forKey: handle.id) else {
            throw ASRError.processingFailed("Prepared encoder output is unavailable")
        }

        do {
            let rawEncoderOutput = try extractFeatureValue(
                from: preparedOutput.encoderOutput, key: "encoder", errorMessage: "Invalid encoder output"
            )
            let encoderLength = try extractFeatureValue(
                from: preparedOutput.encoderOutput, key: "encoder_length",
                errorMessage: "Invalid encoder output length"
            )

            let encoderSequenceLength = encoderLength[0].intValue

            // Calculate actual audio frames if not provided using shared constants
            let actualFrames =
                actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: originalLength ?? paddedAudio.count)

            try capturePronunciationEncoderFeaturesIfEnabled(
                rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                contextFrameAdjustment: contextFrameAdjustment,
                globalFrameOffset: globalFrameOffset
            )

            let hypothesis = try await tdtDecodeWithTimings(
                encoderOutput: rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                originalAudioSamples: paddedAudio,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset
            )

            await returnEncoderOutputBackings(preparedOutput.ownedBackings)
            return (hypothesis, encoderSequenceLength)
        } catch {
            await returnEncoderOutputBackings(preparedOutput.ownedBackings)
            throw error
        }
    }

    func discardParakeetPreprocessorOutput(_ handle: PreparedParakeetPreprocessorHandle) async {
        guard let preparedOutput = preparedParakeetPreprocessorOutputs.removeValue(forKey: handle.id) else {
            return
        }
        if let audioArray = preparedOutput.audioArray {
            await sharedMLArrayCache.returnArray(audioArray)
        }
    }

    func discardParakeetEncoderOutput(_ handle: PreparedParakeetEncoderHandle) async {
        guard let output = preparedParakeetEncoderOutputs.removeValue(forKey: handle.id) else {
            return
        }
        await returnEncoderOutputBackings(output.ownedBackings)
    }

    private func makeOutputBackings(for model: MLModel) async throws -> [String: MLMultiArray] {
        var backings: [String: MLMultiArray] = [:]
        do {
            for (name, description) in model.modelDescription.outputDescriptionsByName {
                guard let constraint = description.multiArrayConstraint else { continue }
                backings[name] = try await sharedMLArrayCache.getArray(
                    shape: constraint.shape,
                    dataType: constraint.dataType
                )
            }
            guard !backings.isEmpty else {
                throw ASRError.processingFailed("Model has no reusable multi-array outputs")
            }
            return backings
        } catch {
            await returnOutputBackings(Array(backings.values))
            throw error
        }
    }

    private func returnOutputBackings(_ backings: [MLMultiArray]) async {
        for backing in backings {
            await sharedMLArrayCache.returnArray(backing)
        }
    }

    private static func snapshotFeatureProvider(
        _ provider: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        var features: [String: MLFeatureValue] = [:]
        for name in provider.featureNames {
            guard let value = provider.featureValue(for: name) else { continue }
            if let array = value.multiArrayValue {
                features[name] = MLFeatureValue(multiArray: try snapshotMultiArray(array))
            } else {
                features[name] = value
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private static func snapshotMultiArray(_ source: MLMultiArray) throws -> MLMultiArray {
        let storageElementCount = zip(source.shape, source.strides).reduce(1) { extent, pair in
            let (dimension, stride) = pair
            return extent + max(0, dimension.intValue - 1) * stride.intValue
        }
        let byteCount = storageElementCount * ANEMemoryUtils.getElementSize(for: source.dataType)
        let storage = UnsafeMutableRawPointer.allocate(byteCount: max(byteCount, 1), alignment: 64)
        if byteCount > 0 {
            memcpy(storage, source.dataPointer, byteCount)
        }

        do {
            return try MLMultiArray(
                dataPointer: storage,
                shape: source.shape,
                dataType: source.dataType,
                strides: source.strides,
                deallocator: { pointer in pointer.deallocate() }
            )
        } catch {
            storage.deallocate()
            throw error
        }
    }

    private func returnEncoderOutputBackings(_ backings: [MLMultiArray]) async {
        for backing in backings {
            await sharedMLArrayCache.returnArray(backing)
        }
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Streaming-friendly chunk transcription that preserves decoder state and supports start-frame offset.
    /// This is used by both sliding window chunking and streaming paths to unify behavior.
    public func transcribeStreamingChunk(
        _ chunkSamples: [Float],
        source: AudioSource,
        previousTokens: [Int] = [],
        isLastChunk: Bool = false
    ) async throws -> (
        tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int], encoderSequenceLength: Int
    ) {
        // Select and copy decoder state for the source
        var state = (source == .microphone) ? microphoneDecoderState : systemDecoderState

        let originalLength = chunkSamples.count
        let frameAlignedCandidate =
            ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
        let frameAlignedLength: Int
        let alignedSamples: [Float]
        if previousTokens.isEmpty
            && frameAlignedCandidate > originalLength
            && frameAlignedCandidate <= ASRConstants.maxModelSamples
        {
            frameAlignedLength = frameAlignedCandidate
            alignedSamples = chunkSamples + Array(repeating: 0, count: frameAlignedLength - originalLength)
        } else {
            frameAlignedLength = originalLength
            alignedSamples = chunkSamples
        }
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &state,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk
        )

        // Persist updated state back to the source-specific slot
        if source == .microphone {
            microphoneDecoderState = state
        } else {
            systemDecoderState = state
        }

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences
            let adjustedDurations =
                removedCount > 0 ? Array(hypothesis.tokenDurations.dropFirst(removedCount)) : hypothesis.tokenDurations

            return (deduped, adjustedTimestamps, adjustedConfidences, adjustedDurations, encLen)
        }

        return (
            hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations, encLen
        )
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        audioSampleCount: Int? = nil,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {

        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSampleCount ?? audioSamples.count) / TimeInterval(config.sampleRate)

        // Convert timestamps to TokenTiming objects if provided
        let timingsFromTimestamps = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        // Use existing timings if provided, otherwise use timings from timestamps
        let resultTimings = tokenTimings.isEmpty ? timingsFromTimestamps : finalTimings

        // Calculate confidence based on actual model confidence scores from TDT decoder
        let confidence = calculateConfidence(
            duration: duration,
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

    nonisolated internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

    /// Calculate confidence score based purely on TDT model token confidence scores
    /// Returns the average of token-level softmax probabilities from the decoder
    /// Range: 0.1 (empty transcription) to 1.0 (perfect confidence)
    private func calculateConfidence(
        duration: Double, tokenCount: Int, isEmpty: Bool, tokenConfidences: [Float]
    ) -> Float {
        // Empty transcription gets low confidence
        if isEmpty {
            return 0.1
        }

        // We should always have token confidence scores from the TDT decoder
        guard !tokenConfidences.isEmpty && tokenConfidences.count == tokenCount else {
            logger.warning("Expected token confidences but got none - this should not happen")
            return 0.5  // Default middle confidence if something went wrong
        }

        // Return pure model confidence: average of token-level softmax probabilities
        let meanConfidence = tokenConfidences.reduce(0.0, +) / Float(tokenConfidences.count)

        // Ensure confidence is in valid range (clamp to avoid edge cases)
        return max(0.1, min(1.0, meanConfidence))
    }

    /// Convert frame timestamps to TokenTiming objects
    private func createTokenTimings(
        from tokenIds: [Int], timestamps: [Int], confidences: [Float], tokenDurations: [Int] = []
    ) -> [TokenTiming] {
        guard
            !tokenIds.isEmpty && !timestamps.isEmpty && tokenIds.count == timestamps.count
                && confidences.count == tokenIds.count
        else {
            return []
        }

        var timings: [TokenTiming] = []

        // Create combined data for sorting
        let combinedData = zip(
            zip(zip(tokenIds, timestamps), confidences),
            tokenDurations.isEmpty ? Array(repeating: 0, count: tokenIds.count) : tokenDurations
        ).map {
            (tokenId: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }

        // Sort by timestamp to ensure chronological order
        let sortedData = combinedData.sorted { $0.timestamp < $1.timestamp }

        for i in 0..<sortedData.count {
            let data = sortedData[i]
            let tokenId = data.tokenId
            let frameIndex = data.timestamp

            // Convert encoder frame index to time (80ms per frame)
            let startTime = TimeInterval(frameIndex) * 0.08

            // Calculate end time using actual token duration if available
            let endTime: TimeInterval
            if !tokenDurations.isEmpty && data.duration > 0 {
                // Use actual token duration (convert frames to time: duration * 0.08)
                let durationInSeconds = TimeInterval(data.duration) * 0.08
                endTime = startTime + max(durationInSeconds, 0.08)  // Minimum 80ms duration
            } else if i < sortedData.count - 1 {
                // Fallback: Use next token's start time as this token's end time
                let nextStartTime = TimeInterval(sortedData[i + 1].timestamp) * 0.08
                endTime = max(nextStartTime, startTime + 0.08)  // Ensure end > start
            } else {
                // Last token: assume minimum duration
                endTime = startTime + 0.08
            }

            // Validate that end time is after start time
            let validatedEndTime = max(endTime, startTime + 0.001)  // Minimum 1ms gap

            // Get token text from vocabulary if available and normalize for timing display
            let rawToken = vocabulary[tokenId] ?? "token_\(tokenId)"
            let tokenText = normalizedTimingToken(rawToken)

            // Use actual confidence score from TDT decoder
            let tokenConfidence = data.confidence

            let timing = TokenTiming(
                token: tokenText,
                tokenId: tokenId,
                startTime: startTime,
                endTime: validatedEndTime,
                confidence: tokenConfidence
            )

            timings.append(timing)
        }
        return timings
    }

    /// Slice encoder output to remove left context frames (following NeMo approach)
    private func sliceEncoderOutput(
        _ encoderOutput: MLMultiArray,
        from startFrame: Int,
        newLength: Int
    ) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let hiddenSize = shape[2].intValue

        // Create new array with sliced dimensions
        let slicedArray = try MLMultiArray(
            shape: [batchSize, newLength, hiddenSize] as [NSNumber],
            dataType: encoderOutput.dataType
        )

        // Copy data from startFrame onwards
        let sourcePtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        let destPtr = slicedArray.dataPointer.bindMemory(to: Float.self, capacity: slicedArray.count)

        for t in 0..<newLength {
            for h in 0..<hiddenSize {
                let sourceIndex = (startFrame + t) * hiddenSize + h
                let destIndex = t * hiddenSize + h
                destPtr[destIndex] = sourcePtr[sourceIndex]
            }
        }

        return slicedArray
    }

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    /// Ideally this is not needed. We need to make some more fixes to the TDT decoding logic,
    /// this should be a temporary workaround.
    internal func removeDuplicateTokenSequence(
        previous: [Int], current: [Int], maxOverlap: Int = 12
    ) -> (deduped: [Int], removedCount: Int) {

        // Handle single punctuation token duplicates first
        let punctuationTokens = [7883, 7952, 7948]  // period, question, exclamation
        var workingCurrent = current
        var removedCount = 0

        if !previous.isEmpty && !workingCurrent.isEmpty && previous.last == workingCurrent.first
            && punctuationTokens.contains(workingCurrent.first!)
        {
            // Remove the duplicate punctuation token from the beginning of current
            workingCurrent = Array(workingCurrent.dropFirst())
            removedCount += 1
        }

        // Check for suffix-prefix overlap: end of previous matches beginning of current
        let maxSearchLength = min(15, previous.count)  // last 15 tokens of previous
        let maxMatchLength = min(maxOverlap, workingCurrent.count)  // first 12 tokens of current

        guard maxSearchLength >= 2 && maxMatchLength >= 2 else {
            return (workingCurrent, removedCount)
        }

        // Search for overlapping sequences from longest to shortest
        for overlapLength in (2...min(maxSearchLength, maxMatchLength)).reversed() {
            // Check if the last `overlapLength` tokens of previous match the first `overlapLength` tokens of current
            let prevSuffix = Array(previous.suffix(overlapLength))
            let currPrefix = Array(workingCurrent.prefix(overlapLength))

            if prevSuffix == currPrefix {
                logger.debug("Found exact suffix-prefix overlap of length \(overlapLength): \(prevSuffix)")
                let finalRemoved = removedCount + overlapLength
                return (Array(workingCurrent.dropFirst(overlapLength)), finalRemoved)
            }
        }

        // Extended search: look for partial overlaps within the sequences
        // Use boundary search frames from TDT config for NeMo-compatible alignment
        let boundarySearchFrames = config.tdtConfig.boundarySearchFrames
        for overlapLength in (2...min(maxSearchLength, maxMatchLength)).reversed() {
            let prevStart = max(0, previous.count - maxSearchLength)
            let prevEnd = previous.count - overlapLength + 1
            if prevEnd <= prevStart { continue }

            for startIndex in prevStart..<prevEnd {
                let prevSub = Array(previous[startIndex..<(startIndex + overlapLength)])
                let currEnd = max(0, workingCurrent.count - overlapLength + 1)

                // Use boundarySearchFrames to limit search window (NeMo tdt_search_boundary pattern)
                let searchLimit = min(boundarySearchFrames, currEnd)
                for currentStart in 0..<searchLimit {
                    let currSub = Array(workingCurrent[currentStart..<(currentStart + overlapLength)])
                    if prevSub == currSub {
                        logger.debug(
                            "Found duplicate sequence length=\(overlapLength) at currStart=\(currentStart): \(prevSub) (boundarySearch=\(boundarySearchFrames))"
                        )
                        let finalRemoved = removedCount + currentStart + overlapLength
                        return (Array(workingCurrent.dropFirst(currentStart + overlapLength)), finalRemoved)
                    }
                }
            }
        }

        return (workingCurrent, removedCount)
    }

    /// Calculate start frame offset for a sliding window segment (deprecated - now handled by timeJump)
    nonisolated internal func calculateStartFrameOffset(segmentIndex: Int, leftContextSeconds: Double) -> Int {
        // This method is deprecated as frame tracking is now handled by the decoder's timeJump mechanism
        // Kept for test compatibility
        return 0
    }

    // MARK: - Vocabulary Rescoring

    /// Apply vocabulary rescoring to an ASRResult using CTC-based constrained decoding.
    ///
    /// Runs CTC inference on the audio samples and applies vocabulary rescoring to correct
    /// misrecognized words. Returns an updated ASRResult with rescored text and populated
    /// `ctcDetectedTerms`/`ctcAppliedTerms` fields.
    ///
    /// - Parameters:
    ///   - result: The original ASRResult from transcription
    ///   - audioSamples: Audio samples used for CTC inference
    /// - Returns: An ASRResult with rescored text and CTC metadata, or the original result if rescoring was skipped
    internal func applyVocabularyRescoring(
        result: ASRResult, audioSamples: [Float]
    ) async -> ASRResult {
        guard let spotter = ctcSpotter,
            let rescorer = vocabularyRescorer,
            let vocab = customVocabulary,
            let tokenTimings = result.tokenTimings, !tokenTimings.isEmpty
        else {
            return result
        }

        do {
            let spotResult = try await spotter.computeLogProbabilities(audioSamples: audioSamples)

            let logProbs = spotResult.logProbs
            guard !logProbs.isEmpty else {
                logger.debug("Vocabulary rescoring skipped: no log probs from CTC")
                return result
            }

            let vocabConfig = vocabSizeConfig ?? ContextBiasingConstants.rescorerConfig(forVocabSize: 0)
            // Use the higher of the size-based default and the caller-specified threshold
            // so that CustomVocabularyContext.minSimilarity is respected when stricter.
            let effectiveMinSimilarity = max(vocabConfig.minSimilarity, vocab.minSimilarity)

            let rescoreOutput = rescorer.ctcTokenRescore(
                transcript: result.text,
                tokenTimings: tokenTimings,
                logProbs: logProbs,
                frameDuration: spotResult.frameDuration,
                cbw: vocabConfig.cbw,
                marginSeconds: 0.5,
                minSimilarity: effectiveMinSimilarity
            )

            guard rescoreOutput.wasModified else {
                return result
            }

            let detected = rescoreOutput.replacements.compactMap { $0.replacementWord }
            let applied = rescoreOutput.replacements.filter { $0.shouldReplace }.compactMap {
                $0.replacementWord
            }

            logger.info(
                "Vocabulary rescoring applied \(applied.count) replacement(s)"
            )

            return result.withRescoring(
                text: rescoreOutput.text,
                detected: detected.isEmpty ? nil : detected,
                applied: applied.isEmpty ? nil : applied
            )
        } catch {
            logger.warning("Vocabulary rescoring failed: \(error.localizedDescription)")
            return result
        }
    }

}
