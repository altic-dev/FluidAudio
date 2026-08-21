import Foundation

/// A monotonic view of a revisable ASR hypothesis.
@available(macOS 14.0, iOS 17.0, *)
public struct StabilizedTranscriptUpdate: Sendable, Equatable {
    /// Confirmed text followed by the latest unconfirmed tail.
    public let text: String
    /// Text that will no longer be retracted by subsequent updates.
    public let confirmedText: String
    /// Latest revisable suffix after the confirmed prefix.
    public let volatileText: String
    /// Token IDs confirmed by this update.
    public let newlyConfirmedTokenIDs: [Int]
    /// Stable tokens withheld until a word boundary or wait deadline.
    public let withheldStableTokenCount: Int

    public init(
        text: String,
        confirmedText: String,
        volatileText: String,
        newlyConfirmedTokenIDs: [Int],
        withheldStableTokenCount: Int
    ) {
        self.text = text
        self.confirmedText = confirmedText
        self.volatileText = volatileText
        self.newlyConfirmedTokenIDs = newlyConfirmedTokenIDs
        self.withheldStableTokenCount = withheldStableTokenCount
    }
}

/// Merges revisable `ASRResult` hypotheses into a monotonic confirmed prefix
/// plus a volatile tail. Keep one stabilizer per recording.
@available(macOS 14.0, iOS 17.0, *)
public actor ASRTranscriptStabilizer {
    private final class TokenVocabulary {
        var pieces: [Int: String] = [:]
    }

    private let streamID = 0
    private let vocabulary: TokenVocabulary
    private let emitter: StabilizedStreamingEmitter
    private var committedTokenIDs: [Int] = []
    private var latestTokenIDs: [Int] = []
    private var latestFallbackText = ""

    public init(config: StreamingStabilizerConfig = .lowLatency) {
        let vocabulary = TokenVocabulary()
        self.vocabulary = vocabulary
        emitter = StabilizedStreamingEmitter(
            config: config,
            tokenDecoder: { tokenID in vocabulary.pieces[tokenID] }
        )
    }

    /// Adds the latest full hypothesis and returns confirmed and volatile text.
    /// Results without token timings remain visible as volatile fallback text.
    public func merge(
        _ result: ASRResult,
        nowMilliseconds: Int? = nil
    ) -> StabilizedTranscriptUpdate {
        latestFallbackText = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let timings = result.tokenTimings, !timings.isEmpty else {
            return makeUpdate(
                volatileText: latestFallbackText,
                newlyConfirmedTokenIDs: [],
                withheldStableTokenCount: 0
            )
        }

        for timing in timings {
            vocabulary.pieces[timing.tokenId] = timing.token
        }
        latestTokenIDs = timings.map(\.tokenId)
        let emitted = emitter.update(
            uid: streamID,
            tokenIds: latestTokenIDs,
            nowMs: nowMilliseconds ?? Self.monotonicMilliseconds()
        )
        committedTokenIDs.append(contentsOf: emitted.committedTokens)
        return makeUpdate(
            volatileText: volatileText(),
            newlyConfirmedTokenIDs: emitted.committedTokens,
            withheldStableTokenCount: emitted.withheldStableTokenCount
        )
    }

    /// Commits the last hypothesis and returns the completed stabilized text.
    public func finish(nowMilliseconds: Int? = nil) -> StabilizedTranscriptUpdate {
        let emitted = emitter.flush(
            uid: streamID,
            nowMs: nowMilliseconds ?? Self.monotonicMilliseconds()
        )
        committedTokenIDs.append(contentsOf: emitted.committedTokens)
        return makeUpdate(
            volatileText: "",
            newlyConfirmedTokenIDs: emitted.committedTokens,
            withheldStableTokenCount: 0
        )
    }

    /// Clears all state so the instance can be reused for another recording.
    public func reset() {
        emitter.reset(uid: streamID)
        vocabulary.pieces.removeAll(keepingCapacity: true)
        committedTokenIDs.removeAll(keepingCapacity: true)
        latestTokenIDs.removeAll(keepingCapacity: true)
        latestFallbackText = ""
    }

    private func makeUpdate(
        volatileText: String,
        newlyConfirmedTokenIDs: [Int],
        withheldStableTokenCount: Int
    ) -> StabilizedTranscriptUpdate {
        let confirmedText = decode(committedTokenIDs)
        let text: String
        if latestTokenIDs.isEmpty {
            text = latestFallbackText
        } else {
            let pendingStart = min(committedTokenIDs.count, latestTokenIDs.count)
            let pendingTokenIDs = Array(latestTokenIDs.dropFirst(pendingStart))
            text = decode(committedTokenIDs + pendingTokenIDs)
        }
        return StabilizedTranscriptUpdate(
            text: text,
            confirmedText: confirmedText,
            volatileText: volatileText,
            newlyConfirmedTokenIDs: newlyConfirmedTokenIDs,
            withheldStableTokenCount: withheldStableTokenCount
        )
    }

    private func volatileText() -> String {
        let pendingStart = min(committedTokenIDs.count, latestTokenIDs.count)
        return decode(Array(latestTokenIDs.dropFirst(pendingStart)))
    }

    private func decode(_ tokenIDs: [Int]) -> String {
        tokenIDs
            .compactMap { vocabulary.pieces[$0] }
            .joined()
            .replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private nonisolated static func monotonicMilliseconds() -> Int {
        Int((ProcessInfo.processInfo.systemUptime * 1_000).rounded())
    }
}
