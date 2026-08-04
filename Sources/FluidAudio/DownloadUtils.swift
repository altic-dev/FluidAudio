import CoreML
import Foundation
import OSLog
import os

/// HuggingFace model downloader using URLSession
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

    /// Test-only override for the session used by internal request helpers.
    /// `nonisolated(unsafe)` is acceptable here because it is a test seam: set once
    /// before any concurrent access (in `setUp`), cleared in `tearDown`, never mutated
    /// while requests are in flight.
    nonisolated(unsafe) internal static var sessionOverride: URLSession?

    /// Session used for all internal request/download traffic. Prefer this over
    /// `sharedSession` inside the type so tests can substitute a stub session.
    private static var session: URLSession { sessionOverride ?? sharedSession }

    /// Get HuggingFace token from environment or the HF CLI token file if available.
    /// Supports multiple env vars for compatibility with different HuggingFace tools:
    /// - HF_TOKEN: Official HuggingFace CLI
    /// - HUGGING_FACE_HUB_TOKEN: Python huggingface_hub library
    /// - HUGGINGFACEHUB_API_TOKEN: LangChain and older integrations
    /// - ~/.cache/huggingface/token: HF CLI login token file (used when no env var is set)
    /// Resolved once per process: a token added after launch is picked up on next start.
    private static let huggingFaceToken: String? = {
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGINGFACEHUB_API_TOKEN"]
        {
            return token
        }
        return tokenFromCLICacheFile
    }()

    /// Best-effort read of the HF CLI's cached login token. Never throws — sandboxed
    /// apps without access to the home directory simply get `nil`.
    private static var tokenFromCLICacheFile: String? {
        let tokenFileURL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/token")
        guard let contents = try? String(contentsOf: tokenFileURL, encoding: .utf8) else { return nil }
        let trimmed = contents.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// Create a URLRequest with optional auth header and timeout
    private static func authorizedRequest(
        url: URL, timeout: TimeInterval = DownloadConfig.default.timeout
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.setValue("FluidAudio-Swift", forHTTPHeaderField: "User-Agent")
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Fetch data from a URL with HuggingFace authentication if available
    /// Use this for API calls that need auth tokens for private repos or higher rate limits
    public static func fetchWithAuth(from url: URL) async throws -> (Data, URLResponse) {
        let request = authorizedRequest(url: url)
        return try await session.data(for: request)
    }

    /// Validate that response data is JSON, not HTML error page
    /// HuggingFace sometimes returns 200 OK with HTML error pages during rate limiting/timeouts
    private static func validateJSONResponse(_ data: Data, path: String) throws {
        // Check if response starts with HTML markers
        if let responseString = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
            if responseString.hasPrefix("<") || responseString.lowercased().contains("<!doctype html") {
                let snippet = String(responseString.prefix(100))
                throw HuggingFaceDownloadError.htmlErrorResponse(path: path, snippet: snippet)
            }
        }
    }

    // MARK: - Rate-limit-aware retry helpers

    /// Whether an HTTP status code indicates HuggingFace rate limiting.
    internal static func isRateLimitedStatus(_ code: Int) -> Bool {
        code == 429 || code == 503
    }

    /// Whether an HTTP status code indicates a transient server-side failure worth retrying.
    internal static func isTransientServerStatus(_ code: Int) -> Bool {
        code == 500 || code == 502 || code == 504
    }

    /// Compute the delay before the next retry attempt.
    ///
    /// Priority: the `ratelimit` response header's `t=<seconds>` field (HuggingFace's
    /// window-reset hint) → `Retry-After` → exponential backoff. Result is clamped to
    /// `[0.5, 300]` seconds.
    internal static func retryDelaySeconds(
        from response: HTTPURLResponse?, attempt: Int, minBackoff: TimeInterval
    ) -> TimeInterval {
        let delay: TimeInterval
        if let seconds = rateLimitResetSeconds(from: response) {
            delay = seconds
        } else if let retryAfter = response?.value(forHTTPHeaderField: "Retry-After"),
            let seconds = TimeInterval(retryAfter.trimmingCharacters(in: .whitespaces))
        {
            delay = seconds
        } else {
            delay = pow(2.0, Double(max(attempt - 1, 0))) * minBackoff
        }
        return min(max(delay, 0.5), 300)
    }

    /// Parse the `t=<seconds>` field out of the HuggingFace `ratelimit` header, e.g.
    /// `"api";r=0;t=280`.
    private static func rateLimitResetSeconds(from response: HTTPURLResponse?) -> TimeInterval? {
        guard let header = response?.value(forHTTPHeaderField: "ratelimit") else { return nil }
        for component in header.split(separator: ";") {
            let trimmed = component.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("t=") else { continue }
            return TimeInterval(trimmed.dropFirst(2))
        }
        return nil
    }

    /// Parse the RFC 5988 `Link` response header for a `rel="next"` pagination URL.
    internal static func nextLinkURL(from response: HTTPURLResponse) -> URL? {
        guard let linkHeader = response.value(forHTTPHeaderField: "Link") else { return nil }
        for part in linkHeader.split(separator: ",") {
            let segments = part.split(separator: ";").map { $0.trimmingCharacters(in: .whitespaces) }
            guard let urlSegment = segments.first, urlSegment.hasPrefix("<"), urlSegment.hasSuffix(">") else {
                continue
            }
            let isNext = segments.dropFirst().contains { segment in
                let normalized = segment.lowercased().replacingOccurrences(of: "\"", with: "")
                return normalized == "rel=next"
            }
            guard isNext else { continue }
            let urlString = String(urlSegment.dropFirst().dropLast())
            return URL(string: urlString)
        }
        return nil
    }

    /// Perform a data request, retrying on 429/503 (rate limit) and transient network
    /// errors with an appropriate backoff. Throws `HuggingFaceDownloadError.rateLimited`
    /// once attempts are exhausted while rate limited.
    private static func dataWithRetry(
        request: URLRequest,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> (Data, HTTPURLResponse) {
        var attempt = 1
        while true {
            do {
                let (data, response) = try await session.data(for: request)
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                let status = httpResponse.statusCode
                if isRateLimitedStatus(status) || isTransientServerStatus(status) {
                    if attempt < maxAttempts {
                        let delay = retryDelaySeconds(from: httpResponse, attempt: attempt, minBackoff: minBackoff)
                        logger.warning(
                            "HTTP \(status) while \(description), attempt \(attempt)/\(maxAttempts). Retrying in \(String(format: "%.1f", delay))s."
                        )
                        try Task.checkCancellation()
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        attempt += 1
                        continue
                    } else if isRateLimitedStatus(status) {
                        throw HuggingFaceDownloadError.rateLimited(
                            statusCode: status,
                            message: "Rate limited while \(description)")
                    }
                    // Exhausted retries on a transient 5xx: return the response and let the
                    // caller's status validation surface the failure.
                }

                return (data, httpResponse)
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as URLError where error.code == .cancelled {
                throw error
            } catch let error as HuggingFaceDownloadError {
                throw error
            } catch {
                if attempt < maxAttempts {
                    let delay = pow(2.0, Double(max(attempt - 1, 0))) * minBackoff
                    logger.warning(
                        "Request failed while \(description), attempt \(attempt)/\(maxAttempts): \(error.localizedDescription). Retrying in \(String(format: "%.1f", delay))s."
                    )
                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    attempt += 1
                    continue
                }
                throw error
            }
        }
    }

    /// Perform a download request, retrying on 429/503 (rate limit) and transient
    /// network errors with an appropriate backoff.
    private static func downloadWithRetry(
        request: URLRequest,
        description: String,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> (URL, HTTPURLResponse) {
        var attempt = 1
        while true {
            do {
                let tempFileURL: URL
                let httpResponse: HTTPURLResponse
                if let onProgress {
                    (tempFileURL, httpResponse) = try await downloadWithProgress(
                        request: request, onProgress: onProgress)
                } else {
                    let (url, response) = try await session.download(for: request)
                    guard let resp = response as? HTTPURLResponse else {
                        throw HuggingFaceDownloadError.invalidResponse
                    }
                    tempFileURL = url
                    httpResponse = resp
                }

                let status = httpResponse.statusCode
                if isRateLimitedStatus(status) || isTransientServerStatus(status) {
                    if attempt < maxAttempts {
                        let delay = retryDelaySeconds(from: httpResponse, attempt: attempt, minBackoff: minBackoff)
                        logger.warning(
                            "HTTP \(status) while downloading \(description), attempt \(attempt)/\(maxAttempts). Retrying in \(String(format: "%.1f", delay))s."
                        )
                        // Discard the (empty/error-body) temp file from this attempt before
                        // sleeping and retrying — nothing downstream will ever consume it.
                        try? FileManager.default.removeItem(at: tempFileURL)
                        try Task.checkCancellation()
                        try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                        attempt += 1
                        continue
                    } else if isRateLimitedStatus(status) {
                        try? FileManager.default.removeItem(at: tempFileURL)
                        throw HuggingFaceDownloadError.rateLimited(
                            statusCode: status,
                            message: "Rate limited while downloading \(description)")
                    }
                    // Exhausted retries on a transient 5xx: return the response and let the
                    // caller's status validation surface the failure.
                }

                return (tempFileURL, httpResponse)
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as URLError where error.code == .cancelled {
                throw error
            } catch let error as HuggingFaceDownloadError {
                throw error
            } catch {
                if attempt < maxAttempts {
                    let delay = pow(2.0, Double(max(attempt - 1, 0))) * minBackoff
                    logger.warning(
                        "Download failed while downloading \(description), attempt \(attempt)/\(maxAttempts): \(error.localizedDescription). Retrying in \(String(format: "%.1f", delay))s."
                    )
                    try Task.checkCancellation()
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                    attempt += 1
                    continue
                }
                throw error
            }
        }
    }

    /// Fetch and parse a full (paginated) HuggingFace repo tree listing in as few API
    /// calls as possible, using `recursive=true` instead of walking directories one at
    /// a time.
    private static func listRepoTree(
        remotePath: String, path: String
    ) async throws -> [(path: String, size: Int, type: String)] {
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        let baseURL = try ModelRegistry.apiModels(remotePath, apiPath)

        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            throw HuggingFaceDownloadError.invalidResponse
        }
        var queryItems = components.queryItems ?? []
        queryItems.append(URLQueryItem(name: "recursive", value: "true"))
        components.queryItems = queryItems

        guard var nextURL = components.url else {
            throw HuggingFaceDownloadError.invalidResponse
        }

        // Guards against a misbehaving/malicious server sending an unbounded or looping
        // pagination chain.
        let maxPages = 100
        var seenURLs: Set<URL> = []

        var results: [(path: String, size: Int, type: String)] = []
        var page = 0
        while true {
            page += 1
            guard page <= maxPages else {
                logger.error("Aborting pagination for \(path.isEmpty ? "root" : path) after \(maxPages) pages")
                throw HuggingFaceDownloadError.invalidResponse
            }
            seenURLs.insert(nextURL)

            let request = authorizedRequest(url: nextURL)
            let (data, httpResponse) = try await dataWithRetry(request: request, description: "listing files")

            try validateJSONResponse(data, path: path)

            guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw HuggingFaceDownloadError.invalidResponse
            }

            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }
                let itemSize = item["size"] as? Int ?? -1
                results.append((path: itemPath, size: itemSize, type: itemType))
            }

            guard let next = nextLinkURL(from: httpResponse), !seenURLs.contains(next) else { break }
            nextURL = next
        }

        return results
    }

    /// Pure file-selection logic for `downloadRepo`.
    ///
    /// Given a flat repo tree listing (as returned by `listRepoTree`), the required-model
    /// filter `patterns`, and the optional `subPath`, returns the files that should be
    /// downloaded. Reproduces two effects the old per-directory recursive walk had:
    ///
    /// - Directory pruning: a file is only included if every ancestor directory strictly
    ///   deeper than the listing root (`subPath`, or the repo root when `subPath` is nil)
    ///   passes `directoryPasses`.
    /// - File inclusion: the `subPath` branch requires the file be inside `subPath` and
    ///   either match a pattern or look like metadata (`.json`/`.model`/`.bin`); the
    ///   non-`subPath` branch matches a pattern or a `.json`/`.txt` suffix.
    ///
    /// Extracted as a standalone pure function (no I/O) so this selection logic can be
    /// unit tested directly against synthetic tree listings.
    internal static func selectFilesToDownload(
        tree: [(path: String, size: Int, type: String)],
        patterns: [String],
        subPath: String?
    ) -> [(path: String, size: Int)] {
        // Whether a directory path is allowed to be descended into / must have all its
        // files reachable. Mirrors the pruning a per-directory recursive walk used to do.
        func directoryPasses(_ dirPath: String) -> Bool {
            if let sub = subPath {
                return dirPath == sub || dirPath.hasPrefix("\(sub)/")
                    || patterns.contains { dirPath.hasPrefix($0) || $0.hasPrefix(dirPath + "/") }
            }
            return patterns.isEmpty || patterns.contains { dirPath.hasPrefix($0) || $0.hasPrefix(dirPath + "/") }
        }

        // Progressive ancestor-directory prefixes of a file path, excluding the file itself.
        func ancestorDirectories(of filePath: String) -> [String] {
            let components = filePath.split(separator: "/").map(String.init)
            guard components.count > 1 else { return [] }
            return (1..<components.count).map { components[0..<$0].joined(separator: "/") }
        }

        let rootDepth = subPath?.split(separator: "/").count ?? 0

        var filesToDownload: [(path: String, size: Int)] = []
        for entry in tree where entry.type == "file" {
            let ancestors = ancestorDirectories(of: entry.path)
            let isPruned = ancestors.contains { ancestor in
                let depth = ancestor.split(separator: "/").count
                guard depth > rootDepth else { return false }
                return !directoryPasses(ancestor)
            }
            guard !isPruned else { continue }

            // For subPath repos, only include files within the subPath
            let shouldInclude: Bool
            if let sub = subPath {
                let isInSubPath = entry.path.hasPrefix("\(sub)/")
                let matchesPattern =
                    patterns.isEmpty || patterns.contains { entry.path.hasPrefix($0) }
                let isMetadata =
                    entry.path.hasSuffix(".json") || entry.path.hasSuffix(".model") || entry.path.hasSuffix(".bin")
                shouldInclude = isInSubPath && (matchesPattern || isMetadata)
            } else {
                shouldInclude =
                    patterns.isEmpty || patterns.contains { entry.path.hasPrefix($0) }
                    || entry.path.hasSuffix(".json") || entry.path.hasSuffix(".txt")
            }
            if shouldInclude {
                filesToDownload.append((path: entry.path, size: entry.size))
            }
        }
        return filesToDownload
    }

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)
        case htmlErrorResponse(path: String, snippet: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            case .htmlErrorResponse(let path, let snippet):
                return "HuggingFace returned HTML instead of JSON for \(path) (rate limit or server issue): \(snippet)"
            case .modelNotFound(let path):
                return "Model file not found: \(path)"
            }
        }
    }

    /// Download configuration

    /// Phase of a model download operation.
    public enum DownloadPhase: Sendable {
        /// Listing files from the remote repository.
        case listing
        /// Downloading model files. `completedFiles` / `totalFiles` track per-file progress.
        case downloading(completedFiles: Int, totalFiles: Int)
        /// Compiling CoreML models after download.
        case compiling(modelName: String)
    }

    /// Progress snapshot passed to ``ProgressHandler`` closures.
    public struct DownloadProgress: Sendable {
        /// Fraction complete in [0, 1].
        public let fractionCompleted: Double
        /// Current phase of the operation.
        public let phase: DownloadPhase

        public init(fractionCompleted: Double, phase: DownloadPhase) {
            self.fractionCompleted = fractionCompleted
            self.phase = phase
        }
    }

    /// Callback type for download progress reporting.
    ///
    /// Called on an unspecified queue. If you need to update UI, dispatch to
    /// the main actor inside your handler.
    public typealias ProgressHandler = @Sendable (DownloadProgress) -> Void

    public struct DownloadConfig: Sendable {
        public let timeout: TimeInterval

        public init(timeout: TimeInterval = 1800) {  // 30 minutes for large models
            self.timeout = timeout
        }

        public static let `default` = DownloadConfig()
    }

    public static func loadModels(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        do {
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        } catch {
            guard !isTransientAndUnwipeable(error) else {
                logger.warning(
                    "First load failed with a transient/network error, not wiping cache: \(error.localizedDescription)"
                )
                throw error
            }

            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)
            try? FileManager.default.removeItem(at: repoPath)

            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        }
    }

    /// Errors where wiping the cache and retrying would just discard a resumable
    /// partial download for no benefit: cancellation, rate limiting, HTML error pages
    /// (usually rate-limit related), and any network-layer failure (offline, timeout, …).
    private static func isTransientAndUnwipeable(_ error: Error) -> Bool {
        if error is CancellationError {
            return true
        }
        if error is URLError {
            return true
        }
        if let downloadError = error as? HuggingFaceDownloadError {
            switch downloadError {
            case .rateLimited, .htmlErrorResponse:
                return true
            case .invalidResponse, .downloadFailed, .modelNotFound:
                return false
            }
        }
        return false
    }

    public static func clearModelCache(forRepo repo: Repo, directory: URL) {
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)
    }

    /// Remove all downloaded models and caches.
    ///
    /// Clears both cache locations:
    /// - `~/Library/Application Support/FluidAudio/Models/` (ASR, VAD, Diarization)
    /// - `~/.cache/fluidaudio/Models/` (TTS)
    public static func clearAllModelCaches() {
        let fm = FileManager.default

        // ASR, VAD, Diarization models
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupport.appendingPathComponent("FluidAudio/Models")
            try? fm.removeItem(at: modelsDir)
        }

        // TTS models (Kokoro, PocketTTS)
        #if os(macOS)
        let home = fm.homeDirectoryForCurrentUser
        let ttsCache = home.appendingPathComponent(".cache/fluidaudio/Models")
        try? fm.removeItem(at: ttsCache)
        #else
        if let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let ttsCache = cacheDir.appendingPathComponent("fluidaudio/Models")
            try? fm.removeItem(at: ttsCache)
        }
        #endif

        logger.info("All model caches cleared")
    }

    private static func loadModelsOnce(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
        let allModelsExist = requiredModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(repo, to: directory, variant: variant, progressHandler: progressHandler)
        } else {
            logger.info("Found \(repo.folderName) locally, no download needed")
            progressHandler?(
                DownloadProgress(fractionCompleted: 0.5, phase: .downloading(completedFiles: 0, totalFiles: 0)))
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        var models: [String: MLModel] = [:]
        for (index, name) in modelNames.enumerated() {
            let modelPath = repoPath.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw CocoaError(
                    .fileNoSuchFile,
                    userInfo: [
                        NSFilePathErrorKey: modelPath.path,
                        NSLocalizedDescriptionKey: "Model file not found: \(name)",
                    ])
            }

            var isDirectory: ObjCBool = false
            guard
                FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else {
                throw CocoaError(
                    .fileReadCorruptFile,
                    userInfo: [
                        NSFilePathErrorKey: modelPath.path,
                        NSLocalizedDescriptionKey: "Model path is not a directory: \(name)",
                    ])
            }

            let coremlDataPath = modelPath.appendingPathComponent("coremldata.bin")
            guard FileManager.default.fileExists(atPath: coremlDataPath.path) else {
                logger.error("Missing coremldata.bin in \(name)")
                throw CocoaError(
                    .fileReadCorruptFile,
                    userInfo: [
                        NSFilePathErrorKey: coremlDataPath.path,
                        NSLocalizedDescriptionKey: "Missing coremldata.bin in model: \(name)",
                    ])
            }

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: 0.5 + 0.5 * Double(index) / Double(modelNames.count),
                    phase: .compiling(modelName: name)
                ))

            let start = Date()
            let model = try MLModel(contentsOf: modelPath, configuration: config)
            let elapsed = Date().timeIntervalSince(start)

            models[name] = model

            let ms = elapsed * 1000
            let formatted = String(format: "%.2f", ms)
            logger.info("Compiled model \(name) in \(formatted) ms :: \(SystemInfo.summary())")
        }

        progressHandler?(DownloadProgress(fractionCompleted: 1.0, phase: .compiling(modelName: "")))
        return models
    }

    /// Download a HuggingFace repository using URLSession (does not load models)
    public static func downloadRepo(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
        let subPath = repo.subPath  // e.g., "160ms" for parakeetEou160

        // Build patterns for filtering (relative to subPath if present)
        var patterns: [String] = []
        for model in requiredModels {
            if let sub = subPath {
                patterns.append("\(sub)/\(model)/")
            } else {
                patterns.append("\(model)/")
            }
        }

        // Get all files recursively in a single (paginated) API call, then reproduce the
        // pruning + inclusion semantics of the old per-directory recursive walk.
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        let tree = try await listRepoTree(remotePath: repo.remotePath, path: subPath ?? "")

        let filesToDownload = selectFilesToDownload(tree: tree, patterns: patterns, subPath: subPath)
        logger.info("Found \(filesToDownload.count) files to download")

        // Compute total known bytes for byte-weighted progress.
        // Files with unknown sizes (size == -1) are treated as 0 for weighting.
        let totalBytes: Int64 = filesToDownload.reduce(0) { $0 + Int64(max(0, $1.size)) }
        var completedBytes: Int64 = 0
        // Highest download-phase fraction reported so far; keeps progress monotonic
        // across per-file retries (a retried attempt restarts its byte count at zero).
        let maxReportedFraction = OSAllocatedUnfairLock(initialState: 0.0)

        // Download each file
        for (index, file) in filesToDownload.enumerated() {
            // Strip subPath prefix when saving locally
            var localPath = file.path
            if let sub = subPath, file.path.hasPrefix("\(sub)/") {
                localPath = String(file.path.dropFirst(sub.count + 1))
            }
            let destPath = repoPath.appendingPathComponent(localPath)

            // Skip if already exists
            if FileManager.default.fileExists(atPath: destPath.path) {
                completedBytes += Int64(max(0, file.size))
                continue
            }

            // Create parent directory
            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            // HuggingFace returns 500 for 0-byte files — create empty file locally
            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                continue
            }

            // Download file (use original path for HuggingFace URL)
            let encodedFilePath =
                file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedFilePath)
            let request = authorizedRequest(url: fileURL)

            let tempFileURL: URL
            let httpResponse: HTTPURLResponse

            if let handler = progressHandler {
                let baseBytes = completedBytes
                let fileCount = filesToDownload.count
                let totalBytesSnapshot = totalBytes
                (tempFileURL, httpResponse) = try await downloadWithRetry(
                    request: request,
                    description: file.path,
                    onProgress: { bytesWritten, _ in
                        guard totalBytesSnapshot > 0 else { return }
                        let current = baseBytes + bytesWritten
                        // Download phase occupies 0.0–0.5 of the overall range. Clamp to the
                        // highest fraction reported so far: a retried attempt restarts its
                        // byte count at zero and must not rewind the visible progress.
                        let fraction = min(0.5 * Double(current) / Double(totalBytesSnapshot), 0.5)
                        let monotonic = maxReportedFraction.withLock { maxSoFar in
                            maxSoFar = max(maxSoFar, fraction)
                            return maxSoFar
                        }
                        handler(
                            DownloadProgress(
                                fractionCompleted: monotonic,
                                phase: .downloading(completedFiles: index, totalFiles: fileCount)
                            ))
                    }
                )
            } else {
                (tempFileURL, httpResponse) = try await downloadWithRetry(
                    request: request, description: file.path, onProgress: nil)
            }

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw HuggingFaceDownloadError.downloadFailed(
                    path: file.path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            // Move downloaded file to destination
            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempFileURL, to: destPath)

            completedBytes += Int64(max(0, file.size))

            if (index + 1) % 10 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) files")
            }

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: totalBytes > 0
                        ? 0.5 * Double(completedBytes) / Double(totalBytes)
                        : 0.5 * Double(index + 1) / Double(filesToDownload.count),
                    phase: .downloading(completedFiles: index + 1, totalFiles: filesToDownload.count)
                ))
        }

        // Verify required models are present
        for model in requiredModels {
            let modelPath = repoPath.appendingPathComponent(model)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw HuggingFaceDownloadError.modelNotFound(path: model)
            }
        }

        logger.info("Downloaded all required models for \(repo.folderName)")
    }

    // MARK: - Delegate-based download with per-byte progress

    /// Download a single file using a delegate to get byte-level progress.
    ///
    /// This is a pure transport helper — the caller is responsible for validating
    /// the HTTP status and moving the temporary file to its final destination.
    ///
    /// - Parameters:
    ///   - request: The URLRequest to download.
    ///   - onProgress: Called with `(totalBytesWritten, totalBytesExpected)` as data arrives.
    /// - Returns: The temporary file URL and HTTP response.
    private static func downloadWithProgress(
        request: URLRequest,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> (URL, HTTPURLResponse) {
        let taskHolder = DownloadTaskHolder()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let delegate = DownloadProgressDelegate(
                    onProgress: onProgress,
                    completion: { continuation.resume(with: $0) }
                )
                let delegateSession = URLSession(
                    configuration: session.configuration,
                    delegate: delegate,
                    delegateQueue: nil
                )
                delegate.session = delegateSession

                let task = delegateSession.downloadTask(with: request)
                taskHolder.setTask(task)
                task.resume()
            }
        } onCancel: {
            taskHolder.cancel()
        }
    }

    /// Download a specific subdirectory from a HuggingFace repository.
    ///
    /// Use this for optional model components that aren't part of the required model set
    /// (e.g., the Mimi encoder for PocketTTS voice cloning).
    ///
    /// - Parameters:
    ///   - repo: The HuggingFace repository.
    ///   - subdirectory: Path within the repo to download (e.g. `"mimi_encoder.mlmodelc"`).
    ///   - repoDirectory: Local directory corresponding to the repo root.
    ///     Files are saved at `repoDirectory/<remote_path>`.
    public static func downloadSubdirectory(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL
    ) async throws {
        let tree = try await listRepoTree(remotePath: repo.remotePath, path: subdirectory)
        let filesToDownload: [(path: String, size: Int)] =
            tree
            .filter { $0.type == "file" }
            .map { (path: $0.path, size: $0.size) }
        logger.info("Found \(filesToDownload.count) files in \(subdirectory)")

        for (index, file) in filesToDownload.enumerated() {
            let destPath = repoDirectory.appendingPathComponent(file.path)

            if FileManager.default.fileExists(atPath: destPath.path) {
                continue
            }

            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                continue
            }

            let encodedPath =
                file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedPath)
            let request = authorizedRequest(url: fileURL)

            let (tempURL, httpResponse) = try await downloadWithRetry(
                request: request, description: file.path, onProgress: nil)

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw HuggingFaceDownloadError.downloadFailed(
                    path: file.path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempURL, to: destPath)

            if (index + 1) % 5 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) \(subdirectory) files")
            }
        }

        logger.info("Downloaded \(subdirectory) from \(repo.folderName)")
    }

    /// Fetch a single file from HuggingFace with retry
    ///
    /// Rate-limit (429/503) retries and transient network-error retries both happen
    /// inside `dataWithRetry`, sharing the full `maxAttempts` budget so 429s use
    /// HuggingFace's header-provided delay instead of blind exponential backoff.
    public static func fetchHuggingFaceFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> Data {
        let request = authorizedRequest(url: url)

        let (data, httpResponse) = try await dataWithRetry(
            request: request, description: description, maxAttempts: maxAttempts, minBackoff: minBackoff)

        guard (200..<300).contains(httpResponse.statusCode) else {
            throw HuggingFaceDownloadError.invalidResponse
        }

        return data
    }
}

// MARK: - URLSession download delegate for byte-level progress

private final class DownloadTaskHolder: @unchecked Sendable {
    private let lock = NSLock()
    private var task: URLSessionDownloadTask?
    private var isCancelled = false

    func setTask(_ task: URLSessionDownloadTask) {
        lock.withLock {
            if self.isCancelled {
                task.cancel()
            } else {
                self.task = task
            }
        }
    }

    func cancel() {
        lock.withLock {
            self.isCancelled = true
            self.task?.cancel()
        }
    }
}

/// Delegate-backed downloads are required for continuous `didWriteData` events.
/// Foundation's async download convenience only surfaced file-completion updates here.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    typealias Completion = @Sendable (Result<(URL, HTTPURLResponse), Error>) -> Void

    private let onProgress: @Sendable (Int64, Int64) -> Void
    private let lock = NSLock()
    private var completion: Completion?
    private var downloadResult: Result<(URL, HTTPURLResponse), Error>?
    weak var session: URLSession?

    init(
        onProgress: @escaping @Sendable (Int64, Int64) -> Void,
        completion: @escaping Completion
    ) {
        self.onProgress = onProgress
        self.completion = completion
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let response = downloadTask.response as? HTTPURLResponse else {
            storeResult(.failure(DownloadUtils.HuggingFaceDownloadError.invalidResponse))
            return
        }

        do {
            let retainedURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("FluidAudioDownload-\(UUID().uuidString)")
            try FileManager.default.copyItem(at: location, to: retainedURL)
            storeResult(.success((retainedURL, response)))
        } catch {
            storeResult(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            finish(.failure(error))
        } else {
            let result = lock.withLock { self.downloadResult }
            finish(result ?? .failure(DownloadUtils.HuggingFaceDownloadError.invalidResponse))
        }
        self.session?.finishTasksAndInvalidate()
    }

    private func storeResult(_ result: Result<(URL, HTTPURLResponse), Error>) {
        lock.withLock {
            self.downloadResult = result
        }
    }

    private func finish(_ result: Result<(URL, HTTPURLResponse), Error>) {
        let completion = lock.withLock { () -> Completion? in
            defer { self.completion = nil }
            return self.completion
        }
        completion?(result)
    }
}
