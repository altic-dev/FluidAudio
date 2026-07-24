import Foundation
import XCTest

@testable import FluidAudio

/// A stubbed `URLProtocol` that serves queued responses in order (falling back to a
/// default response once the queue is exhausted) and records every request it saw.
/// Used to exercise `DownloadUtils`' rate-limit retry/pagination logic without any
/// real network traffic.
final class StubURLProtocol: URLProtocol {

    struct StubResponse {
        let statusCode: Int
        let headers: [String: String]
        let data: Data
        /// When set, the protocol reports a transport-level failure (e.g. a timeout)
        /// instead of an HTTP response.
        let failureCode: URLError.Code?

        init(statusCode: Int, headers: [String: String] = [:], data: Data = Data()) {
            self.statusCode = statusCode
            self.headers = headers
            self.data = data
            self.failureCode = nil
        }

        init(failureCode: URLError.Code) {
            self.statusCode = 0
            self.headers = [:]
            self.data = Data()
            self.failureCode = failureCode
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var queuedResponses: [StubResponse] = []
    nonisolated(unsafe) private static var fallbackResponse = StubResponse(statusCode: 200)
    nonisolated(unsafe) private static var _recordedRequests: [URLRequest] = []

    static var recordedRequests: [URLRequest] {
        lock.withLock { _recordedRequests }
    }

    /// Queue responses to be returned in order, one per incoming request. Once the
    /// queue is exhausted, `fallback` is returned for any further requests.
    static func enqueue(_ responses: [StubResponse], fallback: StubResponse = StubResponse(statusCode: 200)) {
        lock.withLock {
            queuedResponses = responses
            fallbackResponse = fallback
        }
    }

    static func reset() {
        lock.withLock {
            queuedResponses = []
            fallbackResponse = StubResponse(statusCode: 200)
            _recordedRequests = []
        }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let stub = Self.lock.withLock { () -> StubResponse in
            Self._recordedRequests.append(self.request)
            guard !Self.queuedResponses.isEmpty else { return Self.fallbackResponse }
            return Self.queuedResponses.removeFirst()
        }

        if let failureCode = stub.failureCode {
            client?.urlProtocol(self, didFailWithError: URLError(failureCode))
            return
        }

        guard let url = request.url,
            let httpResponse = HTTPURLResponse(
                url: url, statusCode: stub.statusCode, httpVersion: "HTTP/1.1", headerFields: stub.headers)
        else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }

        client?.urlProtocol(self, didReceive: httpResponse, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class DownloadUtilsRateLimitTests: XCTestCase {

    override func setUp() {
        super.setUp()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubURLProtocol.self]
        DownloadUtils.sessionOverride = URLSession(configuration: configuration)
    }

    override func tearDown() {
        DownloadUtils.sessionOverride = nil
        StubURLProtocol.reset()
        super.tearDown()
    }

    private func makeTempDirectory() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("DownloadUtilsRateLimitTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func jsonData(_ items: [[String: Any]]) -> Data {
        (try? JSONSerialization.data(withJSONObject: items)) ?? Data()
    }

    private func treeRequests() -> [URLRequest] {
        StubURLProtocol.recordedRequests.filter { $0.url?.path.contains("/tree/") ?? false }
    }

    // MARK: - retryDelaySeconds

    func testRetryDelayUsesRateLimitHeaderTValue() {
        let response = HTTPURLResponse(
            url: URL(string: "https://huggingface.co/")!,
            statusCode: 429,
            httpVersion: nil,
            headerFields: ["ratelimit": "\"api\";r=0;t=7"]
        )
        let delay = DownloadUtils.retryDelaySeconds(from: response, attempt: 1, minBackoff: 1.0)
        XCTAssertEqual(delay, 7, accuracy: 0.001)
    }

    func testRetryDelayFallsBackToExponentialBackoffWithoutHeaders() {
        let delay = DownloadUtils.retryDelaySeconds(from: nil, attempt: 3, minBackoff: 1.0)
        XCTAssertEqual(delay, 4, accuracy: 0.001)  // pow(2, 2) * 1.0
    }

    func testRetryDelayClampsToFloor() {
        let response = HTTPURLResponse(
            url: URL(string: "https://huggingface.co/")!,
            statusCode: 429,
            httpVersion: nil,
            headerFields: ["ratelimit": "\"api\";r=0;t=0"]
        )
        let delay = DownloadUtils.retryDelaySeconds(from: response, attempt: 1, minBackoff: 1.0)
        XCTAssertEqual(delay, 0.5, accuracy: 0.001)
    }

    func testRetryDelayUsesRetryAfterHeaderWhenNoRateLimitHeader() {
        let response = HTTPURLResponse(
            url: URL(string: "https://huggingface.co/")!,
            statusCode: 429,
            httpVersion: nil,
            headerFields: ["Retry-After": "12"]
        )
        let delay = DownloadUtils.retryDelaySeconds(from: response, attempt: 1, minBackoff: 1.0)
        XCTAssertEqual(delay, 12, accuracy: 0.001)
    }

    // MARK: - nextLinkURL

    func testNextLinkURLParsesRelNext() {
        let response = HTTPURLResponse(
            url: URL(string: "https://huggingface.co/api/models/x/tree/main")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: ["Link": "<https://x/api?cursor=abc>; rel=\"next\""]
        )!
        XCTAssertEqual(DownloadUtils.nextLinkURL(from: response), URL(string: "https://x/api?cursor=abc"))
    }

    func testNextLinkURLReturnsNilWhenAbsent() {
        let response = HTTPURLResponse(
            url: URL(string: "https://huggingface.co/")!,
            statusCode: 200,
            httpVersion: nil,
            headerFields: nil
        )!
        XCTAssertNil(DownloadUtils.nextLinkURL(from: response))
    }

    // MARK: - Listing retries on rate limit, then succeeds

    func testDownloadSubdirectoryRetriesOnRateLimitThenSucceeds() async throws {
        StubURLProtocol.enqueue([
            .init(statusCode: 429, headers: ["ratelimit": "\"api\";r=0;t=0"]),
            .init(statusCode: 200, data: jsonData([["type": "file", "path": "a/b.bin", "size": 3]])),
            .init(statusCode: 200, data: Data([1, 2, 3])),
        ])

        let repoDirectory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: repoDirectory) }

        try await DownloadUtils.downloadSubdirectory(.parakeet, subdirectory: "extra", to: repoDirectory)

        XCTAssertEqual(treeRequests().count, 2)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent("a/b.bin").path))
    }

    // MARK: - Single-call recursive listing (no per-directory walk)

    func testListingMakesExactlyOneTreeRequestForNestedPaths() async throws {
        StubURLProtocol.enqueue(
            [
                .init(
                    statusCode: 200,
                    data: jsonData([
                        ["type": "directory", "path": "sub"],
                        ["type": "file", "path": "sub/nested/deep.bin", "size": 0],
                        ["type": "file", "path": "top.bin", "size": 0],
                    ]))
            ], fallback: .init(statusCode: 200))

        let repoDirectory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: repoDirectory) }

        try await DownloadUtils.downloadSubdirectory(.parakeet, subdirectory: "extra", to: repoDirectory)

        XCTAssertEqual(treeRequests().count, 1)
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent("sub/nested/deep.bin").path))
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent("top.bin").path))
    }

    // MARK: - Pagination follows Link rel="next" and unions results

    func testListingFollowsPaginationAndUnionsResults() async throws {
        StubURLProtocol.enqueue([
            .init(
                statusCode: 200,
                headers: ["Link": "<https://huggingface.co/api/models/x/tree/main?cursor=2>; rel=\"next\""],
                data: jsonData([["type": "file", "path": "p1.bin", "size": 0]])
            ),
            .init(statusCode: 200, data: jsonData([["type": "file", "path": "p2.bin", "size": 0]])),
        ])

        let repoDirectory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: repoDirectory) }

        try await DownloadUtils.downloadSubdirectory(.parakeet, subdirectory: "extra", to: repoDirectory)

        XCTAssertEqual(treeRequests().count, 2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent("p1.bin").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent("p2.bin").path))
    }

    // MARK: - Exhausted retries surface a rateLimited error

    func testListingExhaustedRetriesThrowsRateLimited() async {
        StubURLProtocol.enqueue([], fallback: .init(statusCode: 429, headers: ["ratelimit": "\"api\";r=0;t=0"]))

        let repoDirectory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: repoDirectory) }

        do {
            try await DownloadUtils.downloadSubdirectory(.parakeet, subdirectory: "extra", to: repoDirectory)
            XCTFail("Expected rateLimited error")
        } catch let DownloadUtils.HuggingFaceDownloadError.rateLimited(statusCode, message) {
            XCTAssertEqual(statusCode, 429)
            XCTAssertTrue(message.contains("Rate limited while listing files"))
        } catch {
            XCTFail("Expected rateLimited error, got \(error)")
        }
    }

    // MARK: - loadModels does not wipe cache on rate limit / network errors

    func testLoadModelsDoesNotWipeCacheOnRateLimit() async {
        StubURLProtocol.enqueue([], fallback: .init(statusCode: 429, headers: ["ratelimit": "\"api\";r=0;t=0"]))

        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let repoPath = directory.appendingPathComponent(Repo.parakeet.folderName)
        try? FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        let markerPath = repoPath.appendingPathComponent("marker.txt")
        FileManager.default.createFile(atPath: markerPath.path, contents: Data("marker".utf8))

        do {
            _ = try await DownloadUtils.loadModels(.parakeet, modelNames: [], directory: directory)
            XCTFail("Expected rateLimited error")
        } catch is DownloadUtils.HuggingFaceDownloadError {
            // expected
        } catch {
            XCTFail("Expected rateLimited error, got \(error)")
        }

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: markerPath.path),
            "Cache should not be wiped when the failure is a rate limit")
    }

    func testLoadModelsDoesNotWipeCacheOnNetworkTimeout() async {
        StubURLProtocol.enqueue([], fallback: .init(failureCode: .timedOut))

        let directory = makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let repoPath = directory.appendingPathComponent(Repo.parakeet.folderName)
        try? FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        let markerPath = repoPath.appendingPathComponent("marker.txt")
        FileManager.default.createFile(atPath: markerPath.path, contents: Data("marker".utf8))

        do {
            _ = try await DownloadUtils.loadModels(.parakeet, modelNames: [], directory: directory)
            XCTFail("Expected a URLError")
        } catch let error as URLError {
            XCTAssertEqual(error.code, .timedOut)
        } catch {
            XCTFail("Expected URLError.timedOut, got \(error)")
        }

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: markerPath.path),
            "Cache should not be wiped when the failure is a network error")
    }

    // MARK: - selectFilesToDownload (pure file-selection logic)

    func testSelectFilesToDownloadNonSubPathPatternsAndRootMetadata() {
        let tree: [(path: String, size: Int, type: String)] = [
            ("model_a/weights.mlmodelc", 10, "file"),
            ("model_a", 0, "directory"),
            ("model_b/weights.mlmodelc", 10, "file"),
            ("config.json", 5, "file"),
            ("README.txt", 3, "file"),
            ("notes.md", 3, "file"),
        ]
        let patterns = ["model_a/"]

        let selected = DownloadUtils.selectFilesToDownload(tree: tree, patterns: patterns, subPath: nil)
        let paths = Set(selected.map { $0.path })

        XCTAssertEqual(
            paths, ["model_a/weights.mlmodelc", "config.json", "README.txt"],
            "Only pattern-matched files plus root .json/.txt metadata should be selected")
    }

    func testSelectFilesToDownloadSubPathMetadataRule() {
        let tree: [(path: String, size: Int, type: String)] = [
            ("160ms/encoder.mlmodelc/model.mlmodel", 10, "file"),
            ("160ms/vocab.json", 2, "file"),
            ("160ms/tokenizer.model", 2, "file"),
            ("160ms/extra.bin", 2, "file"),
            ("160ms/readme.md", 2, "file"),
            ("320ms/encoder.mlmodelc/model.mlmodel", 10, "file"),
        ]
        let patterns = ["160ms/encoder.mlmodelc/"]

        let selected = DownloadUtils.selectFilesToDownload(tree: tree, patterns: patterns, subPath: "160ms")
        let paths = Set(selected.map { $0.path })

        XCTAssertEqual(
            paths,
            [
                "160ms/encoder.mlmodelc/model.mlmodel", "160ms/vocab.json", "160ms/tokenizer.model",
                "160ms/extra.bin",
            ],
            "Files inside subPath matching a pattern or .json/.model/.bin metadata should be selected;"
                + " files outside subPath or without a matching suffix should not")
        XCTAssertFalse(paths.contains("160ms/readme.md"))
        XCTAssertFalse(paths.contains("320ms/encoder.mlmodelc/model.mlmodel"))
    }

    func testSelectFilesToDownloadPrunesFilesUnderRejectedAncestorDirectory() {
        // "other_model/config.json" matches the root-level .json metadata rule by suffix,
        // but its ancestor directory "other_model" does not pass directoryPasses (it isn't
        // a required-model pattern), so the old per-directory walk would never have
        // descended into it. The flat-tree selector must reproduce that pruning.
        let tree: [(path: String, size: Int, type: String)] = [
            ("required_model/weights.mlmodelc", 10, "file"),
            ("other_model/config.json", 5, "file"),
            ("top_level.json", 5, "file"),
        ]
        let patterns = ["required_model/"]

        let selected = DownloadUtils.selectFilesToDownload(tree: tree, patterns: patterns, subPath: nil)
        let paths = Set(selected.map { $0.path })

        XCTAssertEqual(
            paths, ["required_model/weights.mlmodelc", "top_level.json"],
            "A file whose suffix matches but whose ancestor directory is pruned must be excluded")
        XCTAssertFalse(paths.contains("other_model/config.json"))
    }
}
