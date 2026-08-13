import Foundation
import MachTaskSelfWrapper
import OSLog

#if canImport(Darwin)
import Darwin
#endif

/// System information utilities and one-time environment reporting.
public enum SystemInfo {
    // MARK: - Architecture detection

    public static var isAppleSilicon: Bool {
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }

    public static var isIntelMac: Bool {
        #if arch(x86_64)
        return true
        #else
        return false
        #endif
    }
    /// Collects a concise set of host and OS details.
    public static func summary() -> String {
        let pinfo = ProcessInfo.processInfo

        // OS name by platform
        #if os(macOS)
        let osName = "macOS"
        #elseif os(iOS)
        let osName = "iOS"
        #elseif os(tvOS)
        let osName = "tvOS"
        #elseif os(watchOS)
        let osName = "watchOS"
        #else
        let osName = "UnknownOS"
        #endif

        // Version (human-readable string typically includes build)
        let osVersionString = pinfo.operatingSystemVersionString

        // Architecture (compile-time slice used at runtime)
        let arch: String = {
            #if arch(arm64)
            return "arm64"
            #elseif arch(x86_64)
            return "x86_64"
            #elseif arch(arm)
            return "arm"
            #elseif arch(i386)
            return "i386"
            #else
            return "unknown"
            #endif
        }()

        // Chip / CPU brand if available
        let chip =
            sysctlString("machdep.cpu.brand_string")
            ?? sysctlString("hw.model")
            ?? "UnknownChip"

        // CPU and memory
        let coresTotal = pinfo.processorCount
        let coresActive = pinfo.activeProcessorCount
        let memBytes = pinfo.physicalMemory
        let mem = ByteCountFormatter.string(fromByteCount: Int64(memBytes), countStyle: .binary)

        // Rosetta translation status (best-effort)
        let translated = isRunningTranslated()

        return
            "\(osName) \(osVersionString), arch=\(arch), chip=\(chip), cores=\(coresActive)/\(coresTotal), mem=\(mem), rosetta=\(translated)"
    }

    /// Logs the environment exactly once per process using the provided logger.
    public static func logOnce(using logger: AppLogger) async {
        await SystemInfoReporter.shared.logIfNeeded(logger: logger)
    }

    /// Returns memory currently available to this process, if supported by the OS.
    public static func availableMemoryBytes() -> UInt64? {
        #if os(macOS)
        var statistics = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &statistics) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else { return nil }
        let availablePages = UInt64(statistics.free_count) + UInt64(statistics.inactive_count)
        return availablePages * UInt64(pageSize)
        #elseif os(iOS) || os(tvOS) || os(watchOS)
        UInt64(os_proc_available_memory())
        #else
        nil
        #endif
    }

    #if canImport(Darwin)
    /// Returns the current resident memory usage for this process, if available.
    public static func currentResidentMemoryBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count =
            mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size)
            / mach_msg_type_number_t(MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    get_current_task_port(),
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count)
            }
        }

        guard result == KERN_SUCCESS else { return nil }
        return info.resident_size
    }

    /// Returns the peak resident memory usage for this process, if available.
    public static func peakResidentMemoryBytes() -> UInt64? {
        var info = task_vm_info_data_t()
        var count =
            mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size)
            / mach_msg_type_number_t(MemoryLayout<natural_t>.size)

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    get_current_task_port(),
                    task_flavor_t(TASK_VM_INFO),
                    $0,
                    &count)
            }
        }

        guard result == KERN_SUCCESS else { return nil }
        return info.resident_size_peak
    }
    #endif

    // MARK: - Private helpers

    private static func sysctlString(_ key: String) -> String? {
        #if canImport(Darwin)
        var size: size_t = 0
        guard sysctlbyname(key, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [CChar](repeating: 0, count: Int(size))
        let result = buffer.withUnsafeMutableBufferPointer { ptr in
            sysctlbyname(key, ptr.baseAddress, &size, nil, 0)
        }
        guard result == 0 else { return nil }
        return String(cString: buffer)
        #else
        return nil
        #endif
    }

    private static func isRunningTranslated() -> Bool {
        #if canImport(Darwin)
        if #available(macOS 11.0, *) {
            var translated: Int32 = 0
            var size = MemoryLayout<Int32>.size
            if sysctlbyname("sysctl.proc_translated", &translated, &size, nil, 0) == 0 {
                return translated == 1
            }
        }
        #endif
        return false
    }
}

// MARK: - One-time reporter

actor SystemInfoReporter {
    static let shared = SystemInfoReporter()
    private var didLog = false

    func logIfNeeded(logger: AppLogger) {
        guard !didLog else { return }
        didLog = true
        logger.info("Host environment: \(SystemInfo.summary())")
    }
}
