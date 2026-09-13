import Foundation

/// Persists a user-chosen folder across launches.
///
/// On iOS the app is sandboxed and the document picker hands back a
/// *security-scoped* URL that is invalid on the next launch unless we store a
/// security-scoped **bookmark** and re-resolve it. On (non-sandboxed) macOS a
/// plain path is enough — the app has full filesystem access.
enum FolderBookmark {
    static func store(_ url: URL, forKey key: String) {
        let defaults = UserDefaults.standard
        #if os(macOS)
        defaults.set(url.path, forKey: key)
        #else
        // Create the bookmark while the picker's access is still active.
        if let data = try? url.bookmarkData(
            options: [], includingResourceValuesForKeys: nil, relativeTo: nil) {
            defaults.set(data, forKey: key)
        }
        #endif
    }

    /// Resolves the saved folder, starting security-scoped access on iOS.
    static func resolve(forKey key: String) -> URL? {
        let defaults = UserDefaults.standard
        #if os(macOS)
        guard let path = defaults.string(forKey: key) else { return nil }
        let url = URL(fileURLWithPath: path, isDirectory: true)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
        #else
        guard let data = defaults.data(forKey: key) else { return nil }
        var stale = false
        guard let url = try? URL(
            resolvingBookmarkData: data, options: [], relativeTo: nil,
            bookmarkDataIsStale: &stale) else { return nil }
        _ = url.startAccessingSecurityScopedResource() // held for app lifetime
        return url
        #endif
    }

    static func clear(forKey key: String) {
        UserDefaults.standard.removeObject(forKey: key)
    }
}
