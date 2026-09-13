import Foundation
import Security

/// Stores an optional Hugging Face access token in the Keychain (a generic-password
/// item) so downloads from a private fork of the model repos work on-device without
/// environment variables or `hf auth login`. Works on both macOS and iOS.
enum TokenStore {
    private static let service = "com.iliasaz.ChatterboxApp.huggingface"
    private static let account = "HF_TOKEN"

    /// Returns the saved token, or "" if none is stored.
    static func load() -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let token = String(data: data, encoding: .utf8) else {
            return ""
        }
        return token
    }

    /// Upserts the token (trimmed). An empty value deletes the stored token.
    static func save(_ token: String) {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(base as CFDictionary) // simplest upsert: clear then add

        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return }

        var add = base
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(add as CFDictionary, nil)
    }
}
