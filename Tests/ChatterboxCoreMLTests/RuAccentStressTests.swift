import Testing
import Foundation
@testable import ChatterboxCoreML

/// Parity gate: the `RuAccentStress` adapter, fed text exactly as
/// `MTLTextTokenizer` would (lowercased + NFKD), must reproduce the ruaccent
/// team's verified §6 outputs (shown composed). Validates the NFC round-trip
/// (`ё`/`й` survive) + `.combiningAcute` notation + manual-mark preservation.
///
/// Gated on a local ruaccent model dir (`coreml/`,`dictpack/`,`nn/` — the
/// converter's `_work`). Override with `RUACCENT_WORK_DIR`; defaults to the
/// sibling checkout `../ruaccent-coreml/converter/_work`. Skips when absent.
struct RuAccentStressTests {
    private static var workDir: URL? {
        if let env = ProcessInfo.processInfo.environment["RUACCENT_WORK_DIR"] {
            let u = URL(fileURLWithPath: env)
            return FileManager.default.fileExists(atPath: u.appendingPathComponent("coreml").path) ? u : nil
        }
        // repoRoot/../ruaccent-coreml/converter/_work
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let u = repo.deletingLastPathComponent().appendingPathComponent("ruaccent-coreml/converter/_work")
        return FileManager.default.fileExists(atPath: u.appendingPathComponent("coreml").path) ? u : nil
    }

    /// (input, expected-after-stress) from issue #18 §6.
    private static let cases: [(String, String)] = [
        ("мой", "мо́й"),
        ("война", "война́"),
        ("самолёт", "самолёт"),          // ё preserved, no extra acute
        ("ёж", "ёж"),
        ("её ёлка под окном", "её ёлка по́д окно́м"),
        ("на двери висит замок", "на двери́ виси́т замо́к"),   // homograph → lock
        ("сл+ово тут", "сло́во тут"),     // manual + mark honored
    ]

    @Test func sanitySentencesMatchVerifiedOutputs() async throws {
        guard let dir = Self.workDir else { return }   // skip without the model dir
        let stresser = await RuAccentStress.makeOrFallback(localDirectory: dir)
        // If the dir is present it must actually load (not silently fall back).
        #expect(stresser is RuAccentStress, "ruaccent failed to load from \(dir.path)")
        guard stresser is RuAccentStress else { return }

        for (input, expected) in Self.cases {
            // Exactly what MTLTextTokenizer feeds the stresser: lowercase → NFKD.
            let fed = input.lowercased().decomposedStringWithCompatibilityMapping
            let got = stresser.stress(fed).precomposedStringWithCanonicalMapping
            #expect(got == expected.precomposedStringWithCanonicalMapping,
                    "‘\(input)’ → ‘\(got)’ (want ‘\(expected)’)")
        }
    }
}
