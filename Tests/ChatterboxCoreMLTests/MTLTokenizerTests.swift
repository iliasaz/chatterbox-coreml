import Testing
import Foundation
import Tokenizers
@testable import ChatterboxCoreML

/// Parity gate: the Swift `MTLTextTokenizer` must reproduce the
/// Python reference token-id stream byte-for-byte. Fixtures + the staged grapheme
/// tokenizer are generated from upstream's multilingual checkpoint into
/// `Tests/ChatterboxCoreMLTests/Fixtures/mtl-tokenizer/` (not committed — the
/// grapheme tokenizer JSON is an upstream model artifact). The test skips cleanly when they're absent, like the
/// model-dir-gated end-to-end tests.
struct MTLTokenizerTests {
    struct Expected: Decodable {
        struct Fixture: Decodable {
            let name: String
            let text: String
            let language: String?
            let ids: [Int]
            let ids_with_special: [Int]
        }
        let sot: Int
        let eot: Int
        let fixtures: [Fixture]
    }

    private static var fixturesDir: URL? {
        // .../Tests/ChatterboxCoreMLTests/MTLTokenizerTests.swift → Fixtures/mtl-tokenizer
        let dir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/mtl-tokenizer")
        let ok = FileManager.default.fileExists(atPath: dir.appendingPathComponent("expected.json").path)
        return ok ? dir : nil
    }

    @Test func loadsAndMatchesPythonReference() async throws {
        guard let dir = Self.fixturesDir else { return }   // skip without fixtures
        let expected = try JSONDecoder().decode(
            Expected.self,
            from: Data(contentsOf: dir.appendingPathComponent("expected.json")))

        let tok = try await MTLTextTokenizer(modelFolder: dir)

        #expect(MTLTextTokenizer.startTextToken == Int32(expected.sot))
        #expect(MTLTextTokenizer.stopTextToken == Int32(expected.eot))

        for fx in expected.fixtures {
            let ids = tok.encode(fx.text, language: fx.language)
            #expect(ids == fx.ids_with_special.map(Int32.init),
                    "fixture '\(fx.name)' (\(fx.text)) mismatch: got \(ids) want \(fx.ids_with_special)")
            let bare = tok.encodeBare(fx.text, language: fx.language)
            #expect(bare == fx.ids.map(Int32.init), "fixture '\(fx.name)' bare mismatch")
        }
    }

    /// `+` (manual stress) and an explicit U+0301 must normalize to the same ids.
    @Test func plusAndAcuteAreEquivalent() async throws {
        guard let dir = Self.fixturesDir else { return }
        let tok = try await MTLTextTokenizer(modelFolder: dir)
        #expect(tok.encode("мо+й", language: "ru") == tok.encode("мо\u{0301}й", language: "ru"))
    }

    /// NFKD must decompose `ё`→е+U+0308 and `й`→и+U+0306 in the preprocessed string.
    @Test func nfkdDecomposesRussianLetters() {
        // Build a tokenizer-free check on `preprocess` via a stub is overkill; use a
        // pure-Unicode assertion mirroring what `preprocess` relies on.
        let yo = "ё".decomposedStringWithCompatibilityMapping
        #expect(Array(yo.unicodeScalars.map { $0.value }) == [0x0435, 0x0308])
        let i_kratkoye = "й".decomposedStringWithCompatibilityMapping
        #expect(Array(i_kratkoye.unicodeScalars.map { $0.value }) == [0x0438, 0x0306])
    }

    @Test func dictionaryStressMarksKnownWordsOnly() {
        let dict = DictionaryRussianStress(table: ["дом": "до+м"])
        #expect(dict.stress("дом") == "до+м")
        #expect(dict.stress("кот") == "кот")              // OOV → unchanged
        #expect(dict.stress("до+м") == "до+м")            // already marked → unchanged
        #expect(dict.stress("мой дом.") == "мой до+м.")   // per-word, punctuation kept
    }
}
