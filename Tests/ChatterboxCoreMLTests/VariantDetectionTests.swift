import Testing
import Foundation
@testable import ChatterboxCoreML

/// `ModelRepository.detectVariant` is the **pre-load** variant rule: callers (the CLI,
/// the app) preset the sampler / tokenizer / stress from it *before* `load` runs, so it
/// must agree with the variant `load` will report — a disagreement pairs a model with
/// another model's sampler. The load-side rule is cross-checked against the shipped
/// graph in `EndToEndTests`; these cases pin the file-level rule offline.
struct VariantDetectionTests {
    /// A model dir holding just the files detection reads. `hidden` sizes `speech_emb`
    /// (the T3 width); the row count is irrelevant to the rule, so 2 rows keep it small.
    private func modelDir(hidden: Int?, perceiver: Bool) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("variant_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let hidden {
            try writeNPY(Array(repeating: 0, count: 2 * hidden), shape: [2, hidden],
                         to: dir.appendingPathComponent("speech_emb.npy"))
        }
        if perceiver {
            try writeNPY([0], shape: [1], to: dir.appendingPathComponent("perceiver_query.npy"))
        }
        return dir
    }

    @Test func turboAndNanoAreToldApartByTheT3Width() throws {
        let turbo = try modelDir(hidden: Constants.gpt2Hidden, perceiver: false)
        let nano = try modelDir(hidden: NanoConstants.hidden, perceiver: false)
        defer { try? FileManager.default.removeItem(at: turbo); try? FileManager.default.removeItem(at: nano) }
        #expect(ModelRepository.detectVariant(in: turbo) == .turbo)
        #expect(ModelRepository.detectVariant(in: nano) == .nano)
    }

    /// The Perceiver cond block marks multilingual — and wins over the width, which is
    /// 1024 there too (`MultilingualConstants.hidden`) and would otherwise read as turbo.
    @Test func perceiverMarkerWinsOverTheWidth() throws {
        let dir = try modelDir(hidden: MultilingualConstants.hidden, perceiver: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelRepository.detectVariant(in: dir) == .multilingual)
    }

    /// No tables ⇒ no claim. Callers keep their own selection (and `load` then reports
    /// the missing file), rather than being coerced to a variant by an empty directory.
    @Test func directoryWithNoModelNamesNoVariant() throws {
        let dir = try modelDir(hidden: nil, perceiver: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelRepository.detectVariant(in: dir) == nil)
    }

    /// Detection reads the npy header only, so it never pulls a 27 MB table into memory.
    @Test func shapeReadsTheHeaderNotThePayload() throws {
        let dir = try modelDir(hidden: NanoConstants.hidden, perceiver: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        let emb = dir.appendingPathComponent("speech_emb.npy")
        let full = try Data(contentsOf: emb)
        #expect(NPYFloat32.shape(url: emb) == [2, NanoConstants.hidden])
        // Strip every payload byte (npy pads the header to a 64-byte boundary, so the
        // first 128 here are header): the probe still reads the shape off the header.
        try full.prefix(128).write(to: emb)
        #expect(full.count > 128 + 2 * NanoConstants.hidden * 3)   // payload really was there
        #expect(NPYFloat32.shape(url: emb) == [2, NanoConstants.hidden])
    }
}
