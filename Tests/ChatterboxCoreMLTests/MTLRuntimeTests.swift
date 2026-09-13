import Testing
import Foundation
import CoreML
@testable import ChatterboxCoreML

/// Parity gate: the Swift `MTLT3LMRunner` (host-assembled batch-2 CFG
/// prefill + greedy decode, using the reproduced sub-networks) must match a
/// Python driver of the SAME multilingual `T3LM.mlpackage`.
/// Validates the cond block (Perceiver), text/BOS assembly, masks, CFG-combine,
/// per-step speech_pos_emb, and the KV-state decode loop end-to-end on Mac.
///
/// Gated on `out-mtl/` + the runtime fixtures (both gitignored). Skips otherwise.
struct MTLRuntimeTests {
    private static var repo: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private static var paths: (model: URL, tables: URL, fixtures: URL)? {
        let model = repo.appendingPathComponent("out-mtl/T3LM.mlpackage")
        let tables = repo.appendingPathComponent("out-mtl/host_tables")
        let fixtures = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Fixtures/mtl-runtime")
        let fm = FileManager.default
        guard fm.fileExists(atPath: model.path),
              fm.fileExists(atPath: tables.appendingPathComponent("perceiver_query.npy").path),
              fm.fileExists(atPath: fixtures.appendingPathComponent("seed_logits.npy").path)
        else { return nil }
        return (model, tables, fixtures)
    }

    private func f(_ dir: URL, _ name: String) throws -> [Float] {
        try NPYFloat32.read(url: dir.appendingPathComponent("\(name).npy")).data
    }
    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<min(a.count, b.count) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    @Test func greedyDecodeMatchesPythonDriver() async throws {
        guard let p = Self.paths else { return }   // skip without out-mtl + fixtures
        let fx = p.fixtures

        let speaker = try f(fx, "speaker_emb")
        let promptTokens = try f(fx, "prompt_tokens").map { Int32($0) }
        let textTokens = try f(fx, "text_tokens").map { Int32($0) }
        let emotion = try f(fx, "emotion").first ?? 0.5
        let expectedSeed = try f(fx, "seed_logits")
        let expectedTokens = try f(fx, "greedy_tokens").map { Int($0) }

        let conds = Conditionals(
            speakerEmb: speaker,
            condPromptSpeechTokens: promptTokens,
            genEmbedding: [Float](repeating: 0, count: Constants.camppEmbDim),
            promptTokens: [],
            promptFeat: [])

        let compiled = try await MLModel.compileModel(at: p.model)
        let runner = try MTLT3LMRunner(contentsOf: compiled, hostTablesDir: p.tables)

        // Greedy, no repetition penalty, no early stop — pure argmax to match the
        // reference driver.
        let opts = GenerationOptions(
            repetitionPenalty: 1.0, minTokens: 0, maxTokens: expectedTokens.count,
            greedy: true, cfgWeight: 0.5, exaggeration: emotion)
        let out = try runner.generate(textTokens: textTokens, conds: conds, options: opts)

        let seedCos = cosine(out.seedLogits, expectedSeed)
        #expect(seedCos >= 0.999, "seed logits cos \(seedCos)")
        #expect(out.seedLogits.count == MultilingualConstants.speechVocabSize)
        // Same model + near-identical fp16 inputs → identical greedy argmax path.
        #expect(out.tokens == expectedTokens, "tokens \(out.tokens) != \(expectedTokens)")
    }

    /// Full integrated pipeline: `ChatterboxCoreMLModel` auto-detects the
    /// multilingual variant and renders Russian end-to-end (tokenizer →
    /// MTLT3LMRunner → synth). Opt-in via `CHATTERBOX_MTL_MODEL_DIR` pointing at a
    /// flat multilingual bundle (e.g. `out-mtl-full`); skips otherwise.
    @Test func multilingualEndToEndGeneratesAudio() async throws {
        guard let dir = ProcessInfo.processInfo.environment["CHATTERBOX_MTL_MODEL_DIR"]
            .map(URL.init(fileURLWithPath:)) else { return }
        let model = try await ChatterboxCoreMLModel.load(from: dir, watermarker: NoWatermark())
        let buffer = try await model.generate(
            "Привет.",
            options: .multilingual(language: "ru", exaggeration: 0.5, maxTokens: 120))
        #expect(buffer.format.sampleRate == Constants.sampleRate)
        #expect(buffer.frameLength > 0)
    }
}
