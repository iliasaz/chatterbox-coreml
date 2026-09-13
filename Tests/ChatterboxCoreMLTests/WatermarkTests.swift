import Testing
import Foundation
import PerthCoreML
@testable import ChatterboxCoreML

/// Offline coverage of the watermark seam.
@Suite("Watermarking")
struct WatermarkTests {
    @Test("NoWatermark is the identity")
    func noWatermarkIsIdentity() {
        let samples: [Float] = (0..<1000).map { sinf(Float($0) * 0.01) }
        #expect(NoWatermark().watermark(samples) == samples)
        #expect(NoWatermark().watermark([]) == [])
    }

    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("wm-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// The model-dir probe is what keeps an offline install watermarking: if it
    /// misses a `PerthEncoder` sitting right there, `makeOrFallback` falls through
    /// to a Hub fetch that cannot succeed offline, and audio ships unmarked.
    @Test("PerthEncoder is found in either shipping form")
    func findsEncoderInEitherForm() throws {
        for name in ["PerthEncoder.mlpackage", "PerthEncoder.mlmodelc"] {
            let dir = try tempDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            #expect(!PerthWatermark.hasPerthEncoder(in: dir))
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(name), withIntermediateDirectories: true)
            #expect(PerthWatermark.hasPerthEncoder(in: dir))
        }
    }

    @Test("A directory without a Perth encoder is not mistaken for one")
    func rejectsUnrelatedDirectory() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        // The chatterbox model dir's own packages must not read as a Perth one,
        // and neither must the decoder on its own.
        for name in ["T3LM.mlpackage", "S3Encoder.mlmodelc", "PerthDecoder.mlpackage"] {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        #expect(!PerthWatermark.hasPerthEncoder(in: dir))
    }
}

/// End-to-end: audio that leaves the pipeline is actually detectable as watermarked.
/// Opt-in — needs a chatterbox model dir **and** a Perth dir holding `PerthEncoder`
/// + `PerthDecoder` (the detector is what makes the assertion real):
///
/// ```
/// CHATTERBOX_MODEL_DIR=./out PERTH_MODEL_DIR=../perth-coreml/out swift test
/// ```
///
/// This pins the one genuinely load-bearing decision in the integration. The pipeline
/// watermarks **per chunk**, not once per utterance — it has to, because
/// `generateStream` hands each chunk to the caller before the next one exists — and
/// `perth-coreml`'s own docs warn that per-chunk gating is not the same operation as
/// marking the whole signal (`magmask` thresholds every frame against the loudest
/// frame *in the signal it is given*). Measured, it holds: a three-sentence utterance
/// scores 1.0 whole, and every third of it scores ≥0.9995 on its own. But a
/// regression here is silent — the audio still plays, it just stops carrying a mark —
/// so it gets a test rather than a comment.
@Suite("Watermarking end-to-end") struct WatermarkEndToEndTests {
    /// Both dirs, or `nil` to skip — matching how every other model-backed suite
    /// here opts in (`VoiceClonerTests`, `SynthEnvE2ETests`).
    private var dirs: (model: URL, perth: URL)? {
        let env = ProcessInfo.processInfo.environment
        guard let m = env["CHATTERBOX_MODEL_DIR"], let p = env["PERTH_MODEL_DIR"] else { return nil }
        return (URL(fileURLWithPath: m), URL(fileURLWithPath: p))
    }

    @Test("Generated audio detects as watermarked, and the opt-out really opts out")
    func generatedAudioIsWatermarked() async throws {
        guard let dirs else { return }
        // Two sentences + a pause forces more than one chunk, so this exercises the
        // per-chunk apply rather than a single whole-utterance one.
        let text = "The quick brown fox jumps over the lazy dog. "
                 + "Pack my box with five dozen liquor jugs."
        var options = GenerationOptions.default
        options.sentencePause = 0.25
        options.seed = 7

        let detector = try PerthWatermarker(modelDirectory: dirs.perth)

        let marked = try await ChatterboxCoreMLModel.load(
            from: dirs.model,
            watermarker: await PerthWatermark.makeOrFallback(localDirectory: dirs.perth))
        #expect(try detector.getWatermark(try await samples(of: marked, text, options),
                                          sampleRate: 24_000) == 1.0)

        let plain = try await ChatterboxCoreMLModel.load(
            from: dirs.model, watermarker: NoWatermark())
        #expect(try detector.getWatermark(try await samples(of: plain, text, options),
                                          sampleRate: 24_000) == 0.0)
    }

    private func samples(
        of model: ChatterboxCoreMLModel, _ text: String, _ options: GenerationOptions
    ) async throws -> [Float] {
        var out: [Float] = []
        for try await chunk in model.generateStream(text, options: options) {
            out.append(contentsOf: chunk.samples)
        }
        return out
    }
}
