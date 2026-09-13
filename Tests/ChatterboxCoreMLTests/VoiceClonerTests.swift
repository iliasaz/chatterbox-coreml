import Testing
import Foundation
import CoreML
import Safetensors
@testable import ChatterboxCoreML

/// End-to-end parity for on-device voice cloning: run `VoiceCloner` on the same
/// reference Python used, compare each conditioning tensor to the committed
/// `*-conds.safetensors`. Only runs when `CHATTERBOX_MODEL_DIR` points at a dir
/// holding the 5 conditioning `.mlpackage`s (e.g. `./out`).
///
/// The fixture is the raw 24 kHz mono samples Python loads; the Swift path applies
/// its own BS.1770 loudness + AVAudioConverter resample, so this is a true end-to-end
/// check (the resampler/loudness differ slightly from librosa/pyloudnorm — f32
/// tensors are graded by cosine, tokens by exact-match rate).
///
/// Both fixtures derive from the first 15 s of Resemble AI's own
/// `prompts/female_random_podcast.wav` demo prompt — 15 s because that is
/// `ENC_COND_LEN`, so the clip covers everything the conditioning path reads while
/// keeping the fixture at 1.4 MB. The reference conds beside it are produced by the
/// **PyTorch** pipeline (upstream's `ChatterboxTurboTTS.prepare_conditionals`, batch dims
/// squeezed), which is the whole point: grading
/// Swift against a Swift-built file would assert nothing. Note this is deliberately
/// *not* the copy under `ChatterboxApp/…/Voices/`, which is Swift-built so that a
/// clone of this repo can regenerate the shipped voices without Python.
@Suite struct VoiceClonerTests {
    private var modelDir: URL? {
        ProcessInfo.processInfo.environment["CHATTERBOX_MODEL_DIR"].map(URL.init(fileURLWithPath:))
    }

    private func fixtureURL(_ name: String) -> URL? {
        let u = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().appendingPathComponent("Fixtures/\(name)")
        return FileManager.default.fileExists(atPath: u.path) ? u : nil
    }

    private func cos(_ a: [Float], _ b: [Float]) -> Float {
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    @Test func cloneMatchesPythonConds() async throws {
        guard let modelDir, let fx = fixtureURL("female_random_podcast_24k.npy"),
              let refURL = fixtureURL("female_random_podcast-conds.safetensors") else {
            print("VoiceClonerTests: skipped (need CHATTERBOX_MODEL_DIR + fixtures + reference conds)")
            return
        }
        let (samples, _) = try NPYFloat32.read(url: fx)
        let cloner = try await VoiceCloner(modelDirectory: modelDir)
        let conds = try cloner.makeConditionals(fromSamples24k: samples)

        let st = try Safetensors.read(at: refURL)
        let refSpk: [Float] = try st.array(forKey: "t3.speaker_emb")
        let refGenEmb: [Float] = try st.array(forKey: "gen.embedding")
        let refFeat: [Float] = try st.array(forKey: "gen.prompt_feat")
        let refGenTok: [Int32] = try st.array(forKey: "gen.prompt_token")
        let refCondTok: [Int32] = try st.array(forKey: "t3.cond_prompt_speech_tokens")

        let cSpk = cos(conds.speakerEmb, refSpk)
        let cGen = cos(conds.genEmbedding, refGenEmb)
        let cFeat = cos(conds.promptFeat, refFeat)
        func tokMatch(_ a: [Int32], _ b: [Int32]) -> Double {
            let n = min(a.count, b.count); guard n > 0 else { return 0 }
            var same = 0; for i in 0..<n where a[i] == b[i] { same += 1 }
            return Double(same) / Double(n)
        }
        let genTokPct = tokMatch(conds.promptTokens, refGenTok)
        let condTokPct = tokMatch(conds.condPromptSpeechTokens, refCondTok)

        print(String(format: "VoiceCloner parity: speaker_emb cos=%.4f  gen.embedding cos=%.4f  prompt_feat cos=%.4f",
                     cSpk, cGen, cFeat))
        print(String(format: "  gen.prompt_token %.1f%% (%d vs %d)  cond_tokens %.1f%% (%d vs %d)",
                     genTokPct * 100, conds.promptTokens.count, refGenTok.count,
                     condTokPct * 100, conds.condPromptSpeechTokens.count, refCondTok.count))

        #expect(cSpk > 0.99)
        #expect(cGen > 0.99)
        #expect(cFeat > 0.99)
    }
}
