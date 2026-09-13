import Testing
import Foundation
@testable import ChatterboxCoreML

/// Parity gate: the host-side `Perceiver` + `CondBlockBuilder` must
/// reproduce the PyTorch `T3.prepare_conditioning` 34-row cond block. Fixtures are
/// generated from upstream's multilingual checkpoint into
/// `Tests/ChatterboxCoreMLTests/Fixtures/mtl-cond/` (not committed).
/// Skips cleanly when absent.
struct CondBlockTests {
    private static var dir: URL? {
        let d = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/mtl-cond")
        return FileManager.default.fileExists(atPath: d.appendingPathComponent("expected_cond.npy").path) ? d : nil
    }

    private func npy(_ dir: URL, _ name: String) throws -> (data: [Float], shape: [Int]) {
        try NPYFloat32.read(url: dir.appendingPathComponent("\(name).npy"))
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        var m: Float = 0
        for i in 0..<a.count { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    @Test func perceiverMatchesPyTorch() throws {
        guard let dir = Self.dir else { return }
        let perceiver = try Perceiver(dir: dir)
        let (prompt, pShape) = try npy(dir, "cond_prompt_speech_emb")   // (N, 1024)
        let n = pShape[0]
        let (expected, _) = try npy(dir, "expected_perceiver")          // (32, 1024)

        let out = perceiver.forward(prompt, n: n)
        #expect(out.count == Perceiver.queryLen * Perceiver.dim)
        let cos = cosine(out, expected)
        #expect(cos >= 0.9999, "perceiver cos \(cos) (maxabsdiff \(maxAbsDiff(out, expected)))")
    }

    @Test func condBlockMatchesPyTorch() throws {
        guard let dir = Self.dir else { return }
        let builder = try CondBlockBuilder(dir: dir)
        let (speaker, _) = try npy(dir, "speaker_emb")                  // (256,)
        let (prompt, pShape) = try npy(dir, "cond_prompt_speech_emb")   // (N, 1024)
        let (emotionArr, _) = try npy(dir, "emotion")                   // scalar
        let (expected, eShape) = try npy(dir, "expected_cond")          // (34, 1024)

        let block = builder.build(
            speakerEmb: speaker,
            condPromptSpeechEmb: prompt,
            promptLen: pShape[0],
            exaggeration: emotionArr.first ?? 0.5)

        #expect(eShape == [CondBlockBuilder.rows, CondBlockBuilder.dim])
        #expect(block.count == expected.count)
        let cos = cosine(block, expected)
        #expect(cos >= 0.9999, "cond-block cos \(cos) (maxabsdiff \(maxAbsDiff(block, expected)))")

        // Per-row cosine: catch a single bad row (spkr / a perceiver row / emotion)
        // that a global cosine could mask.
        let D = CondBlockBuilder.dim
        for r in 0..<CondBlockBuilder.rows {
            let lo = r * D, hi = lo + D
            let rc = cosine(Array(block[lo..<hi]), Array(expected[lo..<hi]))
            #expect(rc >= 0.999, "cond-block row \(r) cos \(rc)")
        }
    }
}
