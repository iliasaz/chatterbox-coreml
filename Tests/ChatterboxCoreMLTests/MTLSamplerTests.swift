import Testing
import Foundation
@testable import ChatterboxCoreML

/// The multilingual min-p sampler branch (no top-k). Deterministic
/// checks on which tokens survive the `minP · maxProb` floor — the stochastic
/// draw itself isn't asserted (torch.multinomial RNG can't be matched exactly).
struct MTLSamplerTests {
    /// A high min-p that leaves only the dominant token must collapse to argmax,
    /// regardless of seed.
    @Test func highMinPCollapsesToTopToken() {
        var logits = [Float](repeating: -10, count: MultilingualConstants.speechVocabSize)
        logits[42] = 20      // overwhelmingly dominant → prob ≈ 1
        logits[7] = 0        // small
        let opts = GenerationOptions.multilingual(language: "ru", minP: 0.5, seed: 1)
        var sampler = Sampler(options: opts)
        for _ in 0..<20 {
            let t = sampler.sample(logits: logits, previous: [], minTokensReached: true)
            #expect(t == 42)
        }
    }

    /// A low min-p keeps a wider set: the sampled token may be the runner-up, but
    /// never a token below the floor.
    @Test func lowMinPKeepsRunnersUpButNotFloorRejects() {
        var logits = [Float](repeating: -100, count: MultilingualConstants.speechVocabSize)
        logits[1] = 2.0      // top
        logits[2] = 1.9      // close runner-up — above a 0.05 floor
        logits[3] = -100     // far below floor → must never be sampled
        let opts = GenerationOptions.multilingual(language: "ru", temperature: 1.0, minP: 0.05, seed: 7)
        var sampler = Sampler(options: opts)
        var seen = Set<Int>()
        for _ in 0..<50 {
            seen.insert(sampler.sample(logits: logits, previous: [], minTokensReached: true))
        }
        #expect(seen.isSubset(of: [1, 2]))
        #expect(!seen.contains(3))
    }

    /// The multilingual preset drops top-k and turns on min-p.
    @Test func multilingualPresetShape() {
        let o = GenerationOptions.multilingual(language: "ru", exaggeration: 1.3)
        #expect(o.topK == 0)
        #expect(o.minP == 0.05)
        #expect(o.language == "ru")
        #expect(o.exaggeration == 1.3)
        #expect(o.cfgWeight == 0.5)
    }
}
