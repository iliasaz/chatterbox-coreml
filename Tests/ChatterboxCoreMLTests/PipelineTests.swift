import Testing
import Foundation
import CoreML
@testable import ChatterboxCoreML

/// Writes a minimal little-endian Float32 v1.0 `.npy` file (C-order).
func writeNPY(_ values: [Float], shape: [Int], to url: URL) throws {
    var header = "{'descr': '<f4', 'fortran_order': False, 'shape': ("
    header += shape.map(String.init).joined(separator: ", ")
    header += shape.count == 1 ? ",), }" : "), }"
    let preludeLen = 10
    var padded = header
    while (preludeLen + padded.count + 1) % 64 != 0 { padded += " " }
    padded += "\n"

    var data = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00])
    let hlen = UInt16(padded.count)
    data.append(UInt8(hlen & 0xFF))
    data.append(UInt8(hlen >> 8))
    data.append(padded.data(using: .ascii)!)
    values.withUnsafeBytes { data.append(contentsOf: $0) }
    try data.write(to: url)
}

/// The three host tables a GPT-2 (turbo/nano) model dir ships — `speech_emb.npy`,
/// `text_emb.npy`, `spkr_enc_{weight,bias}.npy` — all at `hidden` width. Row counts
/// don't matter to the width guard, so they stay tiny; only the widths are real.
func makeHostTables(hidden: Int) throws -> (SpeechEmbedding, EmbeddingTable, SpeakerProjection) {
    let dir = FileManager.default.temporaryDirectory
    let id = UUID().uuidString
    let sURL = dir.appendingPathComponent("speech_\(id).npy")
    let tURL = dir.appendingPathComponent("text_\(id).npy")
    let wURL = dir.appendingPathComponent("w_\(id).npy")
    let bURL = dir.appendingPathComponent("b_\(id).npy")
    defer { for u in [sURL, tURL, wURL, bURL] { try? FileManager.default.removeItem(at: u) } }

    try writeNPY([Float](repeating: 0, count: 4 * hidden), shape: [4, hidden], to: sURL)
    try writeNPY([Float](repeating: 0, count: 4 * hidden), shape: [4, hidden], to: tURL)
    try writeNPY([Float](repeating: 0, count: hidden * Constants.speakerEmbDim),
                 shape: [hidden, Constants.speakerEmbDim], to: wURL)   // spkr_enc: (H, 256)
    try writeNPY([Float](repeating: 0, count: hidden), shape: [hidden], to: bURL)
    return (
        try SpeechEmbedding(contentsOf: sURL),
        try EmbeddingTable(contentsOf: tURL, name: "text_emb"),
        try SpeakerProjection(weightURL: wURL, biasURL: bURL)
    )
}

struct ConstantsTests {
    @Test func architectureConstants() {
        #expect(Constants.gpt2HeadDim == 64)
        #expect(Constants.gpt2Layers * 2 == 48)      // KV planes (a key + a value per layer)
        #expect(Constants.speechStopToken == 6562)
        #expect(Constants.speechVocabSize == 6563)
    }

    // Nano's geometry is NOT asserted here: restating `NanoConstants` against the
    // literals it is defined with guards nothing. The numbers that matter are pinned
    // against the shipped graph in `EndToEndTests.graphGeometryMatchesVariantConstants`
    // (hidden + KV row) and against the host tables in `T3LMTableWidthGuardTests`.
    // The speech side is variant-independent — same 6563-wide head, same 6561/6562
    // start/stop — so the asserts above and every `SamplerTests` case cover nano as-is.
}

struct NPYReaderTests {
    @Test func readsRowsInOrder() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("emb_\(UUID().uuidString).npy")
        defer { try? FileManager.default.removeItem(at: tmp) }

        // 3 rows x 4 cols: row r = [r, r+0.5, r+1, r+1.5]
        var vals: [Float] = []
        for r in 0..<3 { vals += [Float(r), Float(r) + 0.5, Float(r) + 1, Float(r) + 1.5] }
        try writeNPY(vals, shape: [3, 4], to: tmp)

        let emb = try SpeechEmbedding(contentsOf: tmp)
        #expect(emb.vocab == 3)
        #expect(emb.hidden == 4)
        #expect(emb.row(0) == [0, 0.5, 1, 1.5])
        #expect(emb.row(2) == [2, 2.5, 3, 3.5])
    }
}

struct SamplerTests {
    @Test func greedyPicksArgmax() {
        var sampler = Sampler(options: GenerationOptions(minTokens: 0, greedy: true))
        let logits: [Float] = [0.1, 0.9, 0.2, 5.0, 0.3]
        let token = sampler.sample(logits: logits, previous: [], minTokensReached: true)
        #expect(token == 3)
    }

    @Test func stopTokenForbiddenBeforeMinLength() {
        var logits = [Float](repeating: 0, count: Constants.speechVocabSize)
        logits[Constants.speechStopToken] = 100 // would win if allowed
        logits[5] = 1
        var sampler = Sampler(options: GenerationOptions(minTokens: 4, greedy: true))
        let token = sampler.sample(logits: logits, previous: [], minTokensReached: false)
        #expect(token != Constants.speechStopToken)
        #expect(token == 5)
    }

    @Test func seededSamplingIsDeterministic() {
        let opts = GenerationOptions(temperature: 1.0, topP: 1.0, minTokens: 0, seed: 42)
        var a = Sampler(options: opts)
        var b = Sampler(options: opts)
        let logits: [Float] = (0..<10).map { Float($0) * 0.1 }
        let ta = a.sample(logits: logits, previous: [], minTokensReached: true)
        let tb = b.sample(logits: logits, previous: [], minTokensReached: true)
        #expect(ta == tb)
    }
}

struct Float16HelperTests {
    @Test func float16RoundTrip() throws {
        // Exactly representable in Float16 → lossless round trip.
        let values: [Float] = [0, 1, 2.5, -3, 0.5, -0.25, 100, -100]
        let array = try MLMultiArray.float16(values, shape: [values.count])
        #expect(array.dataType == .float16)
        #expect(array.toFloatArrayAnyPrecision() == values)
    }

    @Test func anyPrecisionReadsFloat32() throws {
        let values: [Float] = [1.1, 2.2, 3.3]
        let array = try MLMultiArray.float32(values, shape: [values.count])
        #expect(array.toFloatArrayAnyPrecision() == values)
    }
}

struct SpeakerProjectionTests {
    @Test func projectsLinearLayer() throws {
        let dir = FileManager.default.temporaryDirectory
        let wURL = dir.appendingPathComponent("w_\(UUID()).npy")
        let bURL = dir.appendingPathComponent("b_\(UUID()).npy")
        defer { try? FileManager.default.removeItem(at: wURL); try? FileManager.default.removeItem(at: bURL) }

        // out=2, in=3.  W = [[1,0,0],[0,2,0]], b = [10, 20].
        try writeNPY([1, 0, 0, 0, 2, 0], shape: [2, 3], to: wURL)
        try writeNPY([10, 20], shape: [2], to: bURL)
        let proj = try SpeakerProjection(weightURL: wURL, biasURL: bURL)
        #expect(proj.outDim == 2)
        #expect(proj.inDim == 3)
        // [5,7,9] → [1*5 + 10, 2*7 + 20] = [15, 34]
        #expect(proj.project([5, 7, 9]) == [15, 34])
    }
}

/// Tests the host-side padded-prefill assembly contract end-to-end with tiny
/// synthetic tables (`docs/T3LM-multifunction-contract.md`).
struct PrefillAssemblerTests {
    private let hidden = 4

    /// Builds an assembler whose tables encode the source token in element 0 of
    /// each row, so assembled rows are identifiable: speech rows → `token`,
    /// text rows → `1000 + token`, speaker row → `5000 + speakerEmb[0]`.
    private func makeAssembler(maxLen: Int) throws -> PrefillAssembler {
        let dir = FileManager.default.temporaryDirectory
        let id = UUID().uuidString
        let speechURL = dir.appendingPathComponent("speech_\(id).npy")
        let textURL = dir.appendingPathComponent("text_\(id).npy")
        let wURL = dir.appendingPathComponent("w_\(id).npy")
        let bURL = dir.appendingPathComponent("b_\(id).npy")

        // speech_emb: full speech vocab (needs row speechStartToken=6561).
        let speechVocab = Constants.speechVocabSize
        var speech = [Float](repeating: 0, count: speechVocab * hidden)
        for r in 0..<speechVocab { speech[r * hidden] = Float(r) }
        try writeNPY(speech, shape: [speechVocab, hidden], to: speechURL)

        // text_emb: 8 rows, marker = 1000 + token.
        var text = [Float](repeating: 0, count: 8 * hidden)
        for r in 0..<8 { text[r * hidden] = Float(1000 + r) }
        try writeNPY(text, shape: [8, hidden], to: textURL)

        // spkr_enc: out=hidden, in=2; element 0 = speakerEmb[0] + 5000.
        var w = [Float](repeating: 0, count: hidden * 2)
        w[0] = 1 // out row 0 picks speakerEmb[0]
        var b = [Float](repeating: 0, count: hidden)
        b[0] = 5000
        try writeNPY(w, shape: [hidden, 2], to: wURL)
        try writeNPY(b, shape: [hidden], to: bURL)

        return PrefillAssembler(
            speechEmb: try SpeechEmbedding(contentsOf: speechURL),
            textEmb: try EmbeddingTable(contentsOf: textURL, name: "text_emb"),
            spkrProjection: try SpeakerProjection(weightURL: wURL, biasURL: bURL),
            maxLen: maxLen,
            hidden: hidden
        )
    }

    private func conds(cond: [Int32]) -> Conditionals {
        Conditionals(
            speakerEmb: [7, 9],
            condPromptSpeechTokens: cond,
            genEmbedding: [1],
            promptTokens: [],
            promptFeat: []
        )
    }

    @Test func assemblesInContractOrderWithLeftPad() throws {
        let maxLen = 16
        let a = try makeAssembler(maxLen: maxLen)
        let result = a.assemble(textTokens: [3, 4, 5], conds: conds(cond: [100, 200]))

        // T_real = 1 (spkr) + 2 (cond) + 3 (text) + 1 (speech start) = 7
        #expect(result.realLength == 7)
        #expect(result.nPad == maxLen - 7)

        func marker(_ rowFromRealStart: Int) -> Float {
            result.inputsEmbeds[(result.nPad + rowFromRealStart) * hidden]
        }
        #expect(marker(0) == 5007)                            // speaker: 7 + 5000
        #expect(marker(1) == 100)                             // cond[0]
        #expect(marker(2) == 200)                             // cond[1]
        #expect(marker(3) == 1003)                            // text 3 → 1000+3
        #expect(marker(4) == 1004)
        #expect(marker(5) == 1005)
        #expect(marker(6) == Float(Constants.speechStartToken)) // speech start row

        // Pad rows are zeroed.
        for j in 0..<result.nPad {
            for d in 0..<hidden { #expect(result.inputsEmbeds[j * hidden + d] == 0) }
        }
    }

    @Test func positionIdsAndMaskAreLeftPadded() throws {
        let maxLen = 16
        let a = try makeAssembler(maxLen: maxLen)
        let r = a.assemble(textTokens: [3, 4, 5], conds: conds(cond: [100, 200]))

        for j in 0..<r.nPad {
            #expect(r.positionIds[j] == 0)
            #expect(r.keyPaddingMask[j] == 0)
        }
        for k in 0..<r.realLength {
            #expect(r.positionIds[r.nPad + k] == Int32(k))
            #expect(r.keyPaddingMask[r.nPad + k] == 1)
        }
        // Start-speech token lands at the last index (logits row).
        #expect(r.nPad + r.realLength == maxLen)
    }

    @Test func truncatesTextNotConditioningWhenOverMax() throws {
        // maxLen 10, T_cond 5 → textBudget = 10 - 5 - 2 = 3.
        let maxLen = 10
        let a = try makeAssembler(maxLen: maxLen)
        let r = a.assemble(textTokens: [1, 2, 3, 4, 5, 6, 7], conds: conds(cond: [10, 11, 12, 13, 14]))
        #expect(r.realLength == maxLen)   // 1 + 5 + 3 + 1
        #expect(r.nPad == 0)
        // First text row marker should be token 1 (1000+1); only 3 text rows kept.
        #expect(r.inputsEmbeds[(1 + 5) * hidden] == 1001)
        #expect(r.inputsEmbeds[(1 + 5 + 2) * hidden] == 1003) // last kept text token = 3
    }

    @Test func slicesRealKVTail() {
        // planes=2, heads=1, maxLen=4, headDim=1, nPad=1 → realLength=3.
        // plane0 = [0,1,2,3], plane1 = [10,11,12,13]; tail t∈[1,4).
        let padded: [Float] = [0, 1, 2, 3, 10, 11, 12, 13]
        let real = PrefillAssembler.sliceRealKV(
            paddedCache: padded, maxLen: 4, nPad: 1, planes: 2, heads: 1, headDim: 1)
        #expect(real == [1, 2, 3, 11, 12, 13])
    }
}

/// The nano port's highest-risk failure mode, and the only one that is SILENT: a
/// nano-width `T3LM` graph fed turbo-width host tables (or vice versa — the two
/// variants ship identical file names) does not crash. CoreML accepts the flat
/// `inputs_embeds` buffer and the pipeline emits noise. `T3LMRunner` rejects the
/// pairing at load; these exercise that predicate directly. It lives in a static func
/// precisely because inside `init` it sits behind two `MLModel` loads, so it is
/// unreachable without a real `.mlpackage` (the e2e `loadRejectsWrongWidthHostTables`
/// covers `init` actually calling it).
struct T3LMTableWidthGuardTests {
    @Test func acceptsMatchingWidths() throws {
        for hidden in [Constants.gpt2Hidden, NanoConstants.hidden] {
            let (s, t, p) = try makeHostTables(hidden: hidden)
            try T3LMRunner.validateTableWidths(
                graphHidden: hidden, speechEmb: s, textEmb: t, spkrProjection: p)
        }
    }

    @Test func rejectsTurboTablesAgainstANanoGraph() throws {
        let (s, t, p) = try makeHostTables(hidden: Constants.gpt2Hidden)
        let err = #expect(throws: ChatterboxError.self) {
            try T3LMRunner.validateTableWidths(
                graphHidden: NanoConstants.hidden, speechEmb: s, textEmb: t, spkrProjection: p)
        }
        guard case .shapeMismatch(let msg)? = err else {
            Issue.record("expected .shapeMismatch, got \(String(describing: err))"); return
        }
        // The diagnostic must name the graph AND the offending table, or the operator
        // can't tell which half of the model dir is stale.
        #expect(msg.contains("hidden=768"))
        #expect(msg.contains("speech_emb=1024"))
    }

    @Test func rejectsNanoTablesAgainstATurboGraph() throws {
        let (s, t, p) = try makeHostTables(hidden: NanoConstants.hidden)
        #expect(throws: ChatterboxError.self) {
            try T3LMRunner.validateTableWidths(
                graphHidden: Constants.gpt2Hidden, speechEmb: s, textEmb: t, spkrProjection: p)
        }
    }

    /// A *partially* swapped model dir — one `.npy` left over from the other variant —
    /// must fail too; each table is checked independently, not just `speech_emb`.
    @Test func rejectsASingleMismatchedTable() throws {
        let (nanoSpeech, _, nanoSpkr) = try makeHostTables(hidden: NanoConstants.hidden)
        let (_, turboText, _) = try makeHostTables(hidden: Constants.gpt2Hidden)
        #expect(throws: ChatterboxError.self) {
            try T3LMRunner.validateTableWidths(
                graphHidden: NanoConstants.hidden,
                speechEmb: nanoSpeech, textEmb: turboText, spkrProjection: nanoSpkr)
        }
    }
}

struct TextChunkerTests {
    /// Fake tokenizer: one token per whitespace-separated word (no normalization),
    /// so `maxTokens` reads as a word budget and chunks are easy to reason about.
    private let chunker = TextChunker(encode: { text in
        text.split(separator: " ").enumerated().map { Int32($0.offset) }
    })

    @Test func shortTextIsOneChunk() {
        let chunks = chunker.chunk("One two three.", maxTokens: 10)
        #expect(chunks.count == 1)
        #expect(chunks[0].count == 3)
    }

    @Test func emptyTextYieldsNoChunks() {
        #expect(chunker.chunk("   \n  ", maxTokens: 10).isEmpty)
    }

    @Test func packsWholeSentencesUnderBudget() {
        // counts: 3, 3, 2 words. budget 5 → [s1] then [s2+s3].
        let chunks = chunker.chunk("One two three. Four five six. Seven eight.", maxTokens: 5)
        #expect(chunks.count == 2)
        #expect(chunks.allSatisfy { $0.count <= 5 })
        #expect(chunks[0].count == 3)
        #expect(chunks[1].count == 5)
    }

    @Test func splitsAnOversizeSentenceByWords() {
        // No terminal punctuation → one segment of 7 words; budget 3.
        let chunks = chunker.chunk("a b c d e f g", maxTokens: 3)
        #expect(chunks.count == 3)
        #expect(chunks.allSatisfy { $0.count <= 3 })
        #expect(chunks.map(\.count) == [3, 3, 1])
    }

    @Test func everyChunkFitsBudget() {
        let text = String(repeating: "Hello there friend. ", count: 40) // 120 words
        let chunks = chunker.chunk(text, maxTokens: 7)
        #expect(!chunks.isEmpty)
        #expect(chunks.allSatisfy { $0.count <= 7 })
    }

    // MARK: - segments (pause-aware)

    @Test func segmentsSplitPerSentenceWithTrailingPause() {
        // Unlike `chunk`, sentences are NOT packed — one unit each — so a pause can
        // follow every sentence. The final unit carries no pause (no dead air).
        let segs = chunker.segments("One two. Three four.", maxTokens: 10,
                                    sentencePause: 0.25, commaPause: 0)
        #expect(segs.count == 2)
        #expect(segs[0].tokens.count == 2)
        #expect(segs[0].trailingPause == 0.25)
        #expect(segs[1].trailingPause == 0)
    }

    @Test func segmentsSplitAtCommasWhenCommaPauseSet() {
        let segs = chunker.segments("One, two, three.", maxTokens: 10,
                                    sentencePause: 0.25, commaPause: 0.1)
        #expect(segs.count == 3)               // "One," / "two," / "three."
        #expect(segs[0].trailingPause == 0.1)
        #expect(segs[1].trailingPause == 0.1)
        #expect(segs[2].trailingPause == 0)    // last, suppressed
    }

    @Test func commasStayInSentenceWhenCommaPauseZero() {
        let segs = chunker.segments("One, two, three.", maxTokens: 10,
                                    sentencePause: 0.25, commaPause: 0)
        #expect(segs.count == 1)               // no comma split; single trailing sentence
        #expect(segs[0].tokens.count == 3)
        #expect(segs[0].trailingPause == 0)    // also the last unit
    }

    @Test func oversizeClauseSubSplitsWithPauseOnlyOnLast() {
        // "a b c d," is a 4-token comma-clause; budget 3 → [a b c][d,]; the pause
        // rides only the last sub. "e." is the final unit → no pause.
        let segs = chunker.segments("a b c d, e.", maxTokens: 3,
                                    sentencePause: 0.2, commaPause: 0.1)
        #expect(segs.count == 3)
        #expect(segs[0].tokens.count == 3)
        #expect(segs[0].trailingPause == 0)
        #expect(segs[1].tokens.count == 1)
        #expect(segs[1].trailingPause == 0.1)
        #expect(segs[2].trailingPause == 0)
    }

    @Test func pausedSegmentsWithoutPausesMatchesChunkPacking() {
        let text = "One two three. Four five six. Seven eight."
        let segs = chunker.pausedSegments(text, maxTokens: 5, options: GenerationOptions())
        #expect(segs.map(\.tokens) == chunker.chunk(text, maxTokens: 5))  // packed, unchanged
        #expect(segs.allSatisfy { $0.trailingPause == 0 })
    }
}

/// Windowing the generated speech tokens so `prompt ++ generated` never exceeds
/// the S3Encoder's `speech_tokens` cap (`Constants.maxVocoderTokens`).
struct SynthTokenWindowTests {
    @Test func shortSequenceIsOneWindowUnchanged() {
        let tokens: [Int32] = [1, 2, 3, 4, 5]
        let windows = SynthRunner.tokenWindows(tokens, maxPerWindow: 10)
        #expect(windows.count == 1)
        #expect(windows[0] == tokens)
    }

    @Test func exactlyAtWindowStaysOneWindow() {
        let tokens = Array<Int32>(0..<10)
        let windows = SynthRunner.tokenWindows(tokens, maxPerWindow: 10)
        #expect(windows.count == 1)
        #expect(windows[0] == tokens)
    }

    @Test func oversizeSequenceSplitsWithoutDroppingTokens() {
        // 809 generated + a 250-token prompt → window 774; this reproduces the
        // overflow from the bug report (250 + 809 = 1059 > 1024).
        let tokens = Array<Int32>(0..<809)
        let windows = SynthRunner.tokenWindows(tokens, maxPerWindow: 1024 - 250)
        #expect(windows.count == 2)
        #expect(windows.allSatisfy { $0.count <= 774 })
        // Every token preserved, in order, exactly once.
        #expect(windows.flatMap { $0 } == tokens)
    }

    @Test func emptyGeneratedYieldsASingleEmptyWindow() {
        let windows = SynthRunner.tokenWindows([], maxPerWindow: 774)
        #expect(windows.count == 1)
        #expect(windows[0].isEmpty)
    }

    @Test func degenerateMaxIsClampedToOne() {
        let tokens: [Int32] = [1, 2, 3]
        let windows = SynthRunner.tokenWindows(tokens, maxPerWindow: 0)
        #expect(windows.count == 3)
        #expect(windows.flatMap { $0 } == tokens)
    }

    // MARK: - windowBudget (padded-ANE window cap)

    @Test func windowBudgetOffModeUsesEncoderCapOnly() {
        // pad off (foreground cpuAndGPU): the only bound is the encoder's 1024 cap.
        #expect(SynthRunner.windowBudget(promptTokens: 250, pad: false)
                == Constants.maxVocoderTokens - 250)
        #expect(SynthRunner.windowBudget(promptTokens: 0, pad: false)
                == Constants.maxVocoderTokens)
    }

    @Test func windowBudgetPadModeCapsAtHalfPadTargetMinusPrompt() {
        // pad on: additionally cap by cfmPadTarget/2 − prompt so T_h = 2·(prompt+gen)
        // ≤ cfmPadTarget. Turbo prompt 250 → 512 − 250 = 262 (the pre-fix overflow
        // point: 263 gen → T_h 1026 > 1024).
        #expect(SynthRunner.windowBudget(promptTokens: 250, pad: true)
                == SynthRunner.cfmPadTarget / 2 - 250)
        #expect(SynthRunner.windowBudget(promptTokens: 250, pad: true) == 262)
        // Every window fits the pad target: (prompt + budget) · 2 ≤ cfmPadTarget.
        for prompt in [0, 100, 250, 400, 511] {
            let budget = SynthRunner.windowBudget(promptTokens: prompt, pad: true)
            #expect(2 * (prompt + budget) <= SynthRunner.cfmPadTarget)
        }
    }

    @Test func windowBudgetDegeneratePromptFallsBackToEncoderCap() {
        // prompt ≥ cfmPadTarget/2 → ANE cap ≤0; keep the unpadded encoder budget
        // (CPU fallback) instead of absurd 1-token windows.
        let huge = SynthRunner.cfmPadTarget / 2 + 100      // 612
        #expect(SynthRunner.windowBudget(promptTokens: huge, pad: true)
                == Constants.maxVocoderTokens - huge)      // 412, unpadded
        // Exactly at the boundary the ANE cap is 0 → also falls back.
        #expect(SynthRunner.windowBudget(promptTokens: SynthRunner.cfmPadTarget / 2, pad: true)
                == Constants.maxVocoderTokens - SynthRunner.cfmPadTarget / 2)
    }

    @Test func windowBudgetNeverDropsBelowOne() {
        // Prompt past the encoder cap too → clamped to 1, not negative.
        #expect(SynthRunner.windowBudget(promptTokens: 2000, pad: false) == 1)
        #expect(SynthRunner.windowBudget(promptTokens: 2000, pad: true) == 1)
    }

    @Test func windowBudgetSplitsBugReportUtterance() {
        // The device-traced overflow: prompt 250, 282 generated tokens. Off mode
        // keeps it one 774-window (T_h 1064 > 1024 → CPU fallback, the bug); pad
        // mode caps at 262 → 2 windows [262, 20], each T_h ≤ 1024 → ANE per window.
        let gen = Array<Int32>(0..<282)
        let off = SynthRunner.tokenWindows(gen, maxPerWindow:
            SynthRunner.windowBudget(promptTokens: 250, pad: false))
        #expect(off.count == 1)
        let padded = SynthRunner.tokenWindows(gen, maxPerWindow:
            SynthRunner.windowBudget(promptTokens: 250, pad: true))
        #expect(padded.map(\.count) == [262, 20])
        #expect(padded.allSatisfy { 2 * (250 + $0.count) <= SynthRunner.cfmPadTarget })
    }

    // MARK: - The 1024 default-shape crash (encoder input never reaches the limit)

    @Test func encoderCapStaysBelowTheGraphLimitAndIsAMultipleOfFour() {
        // Feeding the S3Encoder exactly `encoderTokenLimit` (its RangeDim DEFAULT
        // shape) fails on both accelerators — on iOS as an uncatchable ObjC
        // NSGenericException that terminates the app. The fed cap must stay strictly
        // under it, and be a multiple of 4 so the pad below can't climb back onto it.
        #expect(Constants.maxVocoderTokens < Constants.encoderTokenLimit)
        #expect(Constants.maxVocoderTokens % 4 == 0)
    }

    @Test func noWindowEverFeedsTheEncoderItsDefaultShape() {
        // The device crash (2026-07-26): a chunk over-generated to 853 tokens with a
        // 250-token prompt, so `synthesize` took the multi-window path — and
        // `tokenWindows` makes every non-final window exactly `windowBudget` long, so
        // window 1 fed `prompt ++ window` == the cap on the FIRST pass. With the cap at
        // 1024 that was the fatal shape; every foreground multi-window utterance hit it.
        for prompt in [0, 1, 100, 250, 511, 512, 612, 1019] {
            for pad in [false, true] {
                let budget = SynthRunner.windowBudget(promptTokens: prompt, pad: pad)
                for generated in [0, 1, 282, 853, 2000] {
                    let windows = SynthRunner.tokenWindows(
                        Array<Int32>(0..<Int32(generated)), maxPerWindow: budget)
                    for window in windows {
                        let realT = prompt + window.count
                        let fed = realT + SynthRunner.encoderPad(realT)
                        #expect(fed < Constants.encoderTokenLimit,
                                "prompt \(prompt) pad \(pad) window \(window.count) → fed \(fed)")
                    }
                }
            }
        }
    }

    @Test func regressionTheCrashingUtteranceNowSplitsBelowTheLimit() {
        // Exactly the traced case: prompt 250, 853 generated, foreground (pad off).
        let budget = SynthRunner.windowBudget(promptTokens: 250, pad: false)
        #expect(budget == 770)                                   // was 774 → fed 1024
        let windows = SynthRunner.tokenWindows(Array<Int32>(0..<853), maxPerWindow: budget)
        #expect(windows.map(\.count) == [770, 83])
        #expect(250 + windows[0].count == Constants.maxVocoderTokens)   // 1020, ANE-safe
        #expect(windows.flatMap { $0 } == Array<Int32>(0..<853))        // nothing dropped
    }
}

/// Padded-ANE CFM plumbing (`SynthRunner.padFrames`/`padMask`/`sliceFrames`/
/// `shouldPad`): the tensor surgery that lets the CFM Euler loop run at the RangeDim
/// default shape 1024 on the ANE. Pure — exercised offline, no CoreML models.
struct SynthPaddedCFMTests {
    @Test func shouldPadOnlyWithANEorForceAndFittingLength() {
        // Foreground default (cpuAndGPU → usesANE false, no override): never pads.
        #expect(SynthRunner.shouldPad(usesANE: false, forcePad: false, tH: 400) == false)
        // ANE CU + fits the pad target → pads.
        #expect(SynthRunner.shouldPad(usesANE: true, forcePad: false, tH: 400) == true)
        #expect(SynthRunner.shouldPad(usesANE: true, forcePad: false, tH: SynthRunner.cfmPadTarget) == true)
        // Test override forces the padded path on any CU.
        #expect(SynthRunner.shouldPad(usesANE: false, forcePad: true, tH: 400) == true)
        // Past the pad target → unpadded even on the ANE (T>1024 falls back to CPU/GPU).
        #expect(SynthRunner.shouldPad(usesANE: true, forcePad: true, tH: SynthRunner.cfmPadTarget + 1) == false)
    }

    @Test func padFramesRightPadsWithZerosPerChannel() {
        // (mel=2, srcT=3) C-order: ch0 = [1,2,3], ch1 = [4,5,6].
        let src: [Float] = [1, 2, 3, 4, 5, 6]
        let out = SynthRunner.padFrames(src, mel: 2, srcT: 3, dstT: 5)
        // Each channel keeps its 3 real frames at the front, 2 zeros appended.
        #expect(out == [1, 2, 3, 0, 0, 4, 5, 6, 0, 0])
    }

    @Test func padFramesIsIdentityWhenLengthsMatch() {
        let src: [Float] = [1, 2, 3, 4, 5, 6]
        #expect(SynthRunner.padFrames(src, mel: 2, srcT: 3, dstT: 3) == src)
    }

    @Test func padMaskMarksRealFramesThenPad() {
        #expect(SynthRunner.padMask(realT: 3, runT: 5) == [1, 1, 1, 0, 0])
        #expect(SynthRunner.padMask(realT: 3, runT: 3) == [1, 1, 1])   // all-ones unpadded
    }

    @Test func sliceDropsLeftPromptFramesAndRightPad() {
        // Padded (mel=2, runT=5): ch0 = [10,11,12,0,0], ch1 = [20,21,22,0,0];
        // real length T_h=3, prompt melLen1=1 → melLen2=2.
        let padded: [Float] = [10, 11, 12, 0, 0, 20, 21, 22, 0, 0]
        let feat = SynthRunner.sliceFrames(padded, mel: 2, runT: 5, start: 1, count: 2)
        #expect(feat == [11, 12, 21, 22])       // frames [1,3) per channel; pad dropped
    }

    /// The load-bearing invariance: padding to the ANE shape then slicing the real
    /// frames yields the SAME feat as slicing the unpadded latent — pad is on the
    /// RIGHT, the prompt slice drops from the LEFT, so real-frame indices are stable.
    @Test func paddedThenSlicedEqualsUnpaddedSliced() {
        let mel = 4, tH = 6, melLen1 = 2
        let melLen2 = tH - melLen1
        // Distinct per-(channel,frame) values so any mis-stride shows up.
        var latent = [Float](repeating: 0, count: mel * tH)
        for m in 0..<mel { for f in 0..<tH { latent[m * tH + f] = Float(m * 100 + f) } }

        let unpaddedFeat = SynthRunner.sliceFrames(latent, mel: mel, runT: tH, start: melLen1, count: melLen2)
        let runT = SynthRunner.cfmPadTarget
        let padded = SynthRunner.padFrames(latent, mel: mel, srcT: tH, dstT: runT)
        let paddedFeat = SynthRunner.sliceFrames(padded, mel: mel, runT: runT, start: melLen1, count: melLen2)
        #expect(paddedFeat == unpaddedFeat)
    }
}

/// The chunk text budget is bounded by both the prefill window and the vocoder
/// window, so a chunk's generation fits a single S3Encoder pass.
struct TextBudgetTests {
    @Test func prefillBoundWinsForShortSynthPrompt() {
        // prefill = 512 − 470 − 2 = 40; vocoder = (1024 − 0) / 8 = 128 → 40 wins.
        let budget = ChatterboxCoreMLModel.textBudget(
            maxLen: 512, condPromptSpeechTokenCount: 470, synthPromptTokenCount: 0)
        #expect(budget == 40)
    }

    @Test func vocoderBoundCapsLongSynthPrompt() {
        // The bug's voice: prefill 135, but a 250-token synth prompt caps the
        // vocoder budget to (1024 − 250) / 8 = 96.
        let budget = ChatterboxCoreMLModel.textBudget(
            maxLen: 512, condPromptSpeechTokenCount: 375, synthPromptTokenCount: 250)
        #expect(budget == 96)
    }

    @Test func budgetNeverDropsBelowOne() {
        let budget = ChatterboxCoreMLModel.textBudget(
            maxLen: 10, condPromptSpeechTokenCount: 100, synthPromptTokenCount: 2000)
        #expect(budget == 1)
    }
}

/// End-to-end smoke test, only runs when CHATTERBOX_MODEL_DIR points at a local
/// model directory (downloaded via ModelRepository).
struct EndToEndTests {
    private var modelDir: URL? {
        ProcessInfo.processInfo.environment["CHATTERBOX_MODEL_DIR"].map(URL.init(fileURLWithPath:))
    }

    /// The T3LM `decode` graph of the dir under test. `.cpuOnly` because only shapes are
    /// read — no prediction, so no ANE AOT compile.
    private func decodeGraph(in dir: URL) async throws -> MLModel? {
        let fm = FileManager.default
        guard let t3lm = ["T3LM.mlmodelc", "T3LM.mlpackage"]
            .map({ dir.appendingPathComponent($0) })
            .first(where: { fm.fileExists(atPath: $0.path) }) else { return nil }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuOnly
        cfg.functionName = "decode"
        return try MLModel(contentsOf: await ChatterboxCoreMLModel.compiledModel(at: t3lm),
                           configuration: cfg)
    }

    /// The variant the dir under test MUST resolve to, derived from the artifacts that
    /// actually ship: multilingual is the only one with `perceiver_query.npy`, and the two
    /// GPT-2 variants are told apart by the T3LM graph's hidden width. `load` keys off the
    /// host `.npy` widths instead, so this is a cross-check against a different source, not
    /// a restatement of its rule.
    private func expectedVariant(in dir: URL, graph: MLModel) -> ModelRepository.Variant? {
        if FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("perceiver_query.npy").path) { return .multilingual }
        switch graphHidden(graph) {
        case NanoConstants.hidden: return .nano
        case Constants.gpt2Hidden: return .turbo
        case let h:
            Issue.record("T3LM graph hidden=\(String(describing: h)) is neither nano nor turbo")
            return nil
        }
    }

    private func graphHidden(_ graph: MLModel) -> Int? {
        graph.modelDescription.inputDescriptionsByName["inputs_embeds"]?
            .multiArrayConstraint?.shape.last?.intValue
    }

    @Test func generatesAudio() async throws {
        guard let modelDir else {
            // Skipped without a model directory.
            return
        }
        let model = try await ChatterboxCoreMLModel.load(from: modelDir, watermarker: NoWatermark())
        // Whichever dir is pointed at (out/ = turbo, out-nano/ = nano, multilingual), the
        // load must self-identify as the variant the artifacts actually ARE, and produce
        // audio at 24 kHz.
        let graph = try #require(await decodeGraph(in: modelDir))
        #expect(model.loadedVariant == expectedVariant(in: modelDir, graph: graph))
        let buffer = try await model.generate(
            "Hello there.",
            options: GenerationOptions(maxTokens: 80)
        )
        #expect(buffer.format.sampleRate == Constants.sampleRate)
        #expect(buffer.frameLength > 0)
    }

    /// The geometry constants vs the graph that actually ships. Nothing in the runtime
    /// reconstructs these — hidden is read off `inputs_embeds`, the KV row off the state
    /// buffer — so a converter that changed a variant's depth or width would otherwise
    /// drift away from `Constants`/`NanoConstants` in silence. The flat KV row is
    /// `lanes · layers · hidden` (turbo 24·1024, nano 12·768; multilingual folds its 2 CFG
    /// lanes into the row).
    @Test func graphGeometryMatchesVariantConstants() async throws {
        guard let modelDir else { return }
        let graph = try #require(await decodeGraph(in: modelDir))
        let variant = try #require(expectedVariant(in: modelDir, graph: graph))
        let (hidden, layers, lanes): (Int, Int, Int) = switch variant {
        case .nano: (NanoConstants.hidden, NanoConstants.layers, 1)
        case .turbo: (Constants.gpt2Hidden, Constants.gpt2Layers, 1)
        case .multilingual: (MultilingualConstants.hidden, MultilingualConstants.layers,
                             MultilingualConstants.batch)
        }
        #expect(graphHidden(graph) == hidden)

        var kvRow = 0
        graph.makeState().withMultiArray(for: "keyCache") { kvRow = $0.shape.last?.intValue ?? 0 }
        #expect(kvRow == lanes * layers * hidden)
    }

    /// The width guard against the REAL compiled graph: same model dir, but host tables
    /// from the *other* GPT-2 variant. `T3LMRunner.init` must throw rather than
    /// synthesize noise. The offline `T3LMTableWidthGuardTests` cover the predicate;
    /// this is what proves `init` still calls it. Skipped for a multilingual dir (no
    /// `speech_emb.npy`, different runner).
    @Test func loadRejectsWrongWidthHostTables() async throws {
        guard let modelDir else { return }
        let fm = FileManager.default
        let speechURL = modelDir.appendingPathComponent("speech_emb.npy")
        guard fm.fileExists(atPath: speechURL.path) else { return }
        guard let t3lm = ["T3LM.mlmodelc", "T3LM.mlpackage"]
            .map({ modelDir.appendingPathComponent($0) })
            .first(where: { fm.fileExists(atPath: $0.path) }) else { return }
        let compiled = try await ChatterboxCoreMLModel.compiledModel(at: t3lm)

        // Flip the tables to the other variant's width — the half-swapped-model-dir
        // mistake — and keep the graph real.
        let real = try SpeechEmbedding(contentsOf: speechURL)
        let wrong = real.hidden == NanoConstants.hidden ? Constants.gpt2Hidden : NanoConstants.hidden
        let (s, t, p) = try makeHostTables(hidden: wrong)
        #expect(throws: ChatterboxError.self) {
            _ = try T3LMRunner(contentsOf: compiled, speechEmb: s, textEmb: t, spkrProjection: p)
        }
    }

    @Test func longTextStreamsMultipleChunks() async throws {
        guard let modelDir else { return }
        let model = try await ChatterboxCoreMLModel.load(from: modelDir, watermarker: NoWatermark())
        // Several sentences — comfortably past the ~135-token single-window budget.
        let text = String(
            repeating: "This is a fairly long sentence used to exercise chunked generation. ",
            count: 24) // ~260 text tokens — well past the ~135-token single-window budget
        var chunks = 0
        var frames = 0
        for try await chunk in model.generateStream(text, options: GenerationOptions(maxTokens: 120)) {
            chunks += 1
            frames += chunk.samples.count
            #expect(chunk.prefillTime >= 0)
            #expect(chunk.onnxTime >= 0)
        }
        #expect(chunks > 1)        // long text must split into multiple chunks
        #expect(frames > 0)
    }
}

/// Both nested suites mutate process-global env (`CHATTERBOX_SYNTH_SEED`,
/// `CHATTERBOX_SYNTH_CFM_CU/…_PAD`, `CHATTERBOX_OVERLAP_CHUNKS`) as test input,
/// so they must never run concurrently — `.serialized` applies recursively and
/// keeps this subtree one-test-at-a-time (Swift Testing runs suites in parallel
/// otherwise, which made these two interleave and poison each other's runs).
@Suite(.serialized) struct SynthEnvE2ETests {
    /// Parity between the unpadded CFM path (foreground, `cpuAndGPU`) and the padded-ANE
    /// path forced on the SAME CU via `CHATTERBOX_SYNTH_CFM_PAD=1`. Gated on
    /// `CHATTERBOX_MODEL_DIR` (loads the real synth packages) — the offline
    /// `SynthPaddedCFMTests` cover the pure plumbing. Same seed + same tokens → the
    /// only difference is CFM running at the
    /// padded shape 1024 with inert pad frames, so the waveforms must match ≥0.999.
    struct SynthPaddedParityTests {
        private var modelDir: URL? {
            ProcessInfo.processInfo.environment["CHATTERBOX_MODEL_DIR"].map(URL.init(fileURLWithPath:))
        }

        private func cosine(_ x: [Float], _ y: [Float]) -> Double {
            let n = min(x.count, y.count)
            var dot = 0.0, nx = 0.0, ny = 0.0
            for i in 0..<n { dot += Double(x[i]) * Double(y[i]); nx += Double(x[i]) * Double(x[i]); ny += Double(y[i]) * Double(y[i]) }
            return dot / ((nx * ny).squareRoot() + 1e-12)
        }

        /// Resolves a synth model file in `dir`, compiling a `.mlpackage` (converter
        /// output) to a temp `.mlmodelc`; returns nil when the model is absent (older
        /// turbo-only dirs) so the test skips instead of failing.
        private func synthURL(_ base: String, in dir: URL) async throws -> URL? {
            let fm = FileManager.default
            let compiled = dir.appendingPathComponent("\(base).mlmodelc")
            if fm.fileExists(atPath: compiled.path) { return compiled }
            let pkg = dir.appendingPathComponent("\(base).mlpackage")
            guard fm.fileExists(atPath: pkg.path) else { return nil }
            return try await MLModel.compileModel(at: pkg)
        }

        @Test func paddedCFMMatchesUnpaddedWaveform() async throws {
            guard let modelDir else { return }
            guard let encU = try await synthURL("S3Encoder", in: modelDir),
                  let cfmU = try await synthURL("S3CFM", in: modelDir),
                  let vocU = try await synthURL("S3Vocoder", in: modelDir) else { return }

            let conds = try Conditionals(contentsOf: modelDir.appendingPathComponent("default-conds.safetensors"))
            // Keep T_h = 2·(prompt+gen) ≤ cfmPadTarget so the forced-pad path actually pads.
            let prompt = conds.promptTokens.count
            let genCount = SynthRunner.cfmPadTarget / 2 - prompt - 8
            guard genCount > 0 else { return }              // prompt already fills the window
            let generated = (0..<genCount).map { $0 % 512 }  // deterministic, in-range tokens

            // Force cpuAndGPU (Mac-safe) + a fixed synth-noise seed so both runs are
            // deterministic; only CHATTERBOX_SYNTH_CFM_PAD differs between them.
            setenv("CHATTERBOX_SYNTH_CFM_CU", "gpu", 1)
            setenv("CHATTERBOX_SYNTH_SEED", "1234", 1)
            defer {
                unsetenv("CHATTERBOX_SYNTH_CFM_CU"); unsetenv("CHATTERBOX_SYNTH_SEED")
                unsetenv("CHATTERBOX_SYNTH_CFM_PAD")
            }
            let synth = try SynthRunner(encoderURL: encU, cfmURL: cfmU, vocoderURL: vocU)

            unsetenv("CHATTERBOX_SYNTH_CFM_PAD")
            let unpadded = try synth.synthesize(generatedTokens: generated, conds: conds)
            setenv("CHATTERBOX_SYNTH_CFM_PAD", "1", 1)       // force the padded-ANE path
            let padded = try synth.synthesize(generatedTokens: generated, conds: conds)

            #expect(unpadded.count == padded.count)          // same real length → same #samples
            #expect(unpadded.count > 0)
            #expect(cosine(unpadded, padded) >= 0.999)
        }

        /// Multi-window padded path: generated tokens exceed the per-window padded
        /// budget (`windowBudget` with pad on), so the run splits into ≥2 windows that
        /// each pad to `cfmPadTarget` — the real-utterance case the cap was added for
        /// (turbo prompt 250 + >262 gen → T_h > 1024 → CPU fallback without it).
        /// Exact parity vs the single-window unpadded run is NOT asserted: different
        /// windowing re-runs encoder/CFM/vocoder per segment with the prompt
        /// re-prepended, so the window seams differ. Instead assert well-formedness +
        /// determinism vs a second identical seeded run.
        @Test func paddedCFMSplitsIntoMultipleWindows() async throws {
            guard let modelDir else { return }
            guard let encU = try await synthURL("S3Encoder", in: modelDir),
                  let cfmU = try await synthURL("S3CFM", in: modelDir),
                  let vocU = try await synthURL("S3Vocoder", in: modelDir) else { return }

            let conds = try Conditionals(contentsOf: modelDir.appendingPathComponent("default-conds.safetensors"))
            let prompt = conds.promptTokens.count
            guard prompt < SynthRunner.cfmPadTarget / 2 else { return }  // padded windows need prompt<512
            let budget = SynthRunner.windowBudget(promptTokens: prompt, pad: true)
            let genCount = 2 * budget                        // → exactly 2 padded windows
            let generated = (0..<genCount).map { $0 % 512 }  // deterministic, in-range tokens
            // Setup sanity: this run really exercises the multi-window padded path.
            #expect(SynthRunner.tokenWindows(generated.map(Int32.init), maxPerWindow: budget).count >= 2)

            setenv("CHATTERBOX_SYNTH_CFM_CU", "gpu", 1)      // Mac-safe CU
            setenv("CHATTERBOX_SYNTH_SEED", "1234", 1)       // deterministic synth noise
            setenv("CHATTERBOX_SYNTH_CFM_PAD", "1", 1)       // force padded path + window cap
            defer {
                unsetenv("CHATTERBOX_SYNTH_CFM_CU"); unsetenv("CHATTERBOX_SYNTH_SEED")
                unsetenv("CHATTERBOX_SYNTH_CFM_PAD")
            }
            let synth = try SynthRunner(encoderURL: encU, cfmURL: cfmU, vocoderURL: vocU)
            let a = try synth.synthesize(generatedTokens: generated, conds: conds)
            let b = try synth.synthesize(generatedTokens: generated, conds: conds)
            #expect(a.count > 0)
            #expect(a.count == b.count)                      // deterministic sample count
            #expect(cosine(a, b) >= 0.9999)                  // determinism vs itself (GPU fp16 jitter tol.)
        }
    }

    /// Cross-chunk overlap (issue #22) must reorder *only scheduling*, never output:
    /// the same seeded multi-chunk utterance has to render identically with the
    /// pipeline OFF and ON. Gated on `CHATTERBOX_MODEL_DIR`. Only this test writes
    /// `CHATTERBOX_OVERLAP_CHUNKS` in-process (others merely read it), and each stream
    /// is fully drained before the flag is flipped, so the two runs are deterministic.
    struct OverlapParityTests {
        private var modelDir: URL? {
            ProcessInfo.processInfo.environment["CHATTERBOX_MODEL_DIR"].map(URL.init(fileURLWithPath:))
        }

        private func collect(_ model: ChatterboxCoreMLModel, _ text: String, _ opts: GenerationOptions) async throws -> [ChatterboxCoreMLModel.AudioChunk] {
            var out: [ChatterboxCoreMLModel.AudioChunk] = []
            for try await c in model.generateStream(text, options: opts) { out.append(c) }
            return out
        }

        private func cosine(_ x: [Float], _ y: [Float]) -> Double {
            let n = min(x.count, y.count)
            var dot = 0.0, nx = 0.0, ny = 0.0
            for i in 0..<n { dot += Double(x[i]) * Double(y[i]); nx += Double(x[i]) * Double(x[i]); ny += Double(y[i]) * Double(y[i]) }
            return dot / ((nx * ny).squareRoot() + 1e-12)
        }

        @Test func overlapMatchesSerialAudio() async throws {
            guard let modelDir else { return }
            setenv("CHATTERBOX_SYNTH_SEED", "42", 1)               // seed synth noise
            defer { unsetenv("CHATTERBOX_OVERLAP_CHUNKS"); unsetenv("CHATTERBOX_SYNTH_SEED") }
            let model = try await ChatterboxCoreMLModel.load(from: modelDir, watermarker: NoWatermark())
            let text = String(
                repeating: "This is a fairly long sentence used to exercise chunked generation. ",
                count: 24)
            // Greedy decode → deterministic tokens, so any audio delta is synth-only.
            let opts = GenerationOptions(maxTokens: 100, greedy: true, seed: 42)

            setenv("CHATTERBOX_OVERLAP_CHUNKS", "0", 1)            // serial
            let serial = try await collect(model, text, opts)
            setenv("CHATTERBOX_OVERLAP_CHUNKS", "1", 1)            // overlapped
            let overlap = try await collect(model, text, opts)

            #expect(serial.count > 1)                              // multi-chunk → overlap engaged
            #expect(serial.count == overlap.count)
            for (s, o) in zip(serial, overlap) {
                #expect(s.tokenCount == o.tokenCount)              // identical decode
                #expect(s.samples.count == o.samples.count)
                // Byte-identical on CPU; ≥0.9999 tolerates pre-existing GPU fp16 jitter.
                #expect(cosine(s.samples, o.samples) >= 0.9999)
            }
        }
    }
}
