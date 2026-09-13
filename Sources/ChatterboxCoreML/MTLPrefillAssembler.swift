import Foundation

/// Host-side assembly for the **multilingual** batch-2 CFG prefill
/// (`T3LM` "prefill" function). Builds the
/// front-aligned `inputs_embeds (2, W, 1024)` plus the position / select tensors.
///
/// Lane 0 = cond + text + BOS (conditional); lane 1 = cond + **zeroed** text + BOS
/// (CFG unconditional — only the text embedding is zeroed, cond + BOS are shared).
/// This mirrors `T3.prepare_input_embeds` (`text_emb[1].zero_()`) and the
/// converter's `build_cfg_prefix`.
///
///   cond   = [spkr(1); perceiver(prompt→32); emotion(1)]   (CondBlockBuilder)
///   text   = text_emb(tok) + text_pos_emb(0…n_text−1)      (learned input pos-emb)
///   BOS    = speech_emb(6561) + speech_pos_emb(0)
///
/// Real rows occupy the front `[0..realLength)`, pad at the tail; decode appends
/// from row `realLength`. Free of CoreML so the contract is unit-testable.
struct MTLPrefillAssembler: Sendable {
    let textEmb: EmbeddingTable        // (2454, 1024)
    let textPosEmb: EmbeddingTable     // (≥max_text, 1024)
    let speechEmb: SpeechEmbedding     // (8194, 1024)
    let speechPosEmb: EmbeddingTable   // (≥max_speech, 1024)
    let condBuilder: CondBlockBuilder
    let window: Int
    let hidden: Int

    init(textEmb: EmbeddingTable, textPosEmb: EmbeddingTable,
         speechEmb: SpeechEmbedding, speechPosEmb: EmbeddingTable,
         condBuilder: CondBlockBuilder,
         window: Int = MultilingualConstants.window,
         hidden: Int = MultilingualConstants.hidden) {
        self.textEmb = textEmb; self.textPosEmb = textPosEmb
        self.speechEmb = speechEmb; self.speechPosEmb = speechPosEmb
        self.condBuilder = condBuilder
        self.window = window; self.hidden = hidden
    }

    struct Assembled: Sendable {
        var inputsEmbeds: [Float]      // flat (2 * W * hidden), C-order (lane outermost)
        var positionIds: [Int32]       // (W): reals→0…realLength-1, pad→0
        var logitsSelectMask: [Float]  // (W): one-hot at realLength-1 (BOS row)
        var realLength: Int            // T_cond(34) + T_text + 1(BOS)
        var condRows: Int              // 34
    }

    /// Builds the conditioning-prompt speech embedding the Perceiver consumes:
    /// `speech_emb(tok) + speech_pos_emb(i)` for i in 0…N-1, flat `(N, 1024)`.
    func condPromptSpeechEmb(_ promptTokens: [Int32]) -> [Float] {
        var out = [Float](repeating: 0, count: promptTokens.count * hidden)
        for (i, t) in promptTokens.enumerated() {
            let se = speechEmb.row(Int(t))
            let pe = speechPosEmb.row(i)
            let base = i * hidden
            for d in 0..<hidden { out[base + d] = se[d] + pe[d] }
        }
        return out
    }

    func assemble(textTokens: [Int32], conds: Conditionals, exaggeration: Float) -> Assembled {
        let condRows = MultilingualConstants.condRows
        // Conditioning is never trimmed; trim text so realLength fits the window.
        let textBudget = max(0, window - condRows - 1)
        let usedText = textTokens.count > textBudget ? Array(textTokens.prefix(textBudget)) : textTokens
        let nText = usedText.count
        let realLength = condRows + nText + 1

        // 1. cond block (34, 1024) — shared by both lanes.
        let promptTokens = conds.condPromptSpeechTokens
        let cond = condBuilder.build(
            speakerEmb: conds.speakerEmb,
            condPromptSpeechEmb: condPromptSpeechEmb(promptTokens),
            promptLen: promptTokens.count,
            exaggeration: exaggeration)

        // 2. text rows (n_text, 1024): text_emb(tok) + text_pos_emb(pos).
        var text = [Float](repeating: 0, count: nText * hidden)
        for (p, t) in usedText.enumerated() {
            let te = textEmb.row(Int(t))
            let pe = textPosEmb.row(p)
            let base = p * hidden
            for d in 0..<hidden { text[base + d] = te[d] + pe[d] }
        }

        // 3. BOS row: speech_emb(start) + speech_pos_emb(0).
        let bosE = speechEmb.row(MultilingualConstants.speechStartToken)
        let bosP = speechPosEmb.row(0)
        var bos = [Float](repeating: 0, count: hidden)
        for d in 0..<hidden { bos[d] = bosE[d] + bosP[d] }

        // 4. front-align into (2, W, hidden): lane0 = cond+text+bos,
        //    lane1 = cond + zeros(text) + bos (CFG uncond zeroes only text).
        var inputsEmbeds = [Float](repeating: 0, count: 2 * window * hidden)
        let laneStride = window * hidden
        for lane in 0..<2 {
            let laneBase = lane * laneStride
            // cond rows
            for r in 0..<condRows {
                let dst = laneBase + r * hidden, src = r * hidden
                for d in 0..<hidden { inputsEmbeds[dst + d] = cond[src + d] }
            }
            // text rows (lane 1 stays zero)
            if lane == 0 {
                for r in 0..<nText {
                    let dst = laneBase + (condRows + r) * hidden, src = r * hidden
                    for d in 0..<hidden { inputsEmbeds[dst + d] = text[src + d] }
                }
            }
            // BOS row
            let bosDst = laneBase + (condRows + nText) * hidden
            for d in 0..<hidden { inputsEmbeds[bosDst + d] = bos[d] }
        }

        var positionIds = [Int32](repeating: 0, count: window)
        var logitsSelectMask = [Float](repeating: 0, count: window)
        for j in 0..<realLength { positionIds[j] = Int32(j) }
        logitsSelectMask[realLength - 1] = 1   // BOS row = seed-logits row

        return Assembled(
            inputsEmbeds: inputsEmbeds,
            positionIds: positionIds,
            logitsSelectMask: logitsSelectMask,
            realLength: realLength,
            condRows: condRows)
    }
}
