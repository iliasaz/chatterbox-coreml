# T3LM — stateful multifunction CoreML contract

Status: shipped. One `T3LM.mlpackage` (FP16) exposes two functions —
`prefill` (q_len = 512) and `decode` (q_len = 1) — sharing a single `MLState`
(the 24-layer GPT-2 KV cache). Both are ANE-resident on iPhone (92–94 %
decode residency, measured in an Instruments trace). This document is the
Swift-side integration contract for the `T3LM.mlpackage` published in the
Hugging Face model repos.

## Why this exists in its current form

We arrived here via two correctness pivots, both ANE-specific:

1. **In-graph mask construction (`(causal + (1-kpm)·mask_neg) · (1-eye)`)
   broke ANE compile** with `std::bad_cast → -14` for the 24-layer stateful
   prefill. Fix: lift mask construction to the Swift host and feed
   `attn_mask` + `write_mask` as inputs. (Decode already worked this way.)
2. **ANE's fused SDPA kernel for `q_len >> 1` silently drops the explicit
   `attn_mask`** and honors only an internal causal flag — pad-key columns
   leak into attention for real query rows. Fix: in the prefill wrapper,
   **manually decompose SDPA** into `matmul → scale → add(mask) → softmax →
   matmul` primitives so ANE cannot match against the fused kernel and is
   forced to use the explicit mask. Decode (q_len = 1) uses the ANE GEMV
   kernel which honors `attn_mask` correctly, so its SDPA stays as
   `F.scaled_dot_product_attention(..., is_causal=False)`.

## Constants

`W = 512` (prefill window, `--prefill-max`), `MAX_SEQ = 1536` (KV cache
depth, `--lm-max-seq`), `H = 1024`, `layers = 24`, `heads = 16`,
`headDim = 64`, speech vocab `6563` (start `6561`, stop `6562`, content
`0…6560`), text vocab `50276`, speaker dim `256`. Constants live in
`Sources/ChatterboxCoreML/Constants.swift`.

## Shared state

Both functions read/write one `MLState` containing two buffers:

| state name | shape | dtype |
|---|---|---|
| `keyCache`  | `(MAX_SEQ, layers·heads·headDim)` = `(1536, 24576)` | Float16 |
| `valueCache` | `(MAX_SEQ, layers·heads·headDim)` = `(1536, 24576)` | Float16 |

Layout is "all layers' K (or V) concatenated along the feature dimension
seq-first". Layer *l*'s K rows occupy columns `[l·1024 .. (l+1)·1024)`.

The Swift host creates one state via `prefillModel.makeState()` and passes
the same object to both `prefillModel.prediction(...)` and
`decodeModel.prediction(...)`. Verified live in the on-device probe: the
state's |sum| goes 0 → ~22 M after prefill, decode sees those writes.

## `prefill` function (q_len = W = 512)

### Inputs

| name | shape | dtype | meaning |
|---|---|---|---|
| `inputs_embeds`      | `(1, W, 1024)`     | Float16 | host-assembled, **front-aligned** (real rows `[0..T_real-1]`, pad at the tail) |
| `position_ids`       | `(1, W)`           | Int32   | real → `0…T_real-1`, pad → `0` |
| `attn_mask`          | `(1, 1, W, W)`     | Float16 | host-built additive (causal upper-tri masked to `mask_neg=-1e4` + pad-column masking, diagonal forced to 0 so a fully-padded query row never softmaxes to NaN) |
| `write_mask`         | `(W, 1)`           | Float16 | `1` for real rows, `0` for pad rows — applied in-graph to K and V before they enter the cache so decode never sees junk K/V |
| `logits_select_mask` | `(1, W, 1)`        | Float16 | one-hot at row `T_real-1` (start-speech token row) — picks the right row out of the W-wide hidden state via `sum(hidden * lsm)` instead of a gather |

### Output

| name | shape | dtype | meaning |
|---|---|---|---|
| `logits` | `(1, 6563)` | Float16 | speech-token logits at row `T_real-1` |

### State effect
Writes the **whole** `(MAX_SEQ, 24576)` cache (real K/V in rows
`[0..W-1]`, zeros after) via one `write_state` per state. A partial
`slice_update` does not compile on ANE; padding the prefill write to the
full buffer matches what decode does.

## `decode` function (q_len = 1)

### Inputs

| name | shape | dtype | meaning |
|---|---|---|---|
| `inputs_embeds` | `(1, 1, 1024)`        | Float16 | host embedding of the current speech token (`SpeechEmbedding.row`) |
| `position_ids`  | `(1, 1)`              | Int32   | absolute GPT-2 position = write row in the cache |
| `update_mask`   | `(MAX_SEQ, 1)`        | Float16 | one-hot at the write row; reused per step, mutate one element |
| `attn_mask`     | `(1, 1, 1, MAX_SEQ)`  | Float16 | additive: `0` for written positions, `mask_neg=-1e4` for not-yet-written |

### Output

| name | shape | dtype | meaning |
|---|---|---|---|
| `logits` | `(1, 6563)` | Float16 | next-token logits |

### State effect
Per layer: read the cache slice, splice the new row via
`new = old·(1-update_mask) + k_row·update_mask`, attend over the full
MAX_SEQ-wide cache, then concatenate all layers and `keyCache[:] = …` (one
`coreml_update_state` per state).

## Host-side assembly (`PrefillAssembler`)

Required artifacts (from the converter output dir / HF repo):

- `speech_emb.npy` `(6563, 1024)` Float32 — `SpeechEmbedding`.
- `text_emb.npy` `(50276, 1024)` Float32 — `EmbeddingTable`.
- `spkr_enc_weight.npy` `(1024, 256)` + `spkr_enc_bias.npy` `(1024,)`
  Float32 — `SpeakerProjection` (256 → 1024 linear).

Build the real (unpadded) sequence, in this exact order:

```
spkr_e   = spkr_enc_weight · speaker_emb + spkr_enc_bias     // (1024,)   one row
cond_e   = speech_emb[ condPromptSpeechTokens[i] ]           // (T_cond, 1024)
text_e   = text_emb[ textTokens[i] ]                         // (T_text, 1024)
speech_e = speech_emb[ Constants.speechStartToken ]          // (1024,)   one row

real     = [ spkr_e ] ++ cond_e ++ text_e ++ [ speech_e ]    // (T_real, 1024)
T_real   = 1 + T_cond + T_text + 1
```

`T_cond` is the conditioning length (≤ 375, fixed per voice). `T_text` is
the number of text tokens.

### Front-align to W

```
inputs_embeds[0, 0       ..< T_real, :] = real      // real rows at the front
inputs_embeds[0, T_real  ..< W,      :] = 0         // pad rows at the tail

position_ids [0, 0       ..< T_real] = 0,1,…,T_real-1
position_ids [0, T_real  ..< W]      = 0

logitsSelectMask[0, T_real-1, 0] = 1                // one-hot at start-speech row, else 0
```

`attn_mask` and `write_mask` are built inside `T3LMRunner` per call (see
`buildPrefillAttnMask` / `buildWriteMask`). They depend on `T_real` and so
must be rebuilt each generation; the host-build cost is negligible
(<1 ms; the matrices are 0.5 MB and 1 KB respectively).

### Truncation policy (lengths over W)

The conditioning is never truncated (it is the voice). If
`1 + T_cond + T_text + 1 > W`, truncate the **text tokens** to
`T_text_max = W - T_cond - 2` before assembly. With `W = 512` and the usual
`T_cond = 375`, that's ~135 text tokens (≈ one long sentence). Use
`TextChunker` to split longer text upstream.

## Decode loop (`T3LMRunner.generate`)

```
state = prefillModel.makeState()

// One prefill call seeds the shared state.
prefillOut = prefillModel.prediction(
    from: { inputs_embeds, position_ids, attn_mask, write_mask, logits_select_mask },
    using: state)
seedLogits = prefillOut.logits          // (1, 6563) f16 → [Float]

token = sample(seedLogits)
for step in 0..<maxTokens {
    if token == stopToken && step >= minTokens { break }
    writePos = T_real + step

    // Reuse pre-allocated MLMultiArrays, mutate one element per step.
    updateMask[prevPos] = 0; updateMask[writePos] = 1
    attnMaskD[writePos] = 0                 // unmask the row we're writing
    pos[0] = writePos

    decodeOut = decodeModel.prediction(
        from: { inputs_embeds = speechEmb[token], position_ids = pos,
                update_mask = updateMask, attn_mask = attnMaskD },
        using: state)
    token = sample(decodeOut.logits)
}
```

The same `state` object is passed to both functions, so prefill's writes are
visible to the decode loop. No per-step `autoreleasepool` is needed — CoreML
state ops don't accumulate per-step outputs.

## Numerical parity

**Mac CPU:** seed cos_sim 0.9997, decode steps
0..5 cos_sim ≥ 0.998 vs the PyTorch reference (FP16 rounding).

**iPhone 17 Pro Max ANE (all 4 compute units):**

| CU | predict | top-5 vs Mac CPU | stop logit | cos_sim |
|---|---|---|---|---|
| cpuOnly             | 210 ms | MATCH | 0.79 | 1.000000 |
| cpuAndGPU           | 1867 ms | last 2 swapped (fp16 rounding) | 0.77 | 0.999680 |
| cpuAndNeuralEngine  | 73 ms | MATCH | 0.76 | 0.999723 |
| all (→ ANE)         | 81 ms | MATCH | 0.76 | 0.999723 |

End-to-end on iPhone for the 39-token default test string: 257 speech
tokens generated, `stoppedNaturally=true`, full audio. Prefill 0.127 s,
decode 8.98 s (≈35 ms/tok wall, 26.7 ms/tok ANE-busy), synth 3.19 s.

## Provenance

`T3LM.mlpackage` is converted from upstream's PyTorch T3 by tooling that is not part
of this repository; the published package is what this contract describes. If you
build your own, validate it on an iPhone and not only on a Mac — Mac CPU correctness
does not imply iPhone ANE correctness for this graph, and the fused-SDPA mask drop
above is the prototype example.
