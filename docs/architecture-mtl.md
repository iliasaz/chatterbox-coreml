# Architecture — Multilingual (MTL)

`ChatterboxCoreML` also runs ResembleAI's **Chatterbox-Multilingual** TTS (23
languages, validated on Russian) on Apple Silicon (macOS 15+ / iOS 18+) as the
same all-CoreML pipeline that ships **turbo**. The two variants share one
orchestration (chunking, streaming, synth) behind a `T3Backend` protocol; only
the T3 front end forks. This doc describes the **converted MTL T3 model + its
runtime** and calls out what differs from turbo. For the turbo pipeline see
[`architecture-turbo.md`](architecture-turbo.md); for the wire-level prefill/decode I/O see
[`T3LM-multifunction-contract.md`](T3LM-multifunction-contract.md).

The headline structural difference: the MTL T3 backbone is **LLaMA_520M** (30
layers, RoPE, RMSNorm, SwiGLU, bias-free split q/k/v/o) and decoding runs as a
**CFG batch=2** single ANE pass (lane 0 conditional, lane 1 unconditional). The
host combines the two lanes' logits before sampling.

```
text ─▶ MTLTextTokenizer ─▶ TextChunker ─┐
       (grapheme BPE +                    │  (per chunk, ≤ prefill window)
        Russian Unicode)                  ▼
  Conditionals ─▶ MTLPrefillAssembler ─▶ MTLT3LMRunner ─────────────▶ SynthRunner ─▶ AudioOutput
  (voice)         (CondBlockBuilder:      (CoreML T3LM.mlpackage,       (S3Encoder       (PCM)
                   Perceiver + emotion;    multifunction prefill+decode  → S3CFM-CFG
                   host CFG batch=2)       batch=2 CFG, shared MLState;   → S3Vocoder)
                                           host cfgCombine; both fns
                                           ANE-resident, 8-bit)
```

## Layout

Identical to turbo — the MTL artifacts live in the same model dir and are
selected by which `T3LM.mlpackage` (+ host tables) is loaded. Variant
auto-detect: the presence of `host_tables/perceiver_query.npy` flags
multilingual (`ChatterboxCoreMLModel.load`). Runtime:
`Sources/ChatterboxCoreML/MTL*.swift`, `Perceiver.swift`,
`CondBlockBuilder.swift`, `T3Backend.swift`.

## Backbone — LLaMA_520M (vs turbo's GPT-2-medium)

The converted weights are upstream's `t3_mtl23ls_v3.safetensors` (`T3Config.multilingual()`).
Geometry:

| | value |
|---|---|
| layers | **30** (`LAYERS`) |
| hidden | 1024 (`HIDDEN`) |
| heads | 16 (`HEADS`), head-dim 64 (`HEAD_DIM`), **no GQA** (q/k/v all 16 heads) |
| attn proj | split bias-free `q_proj`/`k_proj`/`v_proj`/`o_proj` (no fused `c_attn`) |
| norm | `LlamaRMSNorm` (replaced by `HardenedRMSNorm`, below) |
| MLP | SiLU **SwiGLU**: `down_proj(silu(gate_proj(x)) · up_proj(x))` |
| positions | **RoPE in-graph** `θ=500000` + **llama3 scaling**, *and* a learned input pos-emb added host-side |
| speech vocab | 8194 (`SPEECH_VOCAB`); start 6561, stop 6562, content 0…6560 |
| text vocab | 2454 (`TEXT_VOCAB`); SOT 255, EOT 0 |

### RoPE (inline, static)

Transformers' rotary helpers emit untraceable `aten::Int`, so RoPE is
hand-inlined from the captured `inv_freq` (already llama3-scaled) + the float
`attention_scaling`, inside the converted graph:
`freqs = pos · inv_freq; emb = cat([freqs, freqs]); cos = emb.cos()·scaling;
sin = …`, with `_rotate_half` using a **static** `HEAD_DIM//2` split.
Keys are **rotated before caching** so decode's later queries
attend pre-rotated keys.

### fp16 RMSNorm hardening — MTL-specific

HF `LlamaRMSNorm` upcasts to fp32 internally, but `compute_precision=FLOAT16`
erases that upcast, so `mean(x²)` overflows fp16 once any activation exceeds
~256 (`x²>65504 → inf → rsqrt→0 → logits exactly 0`). At 30 layers with real
activations this fires on **26/31 layer outputs** (max |hidden| ≈ 917,
measured in `validate`). `HardenedRMSNorm`
factors out `m = max|x|` so only normalized values (≤1) are squared:

```
out = weight · (x/m) · rsqrt(mean((x/m)²) + eps/m²)      # m = max|x|, clamp 1e-4
```

Algebraically identical to `LlamaRMSNorm`, ANE-friendly, applied to every
`input_layernorm` / `post_attention_layernorm` / final `norm`. A/B at full
depth: hardened = worst CFG cos **0.999337**, 9/9 tokens; plain HF fp16 norm =
**0.38**, 2/9 (seed wrecked). Toggle off with `--plain-norm` to reproduce the
overflow. (Turbo's GPT-2 LayerNorm has no equivalent overflow path.)

## Stages

### 1. Tokenize — `MTLTextTokenizer` (vs turbo's GPT-2 BPE)

Grapheme **BPE** (`grapheme_mtl_merged_expanded_v1.json`, **2454** vocab) loaded
via swift-transformers. `encode(_:language:)` runs the MTL preprocessing chain
(`MTLTextTokenizer.swift:125-144`):

```
lowercase → NFKD → [ru stress] → '+'→U+0301 → "[<lang>]" prefix
          → " "→"[SPACE]" → encode(addSpecialTokens:false)
          → prepend SOT(255), append EOT(0)
```

NFKD (`decomposedStringWithCompatibilityMapping`) performs the mandatory
pure-Unicode Russian steps (`ё`→е+U+0308, `й`→и+U+0306). Russian lexical stress
is a pluggable `RussianStressing` seam: `ManualRussianStress` (default; caller
`+`/U+0301 marks survive), `DictionaryRussianStress` (per-word exact-match, OOV
passthrough), or the neural `iliasaz/ruaccent-coreml` adapter wired via
`load(russianStress:)`. The seam receives already-lowercased+NFKD text (matching
Python's `add_russian_stress`-after-`preprocess_text` order). `[START]`/`[STOP]`
are added by id (255/0), not via the tokenizer's special-token machinery.

### 2. Chunk — `TextChunker` (via `MultilingualBackend.chunk`)

Same chunker as turbo, but the prefill budget is **fixed**: the cond prefix is
always 34 rows, so `prefillBudget = window − 34 − 1` independent of prompt
length (`T3Backend.swift:67`). The chunker measures with `tokenizer.encode(_:language:)`
(including `[lang]` + SOT/EOT) so it counts exactly what the prefill consumes.
Budget is `min(prefillBudget, vocoderBudget)` (`T3Backend.swift:67-74`).

### 3. Voice conditioning — `Conditionals`

Same artifact (`*-conds.safetensors`) and same shared voice encoders as turbo —
**conds are model-agnostic** (produced only by ve.pt/CAMPPlus/S3Tokenizer/mel,
which are byte-identical across turbo↔multilingual), so voices interchange
freely. MTL consumes `speakerEmb` (256-d), `condPromptSpeechTokens` (Perceiver
input), and the synth-side CAMPPlus xvec / `promptFeat` / `promptToken`.

### 4. Host-side cond + prefill assembly — `MTLPrefillAssembler` (vs turbo's `PrefillAssembler`)

Token-embedding lookups, the speaker projection, **the Perceiver cond block**,
and CFG lane construction all live on the host (decision #6 — the cond block is
text-independent, computed once per voice). The MTL cond prefix is **34 rows**
(`MultilingualConstants.condRows`), not raw cond speech-token rows:

```
cond   = [ spkr(1) ; perceiver(prompt→32) ; emotion(1) ]      # (34, 1024)
text   = text_emb(tok) + text_pos_emb(0…n-1)                  # learned input pos-emb
BOS    = speech_emb(6561) + speech_pos_emb(0)
```

`CondBlockBuilder.build` (`CondBlockBuilder.swift:46-53`) assembles the 34 rows:
`spkr_enc` (256→1024 linear), `Perceiver.forward` (→32 rows), and
`emotion_adv_fc.weight · exaggeration` (1→1024, **no bias**). The Perceiver
(`Perceiver.swift`) is host-side standard 4-head SDPA (one shared `AttentionBlock2`
applied cross then self; LayerNorm eps 1e-5; scale 256⁻⁰·⁵; residual on the
query side; the `einsum` branch is dead code). cos 0.9999 vs PyTorch.

The assembler builds **two CFG lanes** front-aligned into `inputs_embeds (2, W,
1024)` (`MTLPrefillAssembler.swift:92-111`):

- **lane 0 (cond)** = cond + text + BOS,
- **lane 1 (uncond)** = cond + **zeroed text** + BOS — only the text embedding is
  zeroed; cond + BOS are shared. Mirrors `T3.prepare_input_embeds`
  (`text_emb[1].zero_()`).

Real rows occupy the front `[0..realLength)`, pad at the tail (`realLength =
34 + n_text + 1`); decode appends from row `realLength`. The host tables ship
in the model repo:
`text_emb`, `speech_emb`, `text_pos_emb`, `speech_pos_emb`, `spkr_enc_{weight,bias}`,
`emotion_adv_fc_weight`, and the `perceiver_*` tables.

`position_ids`, `logits_select_mask` are produced here; `attn_mask` /
`write_mask` are built per-call in `MTLT3LMRunner` (shared across lanes —
identical to the turbo contract).

### 5. Prefill + decode — `MTLT3LMRunner` (one multifunction package, batch=2)

One stateful CoreML model (`T3LM.mlpackage`) exposing `{prefill, decode}` that
**share one `MLState`** (the KV cache), built via `MultiFunctionDescriptor` +
`save_multifunction` (weights deduped: 1029 MB merged vs 2053 MB separate).
`MTLT3LMRunner` (`MTLT3LMRunner.swift`) owns one `MLModel` per function (each
with `MLModelConfiguration.functionName`) plus the shared `state =
prefillModel.makeState()` passed to both predictions.

**Manual SDPA decomposition runs in BOTH prefill AND decode** (vs turbo, where
only prefill is decomposed). MTL keeps decode decomposed too
(`matmul(q,kᵀ)·scale + mask → softmax → matmul(·,v)`) — proven ANE-safe;
the fused `F.sdpa` made no numerical difference.

- **`prefill`** (q_len = W = 512, batch=2): consumes front-aligned
  `inputs_embeds (2,W,1024)` + `position_ids (1,W)` + `attn_mask (1,1,W,W)` +
  `write_mask (W,1)` + `logits_select_mask (1,W,1)`; runs all 30 LLaMA layers;
  writes the full KV cache; returns
  `logits (2, 8194)` at the BOS row.
- **`decode`** (q_len = 1, batch=2): consumes `inputs_embeds (2,1,1024)` +
  `position_ids (1,1)` + `update_mask (MAX,1)` + `attn_mask (1,1,1,MAX)`; splices
  the new rotated k/v row via the one-hot `update_mask`, attends over the full
  MAX-wide cache, writes the whole cache back; returns `logits (2, 8194)`.

**KV state — encoding B**: two buffers
`keyCache`/`valueCache` of shape `(MAX_SEQ, 2·FEAT)` fp16, with **batch folded
into the feature axis** (reshaped to batch in-graph; dodges CoreML's
leading-batch-dim state corruption). `FEAT = LAYERS·LS = 30·1024 = 30720`,
`LS = HEADS·HEAD_DIM = 1024`, so the buffer is `(1536, 61440)`. Column layout:
`col = lane·FEAT + layer·LS + d`. Decode reads it via turbo-style **2D column
slices** (no 4D reshape/permute of the 188 MB tensor — that overran the
on-device BNNS AOT compiler at 30 layers). Prefill writes rows
`[0..T_real-1]` as a full-buffer overwrite (partial `slice_update` doesn't
compile on ANE); decode appends one row/step via the one-hot full-buffer rewrite.

**Per-decode-step host work** (`MTLT3LMRunner.swift:168-200`): the LLaMA backbone
uses RoPE **and** a learned `speech_pos_emb`, so `inputs_embeds` for each step is
`speech_emb(token) + speech_pos_emb(step+1)`, **duplicated for both CFG lanes**
(`decodeEmbed`). The decode embed is the only per-step host cost
beyond mask/pos mutation.

**Host CFG-combine.** Every predict returns `logits (2, V)`; before sampling the
host combines `out[i] = cond[i] + cfgWeight·(cond[i] − uncond[i])`
(`cfgCombine`, `MTLT3LMRunner.swift`; mirrors upstream's CFG
combine). The combined `(V,)` logits feed `Sampler`. `cfgWeight` default
0.5 (`MultilingualConstants.defaultCfgWeight`).

**Compute units / load — 8-bit on the ANE.** The shipped MTL `T3LM` is **8-bit
palettized**:
k-means, deterministic so the shared weights palettize byte-identically in
prefill and decode (merge still dedups). The **`speech_head` is skipped** (its
weight axis is `SPEECH_VOCAB` — output precision drives stop-token timing).
This halves the package to ~524 MB and is what lets prefill load on the ANE: at
FP16 the 30-layer prefill graph (q_len=512, ~980 MB weights) **fails the ANE
plan build** (`-14` "Failed to build the model execution plan") while decode
(q_len=1, same size) always loaded fine; 8-bit drops the prefill macho weight
footprint under the ANE wall, and **both functions then load on the ANE**
(device-verified, iPhone 17 Pro Max / iOS 26.5.1). `MTLT3LMRunner` loads via a
CU fallback chain `[.all, .cpuAndGPU, .cpuOnly]` (`MTLT3LMRunner.swift:60-93`) —
`.all` routes to ANE on iOS; the chain self-heals an unexpected too-large model
to GPU. Pin one CU with `CHATTERBOX_DECODE_CU`.

### 6. Sampling — `Sampler` (min-p, no top-k)

MTL order is **rep-penalty → temperature → min-p → top-p** (matches
`T3.inference`'s `MinPLogitsWarper`→`TopPLogitsWarper`). The **top-k branch is
dropped** (turbo uses top-k/top-p). `GenerationOptions.multilingual` carries
`language`, `exaggeration` (feeds `emotion_adv_fc`), `cfgWeight`, and `minP`
(default 0.05). Real-world Russian server defaults: temp 0.8, exaggeration 1.3,
cfg_weight 0.5, top_p 0.95, rep-penalty 1.2.

### 7. Synthesis — `SynthRunner` (S3CFM in multilingual-CFG mode)

The three-package S3Gen synth is **variant-agnostic and auto-detected**
(`SynthRunner.swift`): the multilingual S3Gen is the **non-meanflow** checkpoint
(upstream's `s3gen.pt`). `SynthRunner` detects the CFM mode from the graph — the
multilingual S3CFM exposes a `spks_scale` input (turbo's meanflow CFM has `r`
instead):

- **S3Encoder** — re-exported with MTL weights; `speech_tokens → mu`. ANE.
- **S3CFM (multilingual-CFG)** — a **10-step cosine-schedule, classifier-free
  guidance** Euler loop. `t_span(s) = 1 − cos(s·π/2)`. Each step
  runs **TWO** estimator predicts: conditional (real mu/cond, `spks_scale=1`)
  and unconditional (zeroed mu/cond, `spks_scale=0`), combined
  `dxdt = (1+cfgRate)·v_cond − cfgRate·v_uncond` with `cfgRate = 0.7` (the s3gen
  `inference_cfg_rate` flow constant — **not** the T3 token `cfgWeight`); then
  `x += (r−t)·dxdt`. (Turbo: 2-step meanflow, one predict/step, no CFG.) Step
  count via `CHATTERBOX_CFM_STEPS` (default 10 MTL / 2 turbo). **Pinned
  `cpuAndGPU`** — every other CU SIGBUSes the on-device BNNS compiler. ~1.9
  s/utterance (20 predicts).
- **S3Vocoder** — reused byte-identical (mel2wav same as turbo). `cpuAndGPU`.

The rest of the synth glue (conditioning build, prompt-frame slicing, T%4
encoder padding, token windowing) is identical to turbo. Per-stage CUs
overridable via `CHATTERBOX_SYNTH_{ENC,CFM,VOC}_CU`.

### 8. Audio — `AudioOutput`

Identical to turbo: `[Float]` → 24 kHz mono `AVAudioPCMBuffer` (+ WAV in the CLI).

## Orchestration — `ChatterboxCoreMLModel` + `T3Backend`

One `actor` drives both variants behind the `T3Backend` protocol
(`T3Backend.swift`): `TurboBackend` (`TextTokenizer` + batch-1 `T3LMRunner`) and
`MultilingualBackend` (`MTLTextTokenizer` + batch-2 `MTLT3LMRunner`). The backend
exposes `maxLen`, `chunk(...)`, and `generate(...) → T3Result` so the shared
chunk/stream/synth orchestration is variant-agnostic. `load(from:russianStress:)`
auto-detects the variant via `perceiver_query.npy` and selects the MTL tokenizer,
runner, host tables, and the multilingual stress source. The CLI exposes
`--language`, `--exaggeration`, `--cfg-weight`; the app has a variant picker +
language Picker (23 langs) and variant-aware parameter controls.

## Constants — `MultilingualConstants` (vs `Constants`)

Distinct from the turbo `Constants` (GPT-2-medium) so both coexist
(`Constants.swift:47-76`): hidden 1024 / heads 16 / head-dim 64 / **layers 30**;
speech vocab **8194** (start 6561, stop 6562, max-valid 6560 — shared with
turbo); **text vocab 2454** (SOT 255, EOT 0); speaker 256; perceiver query 32;
`condRows` = **34** (1+32+1); **batch 2**; window 512; MAX_SEQ 1536; default
cfg-weight 0.5. The HF bundle ships no `config.json`, so these are fixed in code
mirroring the converter.

## vs turbo — diff table

| Aspect | Turbo | Multilingual (MTL) |
|---|---|---|
| Backbone | GPT-2-medium, 24 layers, fused `c_attn`, LayerNorm, GELU | LLaMA_520M, **30 layers**, split bias-free q/k/v/o, **RMSNorm**, SiLU **SwiGLU** |
| Positions | learned `wpe` only | **RoPE θ=500000 + llama3 scaling** (in-graph) **and** learned input pos-emb (host) |
| fp16 norm | LayerNorm (no overflow path) | **`HardenedRMSNorm`** (rescale-in-fp16; required — 26/31 layers overflow) |
| CFG / batch | none, **batch=1** | **batch=2** (lane0 cond / lane1 text-zeroed uncond); host `cfgCombine` |
| KV state | `(MAX_SEQ, 24·1024)` | **encoding B `(MAX_SEQ, 2·FEAT)`**, FEAT=30·1024=30720; col=lane·FEAT+layer·LS+d |
| Cond prefix | speaker + raw cond speech-token rows | **spkr(1) + Perceiver(32) + emotion(1) = 34 rows** (host) |
| Manual SDPA | prefill only (decode = fused `F.sdpa`) | **prefill AND decode** decomposed |
| Tokenizer | GPT-2 BPE, 50276 vocab | grapheme BPE, **2454** vocab, `[ru]`/`[en]` + Russian stress seam |
| Speech vocab | 6563 | **8194** (start/stop/content range shared) |
| Sampler | rep→temp→top-k→top-p | rep→temp→**min-p**→top-p (**no top-k**) |
| Decode embed | `speech_emb(tok)` | `speech_emb(tok) + speech_pos_emb(step+1)`, **both lanes** |
| Precision / load | FP16, ANE | **8-bit palettized** (`speech_head` skipped) — FP16 prefill fails ANE plan-build; 8-bit loads both fns on ANE |
| S3CFM | 2-step meanflow, 1 predict/step, no CFG | **10-step cosine CFG**, 2 predicts/step, `dxdt=(1+0.7)·v_cond−0.7·v_uncond` |

## See also

- [`architecture-turbo.md`](architecture-turbo.md) — the turbo pipeline (the structural
  baseline this doc diffs against).
- [`T3LM-multifunction-contract.md`](T3LM-multifunction-contract.md) — the
  wire-level prefill+decode I/O contract and shared-state mechanics (turbo
  numbers; the MTL contract mirrors it with batch=2 / vocab 8194 / encoding B,
  as described above).
