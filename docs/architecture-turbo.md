# Architecture — Turbo (GPT-2 T3)

> This documents the **turbo** variant. For the **multilingual** (LLaMA_520M)
> variant — CFG batch=2, RoPE/RMSNorm, Perceiver cond block, grapheme tokenizer
> + Russian stress, 8-bit-on-ANE — see [`architecture-mtl.md`](architecture-mtl.md),
> which diffs against this doc. Both variants share the orchestration
> (`ChatterboxCoreMLModel` / `T3Backend`), the S3Gen synth (`SynthRunner`),
> chunking, sampling, and audio output.

`ChatterboxCoreML` runs ResembleAI's **Chatterbox Turbo** TTS on Apple Silicon
(macOS 15+ / iOS 18+) as an **all-CoreML** pipeline. A single stateful
multifunction CoreML package (`T3LM.mlpackage`) handles both the one-shot
prefill **and** the per-token decode loop on the ANE, sharing one `MLState`
for the KV cache. Audio synthesis is three CoreML packages (`S3Encoder` /
`S3CFM` / `S3Vocoder`) orchestrated in Swift by `SynthRunner`.

```
text ─▶ TextTokenizer ─▶ TextChunker ─┐
                                      │  (per chunk, ≤ prefill window)
                                      ▼
  Conditionals ─▶ PrefillAssembler ─▶ T3LMRunner ──────────────────────▶ SynthRunner ─▶ AudioOutput
  (voice)         (host embeddings)    (CoreML T3LM.mlpackage — multifunction   (CoreML S3Encoder        (PCM)
                                       prefill + decode sharing one MLState,     → S3CFM → S3Vocoder)
                                       both ANE-resident on iPhone)
```

## Layout

```
Sources/ChatterboxCoreML/      the library (all pipeline logic)
Sources/ChatterboxCLI/         chatterbox-cli — generate a WAV from the terminal
ChatterboxApp/                 SwiftUI testbed app (macOS + iOS), own .xcodeproj
Tests/ChatterboxCoreMLTests/   unit tests + a gated end-to-end smoke test
docs/                          this file + the prefill I/O contract
```

## Stages

### 1. Tokenize — `TextTokenizer`
GPT-2 BPE via swift-transformers, loaded from the model dir's `tokenizer.json`.
`encode` applies chatterbox's `punc_norm` (capitalize, collapse spaces, normalize
punctuation, ensure a terminal mark) and emits raw `Int32` ids (no special
tokens — the prefill builds its own conditioning prefix).

### 2. Chunk — `TextChunker`
The prefill is a **fixed-length** graph: only `MAX − T_cond − 2` text tokens fit
(≈135 with the default voice). Longer text is split into sentence-aligned chunks
under that budget (greedy sentence packing → word-split → hard token split for a
pathological single word). Token counting is delegated to an injected `encode`
closure so the packing is unit-testable without the model. Each chunk is a
self-contained utterance rendered independently.

### 3. Voice conditioning — `Conditionals`
Loaded from a `*-conds.safetensors` file (`default-conds.safetensors`, or a
custom voice). Holds the T3 `speaker_emb` (256-d) and `cond_prompt_speech_tokens`
(used by the prefill) plus the conditional decoder's CAMPPlus `gen.embedding`
(192-d, L2-normalized on demand), `prompt_token` prefix, and `prompt_feat`
(500×80 mel features).

### 4. Host-side embedding assembly — `PrefillAssembler`
The token-embedding lookups and speaker projection live **on the host** (not
in the CoreML graph), so the graph is a clean fixed-length transformer that
the Neural Engine accepts. The assembler builds the real sequence in contract
order — `[spkr_e] ++ cond_e ++ text_e ++ [speech_start]` — then
**front-aligns** it to `W = 512` (pad at the tail; the converter and decode
both expect real rows at indices `[0..T_real-1]`), producing:
- `inputs_embeds` `(1, W, 1024)` f16, real rows first then pad,
- `position_ids` `(1, W)` i32 (real → `0…T_real-1`, pad → 0),
- `logitsSelectMask` `(1, W, 1)` f16 (one-hot at `T_real-1`).

The matching `attn_mask` `(1,1,W,W)` and `write_mask` `(W,1)` are built inside
`T3LMRunner` per call (see step 5). Embedding tables come from `NPYFloat32`
readers: `SpeechEmbedding` (`speech_emb.npy`), `EmbeddingTable`
(`text_emb.npy`), and `SpeakerProjection` (`spkr_enc_weight.npy` +
`spkr_enc_bias.npy`, a 256→1024 linear).

**Why the masks are built on the host, not in the graph.** Constructing `attn_mask`
from a key-padding mask inside the model — `(causal + (1 − kpm)·mask_neg)·(1 − eye)` —
makes the ANE compiler fail on the 24-layer stateful prefill (`std::bad_cast`, error
`-14`), so the graph takes both masks as plain inputs. `buildPrefillAttnMask` and
`buildWriteMask` look like busywork that could be folded into the model; folding them
in is exactly the change that stops the prefill loading on the ANE.

### 5. Prefill + decode — `T3LMRunner` (CoreML, one multifunction package)
One stateful CoreML model (`T3LM.mlmodelc` / `.mlpackage`) exposing two
functions that share an `MLState` (the KV cache):

- **`prefill`** (q_len = W = 512): consumes the front-aligned
  `inputs_embeds` + `position_ids` + `attn_mask` + `write_mask` +
  `logits_select_mask`, runs all 24 GPT-2 layers in one shot, writes the full
  `(MAX_SEQ, layers*heads*headDim)` KV cache via `write_state`, returns
  `logits (1, 6563)` at the start-speech row. The wrapper **manually
  decomposes SDPA** into `matmul/scale/add/softmax/matmul` primitives —
  ANE's fused attention kernel for q_len >> 1 silently drops the explicit
  `attn_mask`; manual decomposition forces the compiler to honor it.
- **`decode`** (q_len = 1): consumes the per-token embedding + abs
  position + one-hot `update_mask` + `(1,1,1,MAX_SEQ)` `attn_mask`, splices
  the new K/V row into the cache, attends over the full MAX_SEQ-wide cache
  (q=1 SDPA uses ANE's GEMV kernel which honors `attn_mask` correctly so it
  is **not** decomposed), writes the whole cache back via
  `coreml_update_state`, returns `logits`.

`T3LMRunner` owns one `MLModel` per function (each constructed with
`MLModelConfiguration.functionName`) plus the shared `MLState`. Loaded with
`computeUnits = .all` (Mac → GPU, iPhone → ANE; 92–94 % decode ANE residency
on iPhone 17 Pro Max, measured in an Instruments trace). FP16 conversion lives in
`MLMultiArray+Helpers.swift`. Full I/O contract:
`docs/T3LM-multifunction-contract.md`. Each decode step runs without an
explicit `autoreleasepool` — CoreML state ops don't accumulate per-step outputs.

### 6. Sampling — `Sampler`
Per step: repetition penalty over a recent window → temperature → top-k → top-p
(nucleus), or greedy. The stop token is forbidden before `minTokens`. Defaults
mirror chatterbox `inference_turbo` (temp 0.8, top-k 1000, top-p 0.95, rep
penalty 1.2). Optional seed (`SeededGenerator`) for reproducible sampling.
Configured via `GenerationOptions`. Driven by `T3LMRunner.generate` between
each `decode` function call.

### 7. Synthesis — `SynthRunner` (three CoreML packages)
Orchestrates the S3Gen synth in Swift, doing the glue between the three packages
that a single fused decoder graph would otherwise hide:
- **S3Encoder** `speech_tokens (1,T) i32 → mu (1,80,T_h=2T)` — prepends the
  voice's `prompt_token`s to the generated tokens (clamped to 0…6560). Runs on
  `cpuAndNeuralEngine`.
- Build conditioning: `mel_len1 = prompt_feat` frames; `cond (1,80,T_h)` =
  `prompt_feat` in the first `mel_len1` frames else 0; all-ones `mask`;
  `spks_raw` = L2-normalized CAMPPlus xvec (192); `z = randn(1,80,T_h)`
  (Box–Muller; `CHATTERBOX_SYNTH_SEED` makes it reproducible for tests).
- **S3CFM** mean-flow Euler loop (no CFG): `t_span = linspace(0,1,n+1)`, each
  step `v = estimator(x,mu,cond,spks_raw,t,r,mask); x += (r−t)·v`. Runs on
  `cpuAndGPU` (**mandatory** — `spk_embed_affine` is baked in; other CUs crash
  the BNNS AOT compiler). Step count via `CHATTERBOX_SYNTH_CFM_STEPS` (default 2).
- Slice `feat = x[:, :, mel_len1:]` → **S3Vocoder** `feat (1,80,T_gen) →
  waveform (1, T_gen·480)`, returning a 24 kHz `[Float]`. `cpuAndGPU` (fast,
  fp16-ISTFT) by default; `cpuOnly` is the cleaner fp32 path.

Per-stage compute units are overridable via `CHATTERBOX_SYNTH_{ENC,CFM,VOC}_CU`.
Non-deterministic (self-seeds CFM noise unless `CHATTERBOX_SYNTH_SEED` is set) —
don't byte-compare audio; compare token sequences.

### 8. Audio — `AudioOutput`
Wraps `[Float]` into a 24 kHz mono `AVAudioPCMBuffer`; also writes WAV (CLI).

## Orchestration — `ChatterboxCoreMLModel`
An `actor` that owns the runners and embedding tables. `load(from:)` resolves the
model dir (accepting `.mlmodelc` or `.mlpackage`), loads the npy tables, the
tokenizer, the T3LM and synth CoreML models, and the default voice. Two entry
points:

- **`generate(_:)`** — chunks the text, renders every chunk, concatenates the
  audio into one buffer.
- **`generateStream(_:)`** — an `AsyncThrowingStream<AudioChunk>` that yields one
  chunk as soon as it's ready. The actor serializes generation while the consumer
  plays the previous chunk → **pipelined playback** (lower time-to-first-audio).
  Cancelling the consuming task cancels generation. Each `AudioChunk` carries the
  chunk's `samples` plus per-stage timings (`prefillTime`, `decodeTime`,
  `synthTime` — all CoreML) and `tokenCount`, so callers can report where time
  went.

`AudioChunk` (Sendable) crosses the actor boundary; the caller builds the
`AVAudioPCMBuffer` from `chunk.samples`. In the app, `Player.enqueue` schedules
buffers back-to-back (options `[]`, not `.interrupts`) for gapless playback, and
the status line shows the prefill/decode/synth/total breakdown.

## Model acquisition — `ModelRepository`
Default repo is the public `iliasaz/chatterbox-turbo-coreml`.
`download(repoId:hfHome:hfToken:)` fetches only the runtime files
(`runtimeGlobs`) via swift-transformers' `Hub.HubApi`. Auth token resolution:
explicit `hfToken` → `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → (else) `HubApi`'s
own `TokenProvider.environment` (which also reads `hf auth login` token files).
`existingModelDirectory(...)` discovers an already-downloaded snapshot across the
swift-transformers / `huggingface_hub` / `--local-dir` layouts under an `HF_HOME`
root. The **local-folder** path (`load(from:)` on an arbitrary dir) needs no Hub
and no auth.

In the app, `TokenStore` (Keychain generic-password) persists an optional HF token,
needed only for a private fork of the model repos.

## Constants — `Constants`
Architecture sizes baked in (the HF bundle ships no `config.json`): speech vocab
6563, start-speech token **6561**, stop 6562, valid generated range 0…6560;
GPT-2 hidden 1024 / 16 heads / 24 layers / head-dim 64; speaker 256, CAMPPlus
192, mel 80; 24 kHz.

## Platform notes
Apple-Silicon-only. The app builds **arm64-only** (`EXCLUDED_ARCHS = x86_64`,
`ONLY_ACTIVE_ARCH = YES`), but a Release/Profile build still compiles an x86_64
slice of the SwiftPM package (local package targets don't inherit the app's arch
exclusions). Because `Float16` is unavailable in the x86_64 macOS ABI, the FP16
helpers are guarded with `#if arch(arm64)` (with `preconditionFailure` on the
`#else` branch so the x86_64 slice compiles and would crash loudly if anyone
ever invoked it); only the arm64 slice is linked/run. T3LM uses
`computeUnits = .all` (Mac → GPU, iPhone → ANE); the synth packages pin their
own compute units (encoder ANE, CFM + vocoder GPU). Large
model → the iOS target carries the `increased-memory-limit` /
`extended-virtual-addressing` entitlements (SDK-scoped to iOS so the macOS
build signs cleanly).
