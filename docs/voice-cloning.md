# Voice cloning — creating a `*-conds.safetensors`

A **voice** in Chatterbox is a *conditioning bundle*: a small `*-conds.safetensors`
file holding the speaker embeddings, reference speech tokens, and a reference mel
prompt extracted from a few seconds of reference audio. At synthesis time it is
loaded via the runner's `voice: URL?` path (`ChatterboxCoreMLModel.generateStream(_:voice:options:)`)
and fed into T3 prefill (`speaker_emb`, `cond_speech_tokens`) and the conditional
S3 decoder (`embedding`, `prompt_token`, `prompt_feat`).

Make one on-device with `VoiceCloner` (below), the CLI's `--clone-voice`, or the
app's **Create Voice** screen. The same path built the reference voices bundled with
the app (`ChatterboxApp/ChatterboxApp/Voices/README.md`), and `VoiceClonerTests`
grades it against upstream's PyTorch `prepare_conditionals` output.

## Voices load under any variant — but short ones only work on multilingual

A `*-conds.safetensors` carries **no T3 weights**. It is produced *only* by the
shared voice encoders (VoiceEncoder / CAMPPlus / S3Tokenizer / MatchaMel), which are
identical across turbo, nano and multilingual, so any voice file **loads** under any
variant.

**Loading is not the same as working.** A reference built from under ~5 s of audio
degenerates on turbo and nano — asked for one sentence, a 3 s English prompt returned
0.18 s of audio and a 1.4 s one returned 27 s of babble — while the very same files
behave on the multilingual model, whose Perceiver resamples the conditioning prompt
onto a fixed query set. The variable is duration, not language: the 3 s prompt was
English, on the English model. Aim for **10 s or more** of clean, expressive speech
when cloning for turbo or nano; the app's Create Voice passage runs 25-30 s.

That is why the app scopes its *bundled* reference voices per variant (`VoiceDemoSet`
in `ChatterboxApp/ChatterboxApp/VoiceCatalog.swift`). User clones
(`Voice.userVoices()`) are shown for every model — whoever recorded one knows how
long it is.

## The `*-conds.safetensors` schema

Flat keys, batch dims squeezed out. Read by `Conditionals.init(contentsOf:)` and
written by `Conditionals.write(to:)` (`Sources/ChatterboxCoreML/Conditionals.swift`);
the layout is upstream's `Conditionals` with its batch dims removed.

| Key | dtype | shape | Consumer | Meaning |
|-----|-------|-------|----------|---------|
| `t3.speaker_emb` | f32 | `[256]` | T3 prefill `speaker_emb` | VoiceEncoder (VEMel+VELSTM) speaker embedding, mean over partials + L2-normalized |
| `t3.cond_prompt_speech_tokens` | i32 | `[≤375]` | T3 prefill `cond_speech_tokens` | S3 tokens of the 15 s reference window, capped at 375 |
| `gen.embedding` | f32 | `[192]` | S3 decoder `speaker_embeddings` | CAMPPlus speaker embedding, **not yet** L2-normalized (the runner normalizes — `Conditionals.normalizedSpeakerEmbedding`) |
| `gen.prompt_token` | i32 | `[T_tok]` | S3 decoder prompt token prefix | S3 tokens of the 10 s decode window, trimmed so `mel_len == 2·token_len` |
| `gen.prompt_token_len` | i32 | `[1]` | (trim length) | valid length of `gen.prompt_token`; the loader prefixes the row to it |
| `gen.prompt_feat` | f32 | `[T_frames, 80]` | S3 decoder `speaker_features` | MatchaMel reference mel, stored **(frames, bins) C-order** |

Notes:
- The default/built-in voice file is named `default-conds.safetensors`. It doubles
  as the model-directory **discovery marker** (`ModelRepository.existingModelDirectory`
  keys off it); `generateStream(voice:)` interprets `voice == nil` (or the
  `Voice.default`) as "use the model directory's `default-conds.safetensors`".
- `melBins` (80) is fixed in `Constants.swift`; the shipped turbo default has
  `gen.prompt_feat` `[500, 80]` and `gen.prompt_token` `[250]` (mel_len = 2·tok).
- Each model repo's `default-conds.safetensors` is upstream's own built-in voice,
  converted to this schema — not a clip recorded for this project.
- `gen.prompt_token` is trimmed to `gen.prompt_token_len` on load
  (`Conditionals.swift:60`); a producer may store the full row and the length, or
  a pre-trimmed row (both are handled).

## Swift flow — in-app cloning (`VoiceCloner`)

The on-device path runs the same conditioning pipeline entirely in CoreML, with
small host glue, and writes the identical safetensors. No PyTorch, no HF
round-trip. Core type: `VoiceCloner` (`Sources/ChatterboxCoreML/VoiceCloner.swift`).

### The five conditioning models

Shipped in each model repo as waveform-in CoreML packages, with their DSP
front-end baked into the graph. They are loaded by `VoiceCloner.init(modelDirectory:)`
from the resolved model directory (same `.mlpackage`/`.mlmodelc` resolution +
on-load compile as `ChatterboxCoreMLModel.load`):

| Model | I/O | Produces |
|-------|-----|----------|
| `MatchaMel` | `wav24 (1,L)` → `mel (1,80,T)` | `gen.prompt_feat` (pure DSP, no weights) |
| `S3Tokenizer` | `wav16 (1,L)` → `h_pre (1,T,8)` | `gen.prompt_token` + `t3.cond_prompt_speech_tokens` (host: `round()+1` + base-3 sum) |
| `VEMel` | `wav16 (1,L)` → `mel (1,T,40)` | speaker-emb front-end |
| `VELSTM` | `partials (N,160,40)` → `(N,256)` | speaker-emb (host: trim, stride, mean+L2) |
| `CAMPPlus` | `wav16 (1,L)` → `(1,192)` | `gen.embedding` |

These are **fp32** (LSTM/FSQ/StatsPool precision), so they run CPU/GPU and never
the ANE. The default compute unit is `cpuAndGPU`, overridable with
`CHATTERBOX_VC_CU` (`cpu`|`gpu`|`ane`|`all`) — see `VoiceCloner.init`. They are
*not* part of synthesis (T3LM + S3Encoder/CFM/Vocoder); they run once per voice.

### What `VoiceCloner` computes

`makeConditionals(from:)` (`VoiceCloner.swift`) mirrors
`tts.prepare_conditionals`, owning the host glue the converters left out
(loudness, resample, trim, VE partial striding, FSQ token decode, mean/L2 — no
FFT in Swift):

1. Decode + resample to 24 kHz mono (`AudioIO.loadMono`); reject anything under
   **1 s**, and log a warning under 5 s (see above).
2. Loudness-normalize @24 k (`AudioIO.normalizeLoudness`), derive the 16 k stream,
   and cut the windows: 10 s @24 k decode window (`DEC_COND_LEN`), its 16 k
   resample, and the 15 s @16 k encode window (`ENC_COND_LEN`).
3. `gen.prompt_feat` ← `MatchaMel(clip24)`, `(1,80,T)` transposed to `(T,80)` C-order.
4. `gen.embedding` ← `CAMPPlus(clip16)`, `(1,192)`.
5. `gen.prompt_token` ← `S3Tokenizer(clip16)`, trimmed to `featFrames/2` so
   `mel_len == 2·token_len`.
6. `t3.cond_prompt_speech_tokens` ← `S3Tokenizer(enc16)`, capped at 375.
7. `t3.speaker_emb` ← VE chain on the full 16 k stream: `trim → VEMel → stride
   into (N,160,40) → VELSTM → mean over N → L2` (`speakerEmbedding`,
   reproducing `ve.embeds_from_wavs([wav]).mean(0)`; striding via `veNumWins`,
   the port of `voice_encoder.get_num_wins`).

`cloneVoice(from:to:)` wraps `makeConditionals` + `Conditionals.write(to:)`,
which serializes the exact schema above (`Conditionals.write`, `Conditionals.swift:80`).

### Where files are saved

User clones land in `<Documents>/Voices/` — `Voice.userVoicesDirectory`
(`VoiceCatalog.swift`). A file `<slug>-conds.safetensors` there is discovered by
`Voice.userVoices()`, given id `user.<slug>`, and shown in the picker for every
model.

### The app UI — `CreateVoiceView`

`ChatterboxApp/ChatterboxApp/CreateVoiceView.swift` drives the in-app flow:

- Records (mono AAC @24 kHz via `VoiceRecorder`, needs `NSMicrophoneUsage
  Description`) **or** imports an audio file (`fileImporter`); a fresh recording
  takes priority over an import (`sourceURL`).
- Shows a ~25–30 s read-aloud passage deliberately mixing whisper / question /
  exclamation / tender dialogue / triumphant close, so the clone captures a wide
  intonation+emotion range (a monotone read clones a monotone voice).
- On **Create**: slugifies the name, builds `<Documents>/Voices/<slug>-conds.safetensors`,
  then `VoiceCloner(modelDirectory:).cloneVoice(from:to:)` and calls
  `onCreated("user.<slug>")` (`CreateVoiceView.create()`).

`modelDirectory` is the resolved model path — the directory that holds the five
conditioning `.mlpackage`s alongside the synth + T3LM packages.

### Worked example (Swift)

In-app: open **Create Voice**, tap **Record**, read the passage, **Stop**, enter
a name, tap **Create**. The result appears in the Voice picker immediately.

Programmatically (e.g. a test or a CLI tool) the same call:

```swift
import ChatterboxCoreML

let cloner = try await VoiceCloner(modelDirectory: modelDir)  // dir with the 5 .mlpackages
let out = Voice.userVoicesDirectory
    .appendingPathComponent("my_narrator-conds.safetensors")
try FileManager.default.createDirectory(at: out.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
try cloner.cloneVoice(from: referenceClipURL, to: out)   // ≥ 10 s for turbo/nano
// `out` now loads via Conditionals(contentsOf:) under any variant.
```

`VoiceClonerTests` (`Tests/ChatterboxCoreMLTests/VoiceClonerTests.swift`) exercises
the parity path: with `CHATTERBOX_MODEL_DIR` set it feeds the exact wav the Python
oracle used via `cloner.makeConditionals(fromSamples24k:)` (the split-out entry
that isolates model+glue parity from resampler/decoder differences) and compares
against PyTorch-produced reference conds (`Fixtures/female_random_podcast-*`).

## See also

- `Sources/ChatterboxCoreML/Conditionals.swift` — schema reader/writer.
- `Sources/ChatterboxCoreML/VoiceCloner.swift` — on-device orchestrator.
- `ChatterboxApp/ChatterboxApp/Voices/README.md` — the bundled voices: sources, durations, per-variant results.
- `ChatterboxApp/ChatterboxApp/CreateVoiceView.swift`, `VoiceCatalog.swift` — app UI + catalog.
- [`architecture-turbo.md`](architecture-turbo.md) / [`architecture-mtl.md`](architecture-mtl.md) — component maps.
