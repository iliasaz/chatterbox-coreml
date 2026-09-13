# ChatterboxCoreML

On-device, **all-CoreML** text-to-speech for Apple Silicon (macOS 15+ / iOS 18+,
arm64). One Swift package runs **three** Chatterbox model variants behind a shared
pipeline:

| | **Turbo** (English) | **Nano** (English) | **Multilingual** (MTL) |
|---|---|---|---|
| T3 backbone | GPT-2-medium, 24 layers / 1024 hidden / 16 heads | GPT-2-small, 12 layers / 768 hidden / 12 heads | LLaMA_520M, 30 layers (RoPE + RMSNorm + SwiGLU) |
| Decoding | batch-1 | batch-1 | **CFG batch=2** (cond / uncond), host-combined |
| Tokenizer | GPT-2 BPE | GPT-2 BPE (same file) | grapheme BPE + Russian stress ([ruaccent](docs/ruaccent-integration.md)) |
| Languages | English | English | 23 languages (validated on **Russian**) |
| `T3LM.mlpackage` | 328 MB, **8-bit palettized** | 120 MB, **8-bit palettized** | **8-bit palettized** (so prefill loads on the ANE) |
| Synth (S3CFM) | 2-step meanflow | 2-step meanflow (same package) | 10-step classifier-free-guidance |
| HF model repo | [`iliasaz/chatterbox-turbo-coreml`](https://huggingface.co/iliasaz/chatterbox-turbo-coreml) | [`iliasaz/chatterbox-nano-coreml`](https://huggingface.co/iliasaz/chatterbox-nano-coreml) | [`iliasaz/chatterbox-multilingual-coreml`](https://huggingface.co/iliasaz/chatterbox-multilingual-coreml) |

**Nano is turbo with a smaller backbone**: head_dim stays 64, and the S3 synth stack,
the voice-cloning encoders, the tokenizer, and the conds are reused **bit-for-bit**
from turbo (upstream ships sha256-identical files) — only `T3LM` differs. It buys
speed: on an **iPhone 17 Pro Max** ANE prefill runs **22 ms** (cos_sim 0.9996) against
**64 ms** (0.9998) for turbo on the same phone; a Mac M5 Release CLI run renders
**4.16 s of audio in 0.65 s** (~6.4x realtime) vs turbo's 4.00 s in 1.02 s (~3.9x).

All three variants share one orchestration (`ChatterboxCoreMLModel` / `T3Backend`),
the three-package S3Gen synth (`SynthRunner`), chunking, sampling, and audio output,
and run their T3 prefill **and** decode on the **Apple Neural Engine** on iPhone. **The variant is
auto-detected at load**: `perceiver_query.npy` (the multilingual Perceiver cond block)
→ multilingual; else a `speech_emb` width of 768 → nano; else turbo.

> **What this repository is:** the Swift runtime and a demo app. The package lives at
> the root, so it can be a remote SwiftPM dependency; design docs are in
> [`docs/`](docs/). The CoreML models it runs are **downloaded from the Hugging Face
> repos above**. They are converted from Resemble AI's published weights by tooling
> that is not part of this repository, and the on-device parity harness used to
> validate them is not public either — so this is a runtime for those artifacts, not
> the pipeline that builds them. The model repos are public and need no token; one is
> still honoured (`HF_TOKEN`, `hf auth login`) for a private fork or rate limits.

```
Text ─▶ tokenizer ─▶ chunk ─▶ host embedding assembly ─▶ T3 prefill (q_len=512) ┐
        (variant)                                                                ├ one MLState (KV cache)
                                                       ─▶ T3 decode  (q_len=1)   ┘
     ─▶ S3Gen synth (S3Encoder → S3CFM → S3Vocoder, via SynthRunner)
     ─▶ 24 kHz AVAudioPCMBuffer
```

Prefill + decode are two functions of one stateful **multifunction** CoreML
package (`T3LM.mlpackage`) sharing a single `MLState`.

**Architecture:** [`docs/architecture-turbo.md`](docs/architecture-turbo.md)
(turbo / GPT-2), [`docs/architecture-nano.md`](docs/architecture-nano.md) (nano,
with a turbo-vs-nano diff table) and [`docs/architecture-mtl.md`](docs/architecture-mtl.md)
(multilingual / LLaMA_520M, with a turbo-vs-MTL diff table). Wire-level T3 I/O:
[`docs/T3LM-multifunction-contract.md`](docs/T3LM-multifunction-contract.md).

## Usage (Swift)

`ChatterboxCoreMLModel.load(from:)` auto-detects the variant from the directory
contents, so the same load + `generate` works for all three — the difference is the
model directory you point at and (for multilingual) the `GenerationOptions` and
the Russian stress source.

### Turbo / Nano (English)

Identical call sites — nano just points at a nano model dir (same tokenizer, same
sampler, same voices):

```swift
import ChatterboxCoreML

// Load from a local model directory …
let model = try await ChatterboxCoreMLModel.load(from: URL(filePath: "/path/to/model"))
// … or download it from the Hub first:
//   let dir = try await ModelRepository.download { print("download \(Int($0*100))%") }.directory
//   let model = try await ChatterboxCoreMLModel.load(from: dir)

let audio = try await model.generate("Hello there.")            // AVAudioPCMBuffer @ 24 kHz

// A custom cloned voice:
let cloned = try await model.generate("Hello there.",
                                      voice: URL(filePath: "my-voice-conds.safetensors"))
```

### Multilingual (Russian and 22 more languages)

The multilingual path adds (1) a **Russian stress source** plugged into the
tokenizer and (2) `GenerationOptions.multilingual(...)` (min-p sampler, CFG
weight, emotion exaggeration, language tag). See
[`docs/ruaccent-integration.md`](docs/ruaccent-integration.md) for the stress
seam in detail.

```swift
import ChatterboxCoreML

// 1. Russian neural stress (downloads iliasaz/ruaccent-coreml; falls back to manual
//    `+`/U+0301 marks if that fails). Turbo ignores this argument.
let stress = await RuAccentStress.makeOrFallback(
    downloadHFHome: nil,                                   // nil → HF_HOME / HF_HUB_CACHE env
    hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"])

// 2. Load the multilingual bundle (auto-detected via perceiver_query.npy).
let model = try await ChatterboxCoreMLModel.load(
    from: URL(filePath: "/path/to/multilingual-model"),
    russianStress: stress)

// 3. Generate. `.multilingual` sets language tag, min-p, CFG weight, exaggeration.
let opts  = GenerationOptions.multilingual(language: "ru", exaggeration: 1.3, cfgWeight: 0.5)
let audio = try await model.generate("Привет, как дела?", options: opts)

model.loadedVariant        // .multilingual — what actually loaded, regardless of request
```

### Download by variant

```swift
// Pick the repo by variant; all three default to the trailing-closure progress form.
let repoId = ModelRepository.Variant.nano.repoId      // or .turbo / .multilingual
let result = try await ModelRepository.download(repoId: repoId) { frac in
    print("download \(Int(frac * 100))%")
}
let dir = result.directory                            // pass this to `load(from:)`
```

`download` returns a `DownloadResult`, not just the directory, because a fetch can
succeed **without landing what it went for** — `HubApi.snapshot` returns success when
it is offline or cancelled. `result.stillStale` names the managed files that are still
behind the Hub, and `result.isPartiallyApplied` says the snapshot disagrees with the
manifest it was fetched against (some files landed, some didn't) — don't load that one;
re-run a plain `download` (which deletes nothing) and check again. Re-running is also
all a *stale* model needs: `HubApi` re-downloads the files whose content changed
upstream and leaves the rest alone. Only `download(force: true)` deletes the snapshot
first, and only ever on explicit user intent.

### Long text: chunking + streaming (all variants)

The prefill window is fixed, so text longer than one chunk is **split into
sentence-aligned chunks** (`TextChunker`) — never silently truncated.
`generate(_:)` renders all chunks and concatenates. For lower time-to-first-audio,
`generateStream(_:)` yields each chunk as it's ready so you can play chunk *N*
while chunk *N+1* generates:

```swift
for try await chunk in model.generateStream("…long text…", options: opts) {
    try player.enqueue(try AudioOutput.pcmBuffer(from: chunk.samples))
    // chunk.prefillTime / chunk.decodeTime / chunk.synthTime  (all CoreML)
}
```

`load(from:)` accepts each model as an `.mlpackage` (as shipped on the Hub) or an
already-compiled `.mlmodelc`. A package is compiled on first load and the compiled
model is kept at a stable Application Support path, so the one-time on-device ANE
compile (tens of seconds on iPhone) is paid once, not on every launch.

### Voices & voice cloning

A **voice** is a `*-conds.safetensors` conditioning bundle (speaker embeddings +
reference mel + reference speech tokens). Voices are **model-agnostic to load** —
built only by the shared voice encoders, which carry no T3 weights, so any voice
file opens under any of the three variants (nano ships the same encoders as turbo).

**Loading and working are different claims, though.** A reference built from a
short clip degenerates on turbo — under ~5 s of source audio, a one-sentence
request comes back as a fraction of a second or as half a minute of babble — while
the same file behaves on the multilingual model, whose Perceiver resamples the
conditioning prompt onto a fixed query set. So prefer **at least ~10 s** of clean
speech when cloning for turbo or nano; the app's Create Voice screen asks for
25-30 s for this reason. Measured per voice in
[`ChatterboxApp/ChatterboxApp/Voices/README.md`](ChatterboxApp/ChatterboxApp/Voices/README.md).

Create one with `VoiceCloner`, the CLI's `--clone-voice`, or the app's **Create
Voice** screen — all on-device, no Python. That is how the bundled reference voices
were built, so a clone of this repo can regenerate them; `VoiceClonerTests` checks
the result against upstream's PyTorch `prepare_conditionals` output.

Full schema + worked examples:
[`docs/voice-cloning.md`](docs/voice-cloning.md).

### Watermarking (on by default)

Every utterance this pipeline produces carries an inaudible **Perth-Net Implicit**
watermark, the same one upstream chatterbox applies — `ChatterboxTTS.generate` ends
with `apply_watermark(...)` and offers no way to turn it off. Here it runs through
[`perth-coreml`](https://github.com/iliasaz/perth-coreml): the encoder conv stack on
the ANE, the STFT/ISTFT and gating on the host, ~0.3 ms of Neural Engine time for
five seconds of audio.

`ChatterboxCoreMLModel.load` resolves the 4.5 MB `PerthEncoder` from the first place
that has it — an explicit directory, then the model directory itself (so a model repo
that ships it stays offline-capable), then `iliasaz/perth-coreml` on the Hub with the
same token as the model. It **never fails a generation**: if no encoder can be loaded
the audio is emitted unwatermarked and the log says `[watermark] UNAVAILABLE`. Pass
`watermarker: NoWatermark()` to opt out in code (the CLI's `--no-watermark`); there is
deliberately no switch for it in the app.

Because `generateStream` hands each chunk to the caller before the next one exists,
the mark is applied **per chunk** rather than once per utterance. That is not the
same operation — Perth gates every frame against the loudest frame in the signal it
is given — so it is measured rather than assumed: a three-sentence utterance scores
**1.0** whole, and each third of it scores **≥0.9995** on its own, against an
unwatermarked control at **0.0**. Pinned by `WatermarkEndToEndTests`
(`CHATTERBOX_MODEL_DIR=./out PERTH_MODEL_DIR=/path/to/perth swift test`).

To check a file yourself:

```bash
swift run --package-path ../perth-coreml perth-cli out.wav --models <perth-dir> --detect
```

## CLI

```bash
# Download a model and speak. --variant picks the repo (default: multilingual when
# --language is given, else turbo):
swift run chatterbox-cli "Hello there." --download --out out.wav
swift run chatterbox-cli "Hello there." --download --variant nano --out out.wav

# Multilingual — --language sets the language tag, sampler and conditioning; the
# Russian stress models download alongside (or --ruaccent-dir PATH to load locally):
swift run chatterbox-cli "Привет, как дела?" --language ru --download --out ru.wav

# Or point at a model directory you already have (the variant is auto-detected):
swift run chatterbox-cli "Hello there." --model-dir /path/to/model --out out.wav
```

Other flags: `--variant turbo|nano|multilingual`, `--hf-token TOKEN`,
`--voice conds.safetensors`, `--greedy`, `--temperature F`, `--top-p F`,
`--max-tokens N`, `--seed N`, `--perth-dir PATH` / `--no-watermark`, plus the
multilingual `--exaggeration F` / `--cfg-weight F`. `--variant` only selects the download/discovery repo — with
`--model-dir` the variant is auto-detected. Without `--model-dir` the CLI falls
back to `CHATTERBOX_MODEL_DIR`, then `HF_HOME`/`HF_HUB_CACHE` discovery.

## Testbed app (`ChatterboxApp`)

`ChatterboxApp/` is a single SwiftUI testbed that builds and runs **natively on
both macOS and iOS** from one Xcode target (`ChatterboxApp.xcodeproj`, used for
signing). It links the local `ChatterboxCoreML` package.

- A **Model** picker selects the variant (Turbo / Nano / Multilingual); a **Language**
  picker (23 langs, multilingual only) and variant-aware parameter controls
  (top-k for the GPT-2 models, turbo + nano; min-p / exaggeration / cfg-weight for
  multilingual) appear accordingly. The picker snaps to the variant that actually loaded.
- A **Voice** picker selects the speaker — built-in `default-conds`, the bundled
  reference voices for the selected variant (upstream's English demo prompt for
  turbo/nano; one prompt per language, all 23, for multilingual), or your own
  on-device clones via **Create Voice**.
- **iOS** uses one fixed location — the app's `Documents` HF_HOME: Download → Load →
  Generate & Play.
- **macOS** adds a **Source** picker — HF_HOME or a **Local folder** (load a model
  directory directly); the choice is remembered across launches (security-scoped
  bookmarks).
- At launch the app checks whether its snapshot is **behind the Hub** and repairs it
  **in place** (nothing is staged or deleted). An update under 25 MB — the class of
  the CoreML-spec fixes this exists for — applies silently; anything larger is
  *offered* ("Model update available (537 MB) — tap Download"), and the model on disk
  loads meanwhile. Offline / rate-limited / out of space never costs you a working
  model. The only things that ever **delete** a snapshot are **Force re-download**, a
  package with a truncated weight, and a model that won't load — i.e. explicit intent,
  or proof that what's there is broken.

An optional HF **token** field — only needed for a private fork of the model repos —
is persisted in the Keychain (`TokenStore`). The status line reports model-load time
and a per-stage breakdown (prefill / decode / synth / total) after generation.
Open `ChatterboxApp/ChatterboxApp.xcodeproj` and pick **My Mac** or an iOS
destination.

**Signing.** The project ships with no development team. To run it, choose your own
team under *Signing & Capabilities* (and change the `com.iliasaz.*` bundle identifier
if Xcode reports it as taken). Command-line builds with `CODE_SIGNING_ALLOWED=NO`
need neither:

```bash
xcodebuild -project ChatterboxApp/ChatterboxApp.xcodeproj -scheme ChatterboxApp \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

## Dependencies

- [`swift-transformers`](https://github.com/huggingface/swift-transformers) — `Tokenizers` (BPE) + `Hub` (downloads).
- [`swift-safetensors`](https://github.com/jkrukowski/swift-safetensors) — voice conditioning files.
- [`swift-huggingface`](https://github.com/huggingface/swift-huggingface) — snapshot downloads.
- [`ruaccent-coreml`](https://github.com/iliasaz/ruaccent-coreml) (`branch: "main"`) — Russian neural stress (multilingual only). See [`docs/ruaccent-integration.md`](docs/ruaccent-integration.md).
- [`perth-coreml`](https://github.com/iliasaz/perth-coreml) (`branch: "main"`) — Perth-Net Implicit audio watermarking, applied to every utterance (see above).

## Apple Silicon only

Targets Apple Silicon; the app is built **arm64-only** (`EXCLUDED_ARCHS = x86_64`,
`ONLY_ACTIVE_ARCH = YES`). A Release/Profile build still compiles an x86_64 slice
of the *package* (local SwiftPM targets don't inherit the app's arch exclusions),
and `Float16` is unavailable in the x86_64 macOS ABI — so the FP16 helpers in
`MLMultiArray+Helpers.swift` (and the MTL mask builders) are guarded with
`#if arch(arm64)` (the `#else` branch `preconditionFailure`s). Only the arm64
slice is ever linked/run, so **Profile** (Release) builds cleanly.

## Architecture & integration docs

- [`docs/architecture-turbo.md`](docs/architecture-turbo.md) — turbo pipeline, component by component.
- [`docs/architecture-nano.md`](docs/architecture-nano.md) — nano model + what it reuses from turbo (+ diff table).
- [`docs/architecture-mtl.md`](docs/architecture-mtl.md) — multilingual model + runtime (+ turbo diff table).
- [`docs/T3LM-multifunction-contract.md`](docs/T3LM-multifunction-contract.md) — prefill+decode I/O contract.
- [`docs/ruaccent-integration.md`](docs/ruaccent-integration.md) — Russian neural stress.
- [`docs/voice-cloning.md`](docs/voice-cloning.md) — creating `*-conds.safetensors`, and how long a reference clip needs to be.

## Responsible use

Voice cloning makes it easy to put words in someone's mouth. Please don't.

- **Only clone a voice you have permission to use** — your own, or that of someone
  who has agreed to it. Don't use this to impersonate real people, to deceive, or to
  make audio that could pass for a genuine recording of someone.
- **Everything this runtime generates is watermarked** with Perth (see
  [Watermarking](#watermarking-on-by-default)), as upstream Chatterbox does, so
  synthetic speech can be identified after the fact. The app has no switch for it.
  `NoWatermark()` exists for tests and for pipelines that watermark downstream —
  stripping the mark from audio you publish defeats its purpose.
- **The bundled voices are anonymous.** They are built from Resemble AI's own
  published demo prompts, not from recordings of identifiable people.

## Credits & license

**The code** — the Swift package, the app and these docs — is **MIT, © Ilia
Sazonov**. See [`LICENSE`](LICENSE).

**The weights are not mine, and this licence does not cover them.** Everything the
runtime downloads is a CoreML *format conversion* of a model trained and published
by someone else; copyright in those stays with their authors. They are MIT as well
— checked at the source rather than assumed: `ResembleAI/chatterbox-turbo`,
`-nano` and `chatterbox` are each tagged `license:mit` and ungated on the Hub, as
are [Perth](https://github.com/resemble-ai/Perth) and
[RUAccent](https://github.com/Den4ikAI/ruaccent). So the converted `.mlpackage`s
inherit MIT and you may redistribute them with attribution — but the attribution is
owed to the model authors below, not to this repository.
[`NOTICE`](NOTICE) sets out who owns what.

- **[Chatterbox](https://github.com/resemble-ai/chatterbox)** by **Resemble AI** (MIT)
  — the models this project converts (`chatterbox-turbo`, `chatterbox-nano`,
  `chatterbox` multilingual), and the reference implementation every stage here is
  validated against. The converted CoreML artifacts are derivatives of their
  MIT-licensed weights.
- **[Perth](https://github.com/resemble-ai/Perth)** by **Resemble AI** (MIT) — the
  Perth-Net Implicit watermarker, ported in
  [`perth-coreml`](https://github.com/iliasaz/perth-coreml).
- **[RUAccent](https://github.com/Den4ikAI/ruaccent)** by **Denis Petrov** (MIT) —
  Russian lexical stress, ported in
  [`ruaccent-coreml`](https://github.com/iliasaz/ruaccent-coreml).
- **[ebrinz/chatterbox-turbo-coreml-converter](https://github.com/ebrinz/chatterbox-turbo-coreml-converter)**
  — an earlier attempt at getting Chatterbox onto Apple Silicon. This project does
  not use its code, but seeing it work at all is what convinced me the problem was
  tractable and to keep digging.
- Swift dependencies:
  [`swift-transformers`](https://github.com/huggingface/swift-transformers),
  [`swift-safetensors`](https://github.com/jkrukowski/swift-safetensors),
  [`swift-huggingface`](https://github.com/huggingface/swift-huggingface).

## Verification

1. `swift build` / `swift test` — unit tests (NPY reader, Float16 MLMultiArray
   helpers, speaker projection, sampler, constants, padded-prefill assembly,
   the multilingual tokenizer + Perceiver/cond-block parity, ruaccent stress).
2. End-to-end: `CHATTERBOX_MODEL_DIR=/path swift test` (turbo smoke) /
   `CHATTERBOX_MTL_MODEL_DIR=/path swift test` (multilingual smoke), or
   `swift run chatterbox-cli "…" --model-dir … --out out.wav`, or run
   **ChatterboxApp** (macOS or iOS).
3. Watermark: `perth-cli out.wav --models <perth-dir> --detect` should report `1.0`
   for generated audio (see [Watermarking](#watermarking-on-by-default)).
