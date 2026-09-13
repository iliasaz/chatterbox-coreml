# RUAccent integration — Russian neural stress

The multilingual model needs lexical stress (ударение) marked **before**
tokenization: the MTL T3 was trained on grapheme text with the stressed vowel
carrying a combining acute (`U+0301`). Russian orthography doesn't write stress,
so an external accentuator supplies it. We use
[`ruaccent-coreml`](https://github.com/iliasaz/ruaccent-coreml) — a standalone,
all-CoreML, on-device port of [RUAccent](https://github.com/Den4ikAI/ruaccent)
(`turbo3.1`) — plugged in behind the `RussianStressing` seam.

**Multilingual-only.** The turbo (GPT-2) path never sees a stresser:
`ChatterboxCoreMLModel.load(from:russianStress:)` ignores the argument unless the
dir is a multilingual bundle (variant detected by `perceiver_query.npy`;
`ChatterboxCoreMLModel.swift:124`). The CLI/app only build a stresser when
`--language ru` / the multilingual variant is selected.

## What ruaccent does

`RUAccent` (`ruaccent-coreml/Sources/RUAccentCoreML/RUAccent.swift:18`) loads four
CoreML models + four packed `.rapack` dictionaries + tokenizers and runs the
upstream `process_all_internal` pipeline
(`Internal/InternalPipeline.swift:66`). Resolution order, manual mark always
wins → dictionary → homograph model → neural OOV accentor, then `ё` restore:

| Stage | Source | Gate / model |
|---|---|---|
| Manual mark | caller's `+` / `U+0301` preserved verbatim | `ManualStress.canonicalize` |
| ё restore | `yo` dicts | **M4** yo model (`useNeuralAccentor`/`restoreYo`) |
| Homographs | `omographs` dict | **M2** context disambiguation (`useHomographModel`) |
| Accent | `accents` dict, then OOV | **M1** neural accentor (`useNeuralAccentor`) |
| Stress-usage | — | **M3** gate (always loaded; gates `_process_accent`) |

`RUAccent.Configuration` (`RUAccent.swift:21`) can disable M1/M2/M4; a disabled
model isn't loaded. Internal output is RUAccent's native `+`-before-vowel form;
`stress(_:notation:)` (`RUAccent.swift:166`) renders it via `StressNotation`
(`.combiningAcute` default = `U+0301` after the vowel, or `.plusBeforeVowel`).

## The `RussianStressing` seam

chatterbox defines its own `RussianStressing`
(`Sources/ChatterboxCoreML/MTLTextTokenizer.swift:33`) — **sync, non-throwing,
`Sendable`** — distinct from ruaccent's protocol (`throwing`, has `notation:`,
class-bound `AnyObject`). The tokenizer calls it mid-preprocess
(`MTLTextTokenizer.preprocess`, `MTLTextTokenizer.swift:125`):

```swift
var t = text.lowercased()
t = t.decomposedStringWithCompatibilityMapping  // NFKD: ё→е+U+0308, й→и+U+0306
if language == "ru" {
    t = stresser.stress(t)                       // RussianStressing seam
    t = t.replacingOccurrences(of: "+", with: Self.combiningAcute)  // "+"→U+0301
}
if let language { t = "[\(language)]" + t }
t = t.replacingOccurrences(of: " ", with: Self.space)
```

So `stress(_:)` receives text that is **already lowercased + NFKD-decomposed**
and returns it with stress marked (`+` after the vowel — converted to `U+0301`
in place downstream — or `U+0301` directly). Built-in implementations:

- `ManualRussianStress` (`MTLTextTokenizer.swift:39`) — identity; relies on
  caller-written marks. The `load` default.
- `DictionaryRussianStress` (`MTLTextTokenizer.swift:49`) — per-word exact-match
  table; manual-marked / OOV words pass through. The pre-neural fallback.
- `RuAccentStress` (`RuAccentStress.swift:31`) — the ruaccent adapter (below).

## `RuAccentStress` adapter

`RuAccentStress` (`Sources/ChatterboxCoreML/RuAccentStress.swift:31`) bridges
ruaccent's `RUAccent` to chatterbox's `RussianStressing`. It lives **inside** the
`ChatterboxCoreML` module so unqualified `RussianStressing` binds to chatterbox's
protocol. `@unchecked Sendable` is sound: `RUAccent` is a `final class` wrapping
load-once immutable CoreML models, and `stress` is a read-only inference path.

```swift
public struct RuAccentStress: RussianStressing, @unchecked Sendable {
    public let accentor: RUAccent
    public func stress(_ nfkdText: String) -> String {
        let composed = nfkdText.precomposedStringWithCanonicalMapping   // NFC for RUAccent
        guard let stressed = try? accentor.stress(composed, notation: .combiningAcute) else {
            return nfkdText                                              // fail-open
        }
        return stressed.decomposedStringWithCompatibilityMapping        // back to NFKD
    }
}
```

Two correctness traps the adapter handles (both verified against the real models):

1. **NFC round-trip (ё/й).** RUAccent's `normalize` accepts only *composed*
   Cyrillic and **strips bare combining marks** (`U+0308`/`U+0306`/`U+0301`).
   Feeding the tokenizer's raw NFKD corrupts `ё`/`й` words
   (`"мой"`→`"мои́"`, `"самолёт"`→`"самолё́т"`). So the adapter NFC-composes
   before stressing, then re-NFKD-decomposes the result to match what the
   grapheme tokenizer was trained on.
2. **Notation = `.combiningAcute`.** RUAccent already emits `U+0301` *after* the
   vowel, so the tokenizer's in-place `"+"`→`U+0301` replace is a harmless no-op.
   `.plusBeforeVowel` would land the acute on the preceding consonant
   (`"м+ой"`→`"м́ой"`) after that replace.

On any failure the adapter returns the input unchanged (fail-open), so TTS
degrades to caller-supplied marks rather than throwing.

### `makeOrFallback` — load or download, fail-open to Manual

`RuAccentStress.makeOrFallback(localDirectory:downloadHFHome:hfToken:)`
(`RuAccentStress.swift:54`) is the factory the app/CLI call. It returns
`RussianStressing` (not `RuAccentStress`) so the fallback type fits:

```swift
public static func makeOrFallback(
    localDirectory: URL? = nil,
    downloadHFHome: URL? = nil,
    hfToken: String? = nil
) async -> RussianStressing
```

- `localDirectory` non-nil → `RUAccent(modelDirectory:)`
  (`RUAccent.swift:125`), the converter's `_work` layout: `coreml/`
  (`M{1,2,3,4}_*.mlpackage`), `dictpack/` (four `.rapack`s), `nn/` (tokenizers).
- else → `RUAccent(hfHome:hfToken:)` (the `downloadingFrom:` convenience,
  `RUAccent.swift:146`) downloads `iliasaz/ruaccent-coreml` via swift-transformers
  `Hub`, with the **same** HF_HOME (and token, if one is set) chatterbox uses, so
  the two write sibling dirs under one cache.
- on any error (offline / download or load failure) → `ManualRussianStress()`,
  logged via `Log.load.error`. TTS still runs on caller-supplied marks.

## Wiring (app + CLI)

`ChatterboxCoreMLModel.load(from:russianStress:)`
(`ChatterboxCoreMLModel.swift:82`) passes the stresser into
`MTLTextTokenizer(modelFolder:stresser:)` for the multilingual variant only
(`ChatterboxCoreMLModel.swift:131`); the turbo branch never reads it.

**App** — `resolveRussianStress()` (`ContentView.swift:718`) returns
`ManualRussianStress()` for turbo, else downloads ruaccent with the app's HF
token + effective HF_HOME:

```swift
private func resolveRussianStress() async -> RussianStressing {
    guard variant == .multilingual else { return ManualRussianStress() }
    return await RuAccentStress.makeOrFallback(
        downloadHFHome: effectiveHFHome,
        hfToken: token.isEmpty ? nil : token)
}
// model = try await ChatterboxCoreMLModel.load(from:…, russianStress: await resolveRussianStress())
```

**CLI** — `--language ru` selects the multilingual repo; `--ruaccent-dir PATH`
points at a local `_work` dir (else it downloads). `ChatterboxCLI/main.swift:137`:

```swift
if args.language != nil {
    let stress = await RuAccentStress.makeOrFallback(
        localDirectory: args.ruaccentDir.map { URL(fileURLWithPath: $0) },
        downloadHFHome: nil,
        hfToken: args.hfToken)
    model = try await ChatterboxCoreMLModel.load(from: modelDir, russianStress: stress)
} else {
    model = try await ChatterboxCoreMLModel.load(from: modelDir)   // turbo: no stresser
}
```

## SwiftPM dependency

`ruaccent-coreml` is consumed as a SwiftPM dependency. Pin it to the merged
`main` branch (the conversions branch is merged into `main`; both packages pin
swift-transformers `from: 1.3.0`, so SPM dedups to one copy):

```swift
// Package.swift — dependencies
.package(url: "https://github.com/iliasaz/ruaccent-coreml", branch: "main"),

// ChatterboxCoreML target — dependencies
// product name `RUAccentCoreML`; package identity `ruaccent-coreml`
.product(name: "RUAccentCoreML", package: "ruaccent-coreml"),
```

## Example — Swift library

```swift
import ChatterboxCoreML

// 1. Build the Russian stress source. Download ruaccent from the Hub, or pass
//    localDirectory: a local ruaccent model dir to load offline.
let stress = await RuAccentStress.makeOrFallback(
    downloadHFHome: nil,                         // nil → HF_HOME / HF_HUB_CACHE env
    hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"])
// → RuAccentStress if loaded, else ManualRussianStress (fail-open).

// 2. Load the MULTILINGUAL model, passing the stresser into the tokenizer.
let model = try await ChatterboxCoreMLModel.load(
    from: multilingualModelDir,                  // bundle containing perceiver_query.npy
    russianStress: stress)

// 3. Generate. The tokenizer lowercases → NFKD → ruaccent stress → "[ru]" → encode.
var opts = GenerationOptions()
opts.language    = "ru"                          // enables the [ru] prefix + stress step
opts.minP        = 0.05                          // multilingual sampler preset
opts.topK        = 0
opts.cfgWeight   = MultilingualConstants.defaultCfgWeight
let buffer = try await model.generate("Привет, как дела?", options: opts)
```

To load a fixed offline dir instead of downloading:

```swift
let stress = await RuAccentStress.makeOrFallback(
    localDirectory: URL(fileURLWithPath: "/path/to/ruaccent/_work"))
```

## Example — CLI

```bash
# Download the multilingual model + ruaccent with the shared HF token:
swift run chatterbox-cli "Привет, как дела?" \
    --language ru --download --hf-token "$HF_TOKEN" --out ru.wav

# Use a local model dir + a local ruaccent _work dir (no network):
swift run chatterbox-cli "Самолёт летит над городом." \
    --language ru \
    --model-dir /path/to/multilingual-out \
    --ruaccent-dir /path/to/ruaccent/_work \
    --out ru.wav
```

If ruaccent is unavailable (offline / no token / load error), generation still
runs — stress falls back to caller-written marks (`+` after a vowel, or `U+0301`
directly), e.g. `"самол+ёт"`. Manual marks always win over the neural source.
