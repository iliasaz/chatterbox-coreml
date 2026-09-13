# Bundled reference voices

Every voice here is built from a demo prompt **Resemble AI publishes themselves**,
for their own Chatterbox demo applications. They are anonymous speakers identified
only by language and gender — upstream's naming, kept as-is. No persona names are
invented, and no identifiable person is cloned.

Only the derived `*-conds.safetensors` are committed. The source audio is not
redistributed here; the URLs below are the originals, and every one was fetched with
HTTP 200 on 2026-09-05.

## How these were made

With the shipped Swift path, so anyone with this repository can rebuild them:

```sh
swift build -c release
.build/release/chatterbox-cli --clone-voice <prompt>.flac \
    --voice ChatterboxApp/ChatterboxApp/Voices/<stem>-conds.safetensors \
    --clone-only --model-dir /path/to/out
```

`VoiceCloner` runs the shared voice encoders (MatchaMel, VEMel/VELSTM, S3Tokenizer,
CAMPPlus), none of which carry T3 weights, so one conds file works under any variant
that will accept it — see the caveat below for what "accept" means in practice.

## Turbo / nano — `gradio_tts_turbo_app.py`

| voice | language | source | prompt | conds |
|---|---|---|---|---|
| `female_random_podcast-conds` | English | `prompts/female_random_podcast.wav` | 29.85 s | 164,767 B |

## Multilingual — `multilingual_app.py` (`LANGUAGE_CONFIG`)

One prompt per supported language, all 23. "Output" is the audio length produced from
upstream's own example sentence for that language, `--seed 7`, on `out-mtl-full`.

| voice | language | code | source (under `mtl_prompts/`) | prompt | conds | output |
|---|---|---|---|---|---|---|
| `ru_m-conds` | Russian | `ru` | `mtl_prompts/ru_m.flac` | 6.26 s | 103,693 B | 6.18s |
| `ar_f-conds` | Arabic | `ar` | `mtl_prompts/ar_f/ar_prompts2.flac` | 2.60 s | 44,386 B | 6.08s |
| `da_m1-conds` | Danish | `da` | `mtl_prompts/da_m1.flac` | 3.27 s | 55,080 B | 6.50s |
| `de_f1-conds` | German | `de` | `mtl_prompts/de_f1.flac` | 8.13 s | 133,825 B | 7.04s |
| `el_m-conds` | Greek | `el` | `mtl_prompts/el_m.flac` | 4.06 s | 68,048 B | 8.28s |
| `en_f1-conds` | English | `en` | `mtl_prompts/en_f1.flac` | 3.04 s | 51,194 B | 4.42s |
| `es_f1-conds` | Spanish | `es` | `mtl_prompts/es_f1.flac` | 12.01 s | 164,485 B | 6.64s |
| `fi_m-conds` | Finnish | `fi` | `mtl_prompts/fi_m.flac` | 6.93 s | 114,399 B | 7.16s |
| `fr_f1-conds` | French | `fr` | `mtl_prompts/fr_f1.flac` | 6.60 s | 109,211 B | 5.96s |
| `he_m1-conds` | Hebrew | `he` | `mtl_prompts/he_m1.flac` | 5.22 s | 86,844 B | 5.70s |
| `hi_f1-conds` | Hindi | `hi` | `mtl_prompts/hi_f1.flac` | 5.25 s | 87,162 B | 6.28s |
| `it_m1-conds` | Italian | `it` | `mtl_prompts/it_m1.flac` | 5.96 s | 98,496 B | 7.06s |
| `ja-conds` | Japanese | `ja` | `mtl_prompts/ja/ja_prompts1.flac` | 3.56 s | 59,940 B | 5.92s |
| `ko_f-conds` | Korean | `ko` | `mtl_prompts/ko_f.flac` | 5.10 s | 84,886 B | 6.58s |
| `ms_f-conds` | Malay | `ms` | `mtl_prompts/ms_f.flac` | 10.98 s | 164,371 B | 6.84s |
| `nl_m-conds` | Dutch | `nl` | `mtl_prompts/nl_m.flac` | 1.44 s | 25,265 B | 4.78s |
| `no_f1-conds` | Norwegian | `no` | `mtl_prompts/no_f1.flac` | 4.89 s | 81,328 B | 5.08s |
| `pl_m-conds` | Polish | `pl` | `mtl_prompts/pl_m.flac` | 2.70 s | 46,000 B | 6.26s |
| `pt_m1-conds` | Portuguese | `pt` | `mtl_prompts/pt_m1.flac` | 15.82 s | 164,787 B | 7.64s |
| `sv_f-conds` | Swedish | `sv` | `mtl_prompts/sv_f.flac` | 5.34 s | 88,778 B | 6.34s |
| `sw_m-conds` | Swahili | `sw` | `mtl_prompts/sw_m.flac` | 8.84 s | 145,499 B | 6.84s |
| `tr_m-conds` | Turkish | `tr` | `mtl_prompts/tr_m.flac` | 4.45 s | 74,198 B | 6.64s |
| `zh_f2-conds` | Chinese | `zh` | `mtl_prompts/zh_f2.flac` | 3.04 s | 51,518 B | 4.24s |

## The one caveat, measured

**Ten of the 23 multilingual prompts are under 5 s, and those degenerate on turbo.**
Not subtly: asked for one sentence, `en_f1` returned 0.18 s of audio and `nl_m`
returned 26.74 s of babble. The same conds are correct on the multilingual model —
which is where these prompts come from, and where all 23 were verified (table above).

Ablated rather than assumed. Duration is the variable, not mel/token parity and not a
language mismatch: `en_f1` is English, on the English model, and still fails at
3.04 s, while `ru_m` (Russian, 6.26 s) and `pt_m1` (15.82 s) are both fine on turbo.
The plausible mechanism is that the multilingual T3 resamples its speech-conditioning
prompt through a Perceiver onto a fixed query set, so a short prompt is handled
gracefully, whereas turbo consumes the prompt tokens directly against a 375-token
budget that a 3 s clip barely fills.

This is why `VoiceDemoSet` scopes the picker to the variant's own set rather than
showing every voice under every model. Conds being model-agnostic is a statement
about whether they *load*, not about whether the result is usable.

## Licence

Upstream chatterbox is MIT (© Resemble AI). The audio assets carry no separate stated
terms; they are the authors' own demo prompts, served from their own bucket and linked
from their own MIT-licensed demo apps for exactly this purpose. See `NOTICE` at the
repository root.
