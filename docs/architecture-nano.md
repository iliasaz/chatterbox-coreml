# Architecture — Nano (GPT-2-small T3)

> Nano is **not** a new variant family — it's turbo with a smaller GPT-2
> backbone. This doc is a delta against
> [`architecture-turbo.md`](architecture-turbo.md) (read that first for the
> component map — tokenizer, chunker, `PrefillAssembler`, `T3LMRunner`,
> `SynthRunner`, orchestration — none of which changed) and
> [`T3LM-multifunction-contract.md`](T3LM-multifunction-contract.md)
> (wire-level prefill/decode I/O — identical shapes and semantics, just
> `H=768`/`layers=12` in place of `H=1024`/`layers=24`).

Upstream `chatterbox-nano`'s `tts_nano.py` is byte-identical to
`tts_turbo.py` except the class name, `REPO_ID`, `hp.llama_config_name =
"GPT2_small"` (vs `"GPT2_medium"`), and the checkpoint filename
(`t3_nano_v1.safetensors`, 870 MB, vs `t3_turbo_v1.safetensors`, 1915 MB).
HF sha256 hashes confirm the nano and turbo HF repos ship **bit-identical**
`s3gen_meanflow.safetensors`, `s3gen.safetensors`, `ve.safetensors`,
`conds.pt`, `vocab.json`, `merges.txt`, and tokenizer configs — only the T3
checkpoint differs. So the entire S3Gen synth (`S3Encoder`/`S3CFM`/
`S3Vocoder`), the voice-cloning encoders (CAMPPlus/MatchaMel/VEMel/VELSTM/
S3Tokenizer), `default-conds`, and the tokenizer are **reused bit-for-bit**
— zero synth work was needed to add nano.

## Geometry delta

| | turbo | nano |
|---|---|---|
| backbone | GPT-2-medium | GPT-2-small |
| layers | 24 | **12** |
| hidden | 1024 | **768** |
| heads | 16 | **12** |
| head_dim | 64 (1024/16) | 64 (768/12) — **same**, KV-layout math unchanged |
| KV feature width (`layers·heads·headDim`) | 24576 | **9216** |
| T3 params | — | **217.5M** |
| checkpoint | `t3_turbo_v1.safetensors`, 1915 MB | `t3_nano_v1.safetensors`, 870 MB |
| `T3LM.mlpackage` (shipped) | 328 MB, **8-bit palettized** | 120 MB, **8-bit palettized** |
| HF repo | `iliasaz/chatterbox-turbo-coreml` | `iliasaz/chatterbox-nano-coreml` |
| speech vocab / start / stop / window `W` / prefill text budget | 6563 / 6561 / 6562 / 512 / ~135 tok | unchanged |

`head_dim` being identical either way (64) is the load-bearing fact — it's
why the KV-cache layout, the mask builders, and the whole `T3LMRunner`
contract carry over with zero shape-logic changes; only the table *widths*
differ.

## What's new in the runtime

- **Swift**: `ModelRepository.Variant.nano`; `T3LMRunner`'s `hidden` is now
  **derived from the loaded graph's `inputs_embeds` width**, not the
  hardcoded `Constants.gpt2Hidden` — see the table-width-guard trap below.
  `ChatterboxCoreMLModel.load` variant-detects 3-way: `perceiver_query.npy`
  present → multilingual; else `speech_emb` table width 768 → nano, else
  turbo. `NanoConstants` (`Constants.swift`) holds `hidden=768, heads=12,
  headDim=64, layers=12`.
- **App/CLI**: three-way variant picker/flag (`--variant
  turbo|nano|multilingual`); the GPT-2-sampler top-k control gates on
  `variant != .multilingual` (was `variant == .turbo`), since nano uses the
  identical sampler.

## Traps

- **The `t3_nano_v1.yaml` / `t3_turbo_v1.yaml` checkpoint configs on HF are
  stale training configs** — they claim `Llama_520M`, 30 layers,
  `speech_cond_prompt_len` 250, none of which is true for the shipped T3.
  `hp` is built in code (`load_pytorch_model_nano()`); **do not read the
  yaml for hyperparameters.**
- **The host `.npy` embedding tables are hidden-width-specific** (768 nano
  / 1024 turbo). Mixing them across variants (e.g. copying `out-nano/*.npy`
  into `out/`) produces **garbage audio with no crash** — the graph runs
  fine, it just reads the wrong rows. `T3LMRunner`'s load-time guard
  (`validateTableWidths`) catches a mismatch between the loaded T3LM graph
  and the host tables and throws instead of silently mis-decoding; never
  hand-copy `.npy` tables between `out/` and `out-nano/`.
- **Exact top-5 logit order is the wrong ANE parity gate — even the shipping
  turbo model fails it.** On-device
  fp16 rounding routinely swaps two logits tied to within ~0.02, on both
  nano and turbo. Judge ANE correctness by **cos_sim ≥ 0.99** and by
  whether the stop-token logit is anywhere near top-1 (the actual
  fused-SDPA-mask-drop failure mode), not by top-5 order.
  (The **shipped 8-bit nano build** actually gets an *exact* top-5 match on
  `cpuAndNeuralEngine`/`all` — cleaner than the fp16 build's benign rank
  3/4 swap. The general point stands regardless: judge by cos_sim and the
  stop logit, not top-5 order, since the shipping turbo model itself
  "fails" the strict check.)

## Verified (2026-07-11, initial port; 2026-07-12, 8-bit palettization)

Converter `--validate` on the initial fp16 port (2026-07-11): seed cos_sim
0.999796, decode steps cos_sim ≥ 0.9995. Full Swift suite against a nano
model dir: 88/88 (incl. e2e audio, multi-chunk streaming, overlap-vs-serial
parity, voice-cloning parity).

**What ships is the 2026-07-12 8-bit palettized follow-up**, not the fp16
build above: `T3LM.mlpackage` is **120 MB**, 8-bit k-means palettized
(`per_grouped_channel`, group size 16; `wpe` and `speech_head` stay fp16 —
6563 speech-vocab channels are prime, so no group size divides them).
Converter `--validate`: seed cos_sim 0.999652, decode steps cos_sim ≥
0.9995. Full Swift suite: 91/26 suites. Mac CLI (M5, Release, same
sentence): nano ~6.4× realtime vs turbo ~3.9× (wall-time is roughly
unchanged from the fp16 build — a different run of a nondeterministic
sampler, not a measured speedup). On-device ANE probe (iPhone 17 Pro Max,
iOS 26.5): 8-bit nano prefill cos_sim 0.999625 at **22 ms**/predict on
`cpuAndNeuralEngine`/`all`, with an **exact top-5 match**, vs a turbo
control run on the same phone/probe at cos_sim 0.9998 / 64 ms — nano's
prefill is ~2.9× faster on the ANE, and smaller and at least as accurate
than the fp16 build it replaces.
