import SwiftUI
import AVFoundation
import ChatterboxCoreML

/// Where the model directory comes from. macOS lets the user choose; iOS always
/// uses the single default location (the app's `Documents` HF_HOME).
/// A ready-made demo line for the Generate text field. The English ones show off
/// intonation/emotion using the paralinguistic tags ([happy] [laugh] …) of the
/// GPT-2 models (turbo + nano) — the **only** models with native tags. The Russian
/// set (multilingual) is tag-free: the multilingual model has no paralinguistic
/// tags, so expressiveness comes from `exaggeration` / `cfg_weight` /
/// `temperature` instead.
struct DemoText: Identifiable, Hashable {
    let id: String
    let text: String
    var name: String { id }

    /// English demos (turbo + nano — same tokenizer, same tags).
    static let english: [DemoText] = [
        DemoText(id: "Storybook",
                 text: "At the edge of a silver forest, where the moss glowed softly at night and the streams sounded like whispered songs, a small fox named Juniper woke to a strange sound beneath her window."),
        DemoText(id: "Excited",
                 text: "[happy] You did it! You actually did it! [laugh] I knew you had it in you all along — this calls for a celebration!"),
        DemoText(id: "Suspense",
                 text: "[dramatic] The lights flickered once, then went out. [whispering] Don't move. [fear] I think something just shifted in the dark behind us."),
        DemoText(id: "Sarcastic",
                 text: "[sarcastic] Oh, brilliant. Another Monday, another genius idea from management. [sigh] What could possibly go wrong this time?"),
        DemoText(id: "Reunion",
                 text: "[surprised] Wait... is that really you? [gasp] After all these years! [crying] I honestly never thought I'd see you again."),
    ]

    /// Russian demos (multilingual). First is the ё/й-heavy stress test.
    static let russian: [DemoText] = [
        DemoText(id: "Скороговорка",
                 text: "Йога успокаивает. Покойный отдых. Тёплый ёжик и его лисёнок шли домой. Мой жёлтый лист. Тёплый ёжик нёс жёлтый лист. Это был мой русский лисёнок, чёрный и весёлый."),
        DemoText(id: "Сказка",
                 text: "Жил-был маленький лисёнок. Каждое утро он выходил к реке и слушал, как поют птицы. Однажды он встретил милого ёжика, и с того дня они стали лучшими друзьями."),
        DemoText(id: "Радость",
                 text: "Ты сделал это! Я и не сомневался, что у тебя получится! Это надо непременно отпраздновать!"),
        DemoText(id: "Спокойствие",
                 text: "Тихий вечер опускался на старый город. Где-то вдалеке негромко звонил колокол. Всё вокруг было спокойно, тепло и удивительно тихо."),
        DemoText(id: "Диалог",
                 text: "Привет! Как у тебя дела? Мы так давно не виделись. Чем ты занимаешься этим летом — всё так же рисуешь?"),
    ]

    /// English demos with paralinguistic tags stripped — for the multilingual
    /// model, which has no tag support and would mis-render `[happy]`/`[laugh]`/…
    /// (it reads them aloud rather than acting them). Same ids/order as `english`.
    static let englishTagFree: [DemoText] = english.map {
        DemoText(id: $0.id, text: stripTags($0.text))
    }

    /// Removes `[...]` paralinguistic tags and tidies the whitespace they leave.
    static func stripTags(_ s: String) -> String {
        s.replacingOccurrences(of: #"\[[^\]]*\]"#, with: "", options: .regularExpression)
            .replacingOccurrences(of: #"\s{2,}"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }
}

enum ModelSource: String, CaseIterable, Identifiable {
    /// A user-chosen (or platform-default) `HF_HOME` root; the model is
    /// discovered/downloaded under `<HF_HOME>/hub/…`.
    case hfHome = "HF_HOME"
    /// The model directory itself (e.g. the converter's `out/`), loaded directly.
    case localFolder = "Local folder"
    var id: String { rawValue }
}

/// Cross-platform testbed UI.
///
/// - **iOS**: one fixed location — the app's `Documents` HF_HOME. Download (with
///   an optional HF token) → Load → Generate & Play.
/// - **macOS**: a Source picker (HF_HOME or a local model folder), with the last
///   choice + picked folders remembered across launches.
struct ContentView: View {
    /// Which model to download / discover / load. The loaded pipeline's actual
    /// variant is auto-detected at load; this selects the HF repo + glob set.
    @State private var variant: ModelRepository.Variant = .turbo
    /// Language code for the multilingual model (tokenizer + sampler). Ignored by
    /// the English models (turbo, nano).
    @State private var language = "ru"
    @State private var source: ModelSource = .hfHome
    /// A user-chosen `HF_HOME` root (macOS); overrides the platform default.
    @State private var hfHome: URL?
    /// A directly-chosen model directory (macOS, `source == .localFolder`).
    @State private var localModelDir: URL?
    @State private var modelPath = ""
    /// Optional Hugging Face access token — only needed for a private fork of the
    /// model repos. Persisted in the Keychain.
    @State private var hfToken = ""
    @State private var text = DemoText.english[0].text
    /// Selected demo preset (id) for the Text dropdown.
    @State private var selectedDemo = DemoText.english[0].id
    @State private var status = ""
    /// Persistent model-load result (kept visible across generations).
    @State private var loadInfo = ""
    /// What happened to the model **update** at launch — an update offered but not applied
    /// (too big to spend unasked), or one that was attempted and failed.
    ///
    /// Deliberately NOT `status`: autorun's very next step is `autoLoad()`, which overwrites
    /// `status` with "Ready to generate." — so a message written there would flash and vanish,
    /// and the user would never learn that the update didn't apply (or that one is waiting).
    /// This survives the load, and stays until an update actually lands.
    @State private var updateNotice = ""
    /// The variant actually loaded (model's own report), shown in the UI; nil = none.
    /// Invariant: while non-nil it equals `variant` — `syncVariantToLoaded` snaps the
    /// picker to it, and a user switch clears it. `onVariantChanged` relies on this to
    /// tell a programmatic snap from a real user switch.
    @State private var loadedVariant: ModelRepository.Variant?

    // MARK: Generation parameters (persisted). Which apply depends on the model —
    // see Chatterbox-TTS-Server docs/parameters.md: temperature/top_p/
    // repetition_penalty/seed are shared; top_k is GPT-2-only (turbo + nano);
    // min_p/exaggeration/cfg_weight are multilingual-only (the GPT-2 models ignore
    // them); language is multilingual.
    @AppStorage("param.temperature") private var temperature = 0.8
    @AppStorage("param.topP") private var topP = 0.95
    @AppStorage("param.repetitionPenalty") private var repetitionPenalty = 1.2
    @AppStorage("param.seed") private var seed = 0                  // 0 = random
    @AppStorage("param.topK") private var topK = 1000              // turbo + nano
    @AppStorage("param.minP") private var minP = 0.05             // multilingual only
    @AppStorage("param.exaggeration") private var exaggeration = 0.5  // multilingual only
    @AppStorage("param.cfgWeight") private var cfgWeight = 0.5    // multilingual only
    // Playback pacing (both models): seconds of silence after each sentence / comma.
    @AppStorage("param.sentencePause") private var sentencePause = 0.25
    @AppStorage("param.commaPause") private var commaPause = 0.25
    /// When on, delete the cached model before downloading (clean full re-fetch).
    @State private var forceRedownload = false
    /// When on, load the WHOLE pipeline (T3LM prefill+decode plus the three synth
    /// models) on the Neural Engine instead of the mixed ANE/GPU default — the
    /// background-capable path. Applies at (re)load; passed to
    /// `ChatterboxCoreMLModel.load` as `PipelineComputeUnits.neuralEngine`.
    /// Default **on**: the ANE path is background-capable, and the app's product need
    /// is speaking while backgrounded/locked (background-mode plan §4, rollout a). Turn
    /// off only for a foreground A/B against the slightly-faster GPU synth path.
    @AppStorage("synth.forceANE") private var forceSynthANE = true
    @State private var model: ChatterboxCoreMLModel?
    /// The reference voice used for synthesis (`.default` = the model's built-in).
    @State private var selectedVoice: Voice = .default
    /// Voices shown in the picker: default + the selected variant's reference set +
    /// user clones. Seeded for the `variant` default and refreshed by
    /// `refreshVoices()` on restore, variant switch, and voice create / delete.
    @State private var voices: [Voice] = Voice.allVoices(for: .turbo)
    /// Presents the record/import "Create Voice" sheet.
    @State private var creatingVoice = false
    @State private var busy = false
    @State private var importing = false
    /// Auto-run guard: load + generate once per app launch if the model is already
    /// on disk. Set after the first `tryAutoRun` to avoid re-firing on every state
    /// change. `CHATTERBOX_NO_AUTORUN=1` disables (debug / manual testing).
    @State private var autoRunDone = false
    // `@State`, NOT `let`: `WindowGroup { ContentView() }` re-runs on every
    // re-evaluation, and a `let` with an inline default allocates a FRESH instance each
    // time. `@State` storage is owned by SwiftUI and survives those re-creations, so
    // there is exactly one of each for the process's lifetime.
    //
    // This is load-bearing, not tidiness. Measured on device 2026-08-11: with `let`,
    // `lifecycle.install()` (called once from `.onAppear`) registered its observers on
    // one instance while the generation loop read a *different*, never-installed one —
    // so every per-chunk line logged `phase=fg` even after `didEnterBackground` had
    // marked `phase=bg`. `Player` had the milder form of the same bug: each instance
    // self-registers its interruption observer in `init`, so the captured one still
    // worked, but every re-creation leaked another `AVAudioEngine` + observer.
    @State private var player = Player()
    /// Observes app/lock lifecycle and stamps `.notice` phase marks in the log; also
    /// holds a background-task assertion (belt-and-suspenders next to the audio
    /// keep-alive).
    @State private var lifecycle = AppLifecycle()

    // UserDefaults keys for remembering the last-used model location (macOS).
    private static let sourceKey = "model.source"
    private static let hfHomeKey = "model.hfHomeBookmark"
    private static let localDirKey = "model.localDirBookmark"
    private static let voiceKey = "voice.selected"
    private static let variantKey = "model.variant"
    private static let languageKey = "model.language"

    /// Restores the HF token, plus (macOS only) the last Source choice and picked
    /// folders. iOS always uses the single default location, so nothing to restore.
    private func restoreState() {
        // UI-test / headless runs skip the Keychain (its prompt blocks automation).
        hfToken = ProcessInfo.processInfo.environment["CHATTERBOX_UI_TEST"] != nil ? "" : TokenStore.load()
        if let raw = UserDefaults.standard.string(forKey: Self.variantKey),
           let saved = ModelRepository.Variant(rawValue: raw) {
            variant = saved
        }
        if let lang = UserDefaults.standard.string(forKey: Self.languageKey), !lang.isEmpty {
            language = lang
        }
        selectedVoice = Voice.named(UserDefaults.standard.string(forKey: Self.voiceKey))
        refreshVoices()                       // filter to the restored variant
        refreshDemos()                        // match demo text to the restored variant/language (tag-free for MTL)
        #if os(macOS)
        hfHome = FolderBookmark.resolve(forKey: Self.hfHomeKey)
        localModelDir = FolderBookmark.resolve(forKey: Self.localDirKey)
        // Set source last so the onChange handler resolves with folders in place.
        if let raw = UserDefaults.standard.string(forKey: Self.sourceKey),
           let saved = ModelSource(rawValue: raw) {
            source = saved
        }
        #endif
    }

    /// Demo presets for the selected model + **language**. The multilingual model
    /// gets tag-free text (Russian on `ru`, otherwise the tag-stripped English set —
    /// it can't render the `[happy]`/`[laugh]` tags); the GPT-2 models (turbo, nano)
    /// get the tagged English demos. Non-Russian multilingual languages fall back to
    /// English (the only Latin set we ship).
    private var demos: [DemoText] {
        if variant == .multilingual {
            return language.hasPrefix("ru") ? DemoText.russian : DemoText.englishTagFree
        }
        return DemoText.english
    }

    /// Snaps the demo dropdown + text to the current set after a language/variant
    /// change. Keeps the same preset when its id still exists (all variants share
    /// ids), but re-reads its text so tags are added/stripped for the new model;
    /// falls back to the first preset when the id is gone (English↔Russian).
    private func refreshDemos() {
        let d = demos
        if let sel = d.first(where: { $0.id == selectedDemo }) {
            text = sel.text
        } else {
            selectedDemo = d[0].id
            text = d[0].text
        }
    }

    /// Rebuilds the voice picker for the selected variant; keeps the selection if
    /// still valid (e.g. survives a user-voice delete or a variant switch), else
    /// falls back to default.
    private func refreshVoices() {
        voices = Voice.allVoices(for: variant)
        if !voices.contains(where: { $0.id == selectedVoice.id }) {
            selectedVoice = .default
            UserDefaults.standard.set(selectedVoice.id, forKey: Self.voiceKey)
        }
    }

    /// Handles a variant switch: persist, drop the loaded model, swap the voice list
    /// + demo set to the new language, and re-resolve the model dir.
    private func onVariantChanged(_ newValue: ModelRepository.Variant) {
        // Already showing the loaded model's own variant ⇒ this is `syncVariantToLoaded`'s
        // snap arriving on the next view update, not a user switch: don't wipe the model.
        guard newValue != loadedVariant else { return }
        UserDefaults.standard.set(newValue.rawValue, forKey: Self.variantKey)
        model = nil
        loadedVariant = nil
        loadInfo = ""
        updateNotice = ""  // it named the *other* variant's repo and size
        refreshVoices()
        refreshDemos()                 // language-aware demo set for the new variant
        resolveModel()
    }

    /// After a load, snap the Model selector + dependent UI (voices, demos,
    /// language) to the variant that **actually** loaded — a local folder's
    /// contents win over the picker, so this keeps the selector from contradicting
    /// the "Loaded:" indicator. No-op when they already match (the HF flow).
    private func syncVariantToLoaded() {
        guard let lv = loadedVariant, lv != variant else { return }
        variant = lv
        UserDefaults.standard.set(lv.rawValue, forKey: Self.variantKey)
        refreshVoices()
        refreshDemos()             // strip/add tags to match the variant that actually loaded
    }

    /// On iOS the app sandbox `Documents` dir is the default `HF_HOME` root: the
    /// snapshot lands under `<Documents>/hub/…`, which download and discovery
    /// agree on. macOS has no default — pick a folder or use the environment.
    private var defaultHFHome: URL? {
        #if os(iOS)
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        #else
        nil
        #endif
    }

    private var effectiveHFHome: URL? { hfHome ?? defaultHFHome }

    private var sourceLabel: String { source == .hfHome ? "HF_HOME folder" : "Model folder" }

    /// Segmented-picker label. `displayName` ("Multilingual (Russian)") is prose for
    /// the "Loaded:" row — three of those truncate to ellipses in a segmented control
    /// on iPhone, so the picker uses the bare model name.
    private func segmentLabel(_ v: ModelRepository.Variant) -> String {
        switch v {
        case .turbo: return "Turbo"
        case .nano: return "Nano"
        case .multilingual: return "Multilingual"
        }
    }

    private var navigationTitle: String {
        switch variant {
        case .turbo: return "Chatterbox Turbo"
        case .nano: return "Chatterbox Nano"
        case .multilingual: return "Chatterbox Multilingual"
        }
    }

    /// One-line summary of which Parameters apply to the selected model.
    private var parametersFooter: String {
        switch variant {
        case .turbo:
            return "Turbo: top_k applies; emotion comes from paralinguistic tags in the text (e.g. [happy], [laugh])."
        case .nano:
            return "Nano: same GPT-2 sampler as Turbo (top_k applies) and the same paralinguistic tags — a smaller, faster English model."
        case .multilingual:
            return "Multilingual: expressiveness comes from exaggeration / cfg_weight / temperature — no paralinguistic tags."
        }
    }

    private var sourceDisplay: String {
        switch source {
        case .localFolder:
            return localModelDir?.path ?? "Choose the model folder (e.g. the converter's out/)."
        case .hfHome:
            if let effectiveHFHome { return effectiveHFHome.path }
            if let envBase = ModelRepository.environmentDownloadBase() {
                return "\(envBase.path) (from environment)"
            }
            return "Not set — choose a folder."
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("Model", selection: $variant) {
                        ForEach(ModelRepository.Variant.allCases) { Text(segmentLabel($0)).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(busy)
                    .onChange(of: variant) { _, newValue in onVariantChanged(newValue) }

                    if variant == .multilingual {
                        Picker("Language", selection: $language) {
                            ForEach(SupportedLanguage.all) { Text("\($0.name) (\($0.code))").tag($0.code) }
                        }
                        .pickerStyle(.menu)
                        .disabled(busy)
                        .onChange(of: language) { _, newValue in
                            UserDefaults.standard.set(newValue, forKey: Self.languageKey)
                            refreshDemos()      // Russian ⇄ English demo texts follow the language
                        }
                    }

                    #if os(macOS)
                    Picker("Source", selection: $source) {
                        ForEach(ModelSource.allCases) { Text($0.rawValue).tag($0) }
                    }
                    .pickerStyle(.segmented)
                    .disabled(busy)
                    .onChange(of: source) { _, newValue in
                        UserDefaults.standard.set(newValue.rawValue, forKey: Self.sourceKey)
                        model = nil
                        loadInfo = ""
                        updateNotice = ""  // it named the dir we just switched away from
                        resolveModel()
                    }
                    #endif

                    #if os(macOS)
                    // macOS: the user chooses locations, so show the root (gray)
                    // and the resolved model dir that Load will use (green).
                    Text(sourceDisplay)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .lineLimit(2)
                        .truncationMode(.middle)

                    Text(modelPath.isEmpty ? "Model: not found" : "Model: \(modelPath)")
                        .font(.caption2)
                        .foregroundStyle(modelPath.isEmpty ? Color.secondary : Color.green)
                        .textSelection(.enabled)
                        .lineLimit(2)
                        .truncationMode(.middle)
                    #else
                    // iOS: one fixed location — just show readiness.
                    Text(modelPath.isEmpty ? "No model yet — tap Download." : "Model ready — tap Load.")
                        .font(.callout)
                        .foregroundStyle(modelPath.isEmpty ? Color.secondary : Color.green)
                    #endif

                    if source == .hfHome {
                        SecureField("Hugging Face token (optional)", text: $hfToken)
                            #if os(iOS)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            #endif
                            .onChange(of: hfToken) { _, newValue in TokenStore.save(newValue) }
                        Text("Stored in Keychain. Leave empty for a public repo, or use the HF_TOKEN env var / `hf auth login`.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Toggle("Force re-download", isOn: $forceRedownload)
                            .font(.caption)
                            .disabled(busy)
                    }

                    if let loadedVariant {
                        Label("Loaded: \(loadedVariant.displayName)", systemImage: "checkmark.seal.fill")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }

                    if !loadInfo.isEmpty {
                        Text(loadInfo)
                            .font(.caption2)
                            .foregroundStyle(.green)
                    }

                    // An update we chose not to apply (over the auto-apply size limit), or one
                    // that failed. Sits next to the Download button that applies it, and is not
                    // wiped by the load that follows.
                    if !updateNotice.isEmpty {
                        Label(updateNotice, systemImage: "arrow.down.circle")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Toggle("Run on Neural Engine", isOn: $forceSynthANE)
                            .font(.caption)
                            .disabled(busy)
                            // Placement is fixed at load, so re-apply by reloading the
                            // current model (no-op when nothing is loaded — takes effect
                            // on the next Load).
                            .onChange(of: forceSynthANE) { _, _ in
                                if model != nil { loadModel() }
                            }
                        Text("On by default: runs the whole pipeline (T3LM plus the synth encoder, flow, vocoder) on the Neural Engine — the background-capable path (iPhone has no background GPU, so a backgrounded/locked GPU submission fails). Turn off only to A/B the slightly-faster GPU synth path in the foreground. T3LM always runs on the Neural Engine regardless. Applies when the model is (re)loaded; toggling reloads the current model. The first load after a placement change pays a one-time ANE recompile of T3LM (cached for later launches).")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    // `.borderless` makes each button its own tap target — in a
                    // Form row, default-styled buttons share one target, so a tap
                    // on Load would also fire Choose/Download.
                    HStack {
                        #if os(macOS)
                        Button("Choose \(sourceLabel)…") { importing = true }.disabled(busy)
                        #endif
                        Spacer()
                        if source == .hfHome {
                            Button("Download") { downloadModel() }.disabled(busy)
                        }
                        Button("Load") { loadModel() }.disabled(busy || modelPath.isEmpty)
                    }
                    .buttonStyle(.borderless)
                }

                Section("Voice") {
                    Picker("Voice", selection: $selectedVoice) {
                        // `voices` is already scoped to the selected variant's demo
                        // set (see `VoiceDemoSet`), so the sections are driven by the
                        // data rather than by hardcoded language headings — which
                        // would otherwise render empty for any variant that has none.
                        Text(Voice.default.displayName).tag(Voice.default)
                        let refs = voices.filter { !$0.isUserVoice && $0.id != Voice.default.id }
                        if !refs.isEmpty {
                            Section("Reference voices") {
                                ForEach(refs) { Text($0.displayName).tag($0) }
                            }
                        }
                        if voices.contains(where: { $0.isUserVoice }) {
                            Section("My Voices") {
                                ForEach(voices.filter { $0.isUserVoice }) { Text($0.displayName).tag($0) }
                            }
                        }
                    }
                    .pickerStyle(.menu)
                    .disabled(busy)
                    .onChange(of: selectedVoice) { _, newValue in
                        UserDefaults.standard.set(newValue.id, forKey: Self.voiceKey)
                    }

                    HStack {
                        Button {
                            creatingVoice = true
                        } label: {
                            Label("Create Voice…", systemImage: "mic.badge.plus")
                        }
                        .disabled(busy || modelPath.isEmpty)
                        Spacer()
                        if selectedVoice.isUserVoice {
                            Button(role: .destructive) {
                                deleteSelectedVoice()
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                            .disabled(busy)
                        }
                    }
                    .buttonStyle(.borderless)
                    if modelPath.isEmpty {
                        Text("Load a model first to create a voice.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                }

                Section("Parameters") {
                    // Shared by both models.
                    sliderRow("Temperature", value: $temperature, range: 0...1.5, step: 0.05,
                              help: "Expressiveness ↔ stability. Higher is livelier but riskier (slurring/glitches); lower is flatter and more consistent.")
                    sliderRow("Top-p", value: $topP, range: 0...1, step: 0.01,
                              help: "Nucleus sampling. Lower is tighter and safer; near 1.0 adds variation and expressive detail.")
                    sliderRow("Repetition penalty", value: $repetitionPenalty, range: 1...2, step: 0.05,
                              help: "Anti-stutter / anti-loop. ~1.2 is the sweet spot; above ~1.5 it backfires (clipped, rushed, drifting pitch).")
                    // Multilingual-only (turbo ignores these).
                    if variant == .multilingual {
                        sliderRow("Min-p", value: $minP, range: 0...0.5, step: 0.01,
                                  help: "Drops tokens below this fraction of the top token's probability. Higher is tighter/cleaner.")
                        sliderRow("Exaggeration", value: $exaggeration, range: 0...2, step: 0.05,
                                  help: "Expressiveness / animation of the delivery. Higher is more dramatic; lower is more neutral.")
                        sliderRow("CFG weight", value: $cfgWeight, range: 0...1, step: 0.05,
                                  help: "How tightly the voice follows the reference style + pacing. Higher adheres more; lower varies more.")
                    }
                    // GPT-2 sampler (turbo + nano); the multilingual preset has no top-k.
                    if variant != .multilingual {
                        VStack(alignment: .leading, spacing: 2) {
                            Stepper(value: $topK, in: 0...2000, step: 50) {
                                HStack {
                                    Text("Top-k"); Spacer()
                                    Text("\(topK)").foregroundStyle(.secondary).monospacedDigit()
                                }
                                .font(.caption)
                            }
                            .disabled(busy)
                            Text("Keeps only the k most-likely acoustic tokens. Lower is cleaner/safer; very low (~5–10) sounds robotic.")
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    // Playback pacing (both models).
                    sliderRow("Sentence pause", value: $sentencePause, range: 0...1, step: 0.05,
                              help: "Seconds of silence added after each sentence (.!?). 0 = off. Each sentence is synthesized as its own unit so the pause lands at every boundary.")
                    sliderRow("Comma pause", value: $commaPause, range: 0...1, step: 0.05,
                              help: "Seconds of silence added after each comma. 0 = off (commas stay inside the sentence). Higher spaces out clauses more.")
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text("Seed").font(.caption)
                            Spacer()
                            TextField("0 = random", value: $seed, format: .number)
                                .frame(width: 90)
                                .multilineTextAlignment(.trailing)
                                #if os(iOS)
                                .keyboardType(.numberPad)
                                #endif
                                .disabled(busy)
                        }
                        Text("0 = random each run. Any non-zero value locks the take (reproducible) — useful for A/B-ing parameters.")
                            .font(.caption2).foregroundStyle(.secondary)
                    }
                    Button("Reset to defaults") { resetParameters() }
                        .buttonStyle(.borderless)
                        .font(.caption)
                        .disabled(busy)
                    Text(parametersFooter)
                        .font(.caption2).foregroundStyle(.secondary)
                }

                Section("Text") {
                    Picker("Demo", selection: $selectedDemo) {
                        ForEach(demos) { Text($0.name).tag($0.id) }
                    }
                    .pickerStyle(.menu)
                    .disabled(busy)
                    .onChange(of: selectedDemo) { _, newValue in
                        if let d = demos.first(where: { $0.id == newValue }) { text = d.text }
                    }
                    TextField("Text to speak", text: $text, axis: .vertical)
                        .lineLimit(2...6)
                    Button {
                        generate()
                    } label: {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Generate & Play")
                            if busy { Spacer(); ProgressView() }
                        }
                    }
                    .buttonStyle(.borderless)
                    .disabled(busy || model == nil)
                }

                Section {
                    HStack(spacing: 8) {
                        if busy { ProgressView() }
                        Text(status).font(.callout).foregroundStyle(.secondary)
                    }
                }
            }
            // macOS `Form` defaults to the two-column `.columns` style, which blows
            // the rows (HStacks-with-Spacer, sliders, segmented pickers, long help
            // text) apart to the window edges — labels clipped off the left, values
            // off the right, blank middle. This UI is authored for the iOS inset-
            // grouped idiom, so force that everywhere. (iOS already renders grouped.)
            .formStyle(.grouped)
            .navigationTitle(navigationTitle)
            .onAppear {
                lifecycle.install()
                restoreState()
                resolveModel()
                tryAutoRun()
            }
            .sheet(isPresented: $creatingVoice) {
                CreateVoiceView(modelDirectory: URL(fileURLWithPath: modelPath)) { newID in
                    refreshVoices()
                    selectedVoice = Voice.named(newID)
                    UserDefaults.standard.set(selectedVoice.id, forKey: Self.voiceKey)
                    status = "Created voice “\(selectedVoice.displayName)”."
                }
            }
            .fileImporter(
                isPresented: $importing,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    // Persist access for security-scoped URLs (sandboxed apps).
                    _ = url.startAccessingSecurityScopedResource()
                    if source == .localFolder {
                        localModelDir = url
                        FolderBookmark.store(url, forKey: Self.localDirKey)
                    } else {
                        hfHome = url
                        FolderBookmark.store(url, forKey: Self.hfHomeKey)
                    }
                    model = nil
                    loadInfo = ""
                    updateNotice = ""  // it named the dir we just switched away from
                    resolveModel()
                case .failure(let error):
                    status = "Selection failed: \(error.localizedDescription)"
                }
            }
        }
    }

    /// At launch, download (if needed) and **load** the model so it's ready to
    /// generate — but does NOT auto-start a generation (the user taps Generate &
    /// Play). Fires once per app launch. Disable entirely via `CHATTERBOX_NO_AUTORUN=1`.
    ///
    /// Re-download is triggered when the model dir is missing `T3LM.*` (handles
    /// iPhones whose previous snapshot still has the obsolete `T3Prefill` /
    /// `T3Decode` standalones), or when the snapshot is **provably** behind the Hub
    /// (a file whose content changed upstream stays present + complete locally, so
    /// nothing else catches it — see ``ModelRepository/staleFiles(dir:manifest:matching:)``).
    ///
    /// That second branch **is the fix.** `HubApi` has always repaired a remotely-changed
    /// file on its own (it re-fetches when the stored etag stops matching the remote one);
    /// what it never got was a reason to run. Autorun only called `download()` when the
    /// model was *missing*, so a device holding a complete-but-outdated snapshot never asked
    /// the Hub anything again — and shipped a pre-`f4af6b3` `S3CFM` into the ANE compiler.
    ///
    /// Three rules govern every branch below, and all three are load-bearing:
    ///   1. **A failed *update* never denies the user their model.** When a fetch fails, only
    ///      the missing-model branch (`.download`) may abandon the launch — it is the only one
    ///      with nothing to fall back on. Every other failure (offline, rate-limited, ENOSPC,
    ///      an update we chose not to apply) falls through to `autoLoad()` with the snapshot
    ///      already on disk, and reports what happened in `updateNotice` (which, unlike
    ///      `status`, survives the load).
    ///      The one further branch that ends a launch without loading is **not** a failed
    ///      fetch: it is a fetch that came back
    ///      ``ModelRepository/DownloadResult/isPartiallyApplied`` and could not be cleared by a
    ///      plain re-fetch. There we hold *positive evidence* the snapshot disagrees with the
    ///      Hub and may mix two revisions — loading that is the ANE-compiler crash this whole
    ///      path exists to prevent. It is still not deleted (rule 3): it stays on disk, the
    ///      user is told to tap Download, and the next launch re-checks.
    ///   2. **We do not spend hundreds of MB unasked.** An update over
    ///      ``ModelRepository/autoUpdateByteLimit`` is *offered*, not applied — the user taps
    ///      Download. (Without that gate, the next launch after a legitimate republish would
    ///      silently pull 120 MB on nano and 537 MB on multilingual, possibly on cellular.)
    ///   3. **We never destroy a model we cannot prove is broken.** Autorun has exactly ONE
    ///      automatic wipe (`autoDownload(force: true)`), and it is reachable only *after a load
    ///      actually failed* — a snapshot that will not load is proof, not suspicion. Every other
    ///      unhappy outcome, including a fetch that came back
    ///      ``ModelRepository/DownloadResult/isPartiallyApplied``, is repaired **in place** by a
    ///      plain re-fetch that deletes nothing; and when even that cannot *prove* the snapshot
    ///      is current, autorun neither wipes it nor loads it — it says so and stops
    ///      (``ModelRepository/repairDecision(afterRetry:)``). "Cannot tell" is not a licence to
    ///      delete the user's only copy of the model.
    private func tryAutoRun() {
        guard !autoRunDone else { return }
        guard ProcessInfo.processInfo.environment["CHATTERBOX_NO_AUTORUN"] == nil else { return }
        guard !busy, model == nil else { return }
        autoRunDone = true
        Task {
            // Step 1: decide what the snapshot on disk needs — nothing, a silent update, an
            // offer, or a first download.
            let (plan, manifest) = await launchPlan()
            // The check awaited the network; re-assert what it ran under.
            guard model == nil else { return }

            switch plan {
            case .load:
                updateNotice = ""  // current, unknowable, or not a snapshot we own

            case .offerUpdate(_, let bytes):
                // Rule 2: too big to spend on the user's behalf at launch. Their model still
                // loads; the notice tells them what it would cost, and Download applies it.
                updateNotice = "Model update available (\(formattedBytes(bytes))) — tap Download to update."

            case .download, .update:
                // `.download`: nothing on disk. `.update`: a KNOWN content mismatch (never
                // merely "couldn't reach the Hub"), small enough to apply unattended — the
                // manifest rides along so `download` doesn't ask the Hub twice.
                status = plan.fetchIsRequired
                    ? "Downloading T3LM…"
                    : "Updating model (\(formattedBytes(plan.bytes)))…"

                switch await autoDownload(
                    manifest: manifest,
                    label: plan.fetchIsRequired ? "Downloading T3LM" : "Updating model")
                {
                case .failure(let error):
                    // Rule 1, and this line is the whole of it: `fetchIsRequired` is true ONLY
                    // for `.download`, where there is genuinely nothing on disk to load. On a
                    // failed *update* we fall through to `autoLoad()` below with the snapshot
                    // the user already had — offline, rate-limited and ENOSPC must not cost
                    // them their model — and say so somewhere the load won't overwrite.
                    if plan.fetchIsRequired { return }
                    let reason = ModelRepository.isOutOfSpace(error)
                        ? "not enough free space" : error.localizedDescription
                    updateNotice = "Model update failed (\(reason)) — using the model already on this device."

                case .success(let result) where result.isPartiallyApplied:
                    // The snapshot disagrees with the manifest we handed the fetch: some of its
                    // files landed and some didn't. That *may* be the thing we fear — a new graph
                    // spec beside an old weight, the inconsistency that crashes the ANE compiler
                    // — so it must not be loaded as-is.
                    //
                    // **Re-fetch plain, with NO manifest, deleting nothing.** Manifest-less is
                    // the whole point, twice over:
                    //   - `download` doesn't re-read the Hub when it is *given* a manifest, so
                    //     handing back the same launch-time one would re-compare against the very
                    //     sizes that raised the flag — it could never clear, no matter how good
                    //     the snapshot is. (This is exactly why the CLI never sees the problem:
                    //     it supplies no manifest.)
                    //   - `isPartiallyApplied` is not proof of a two-revision mix. A commit that
                    //     landed on the Hub *between* our lookup and our fetch raises it on a
                    //     snapshot that is now FULLY CURRENT at `main`. Only current Hub truth
                    //     can tell the two apart — so go ask.
                    // The files that genuinely didn't land still carry the old commit's etag, so
                    // `HubApi` re-downloads exactly them; nothing is staged, moved or deleted.
                    status = "Model update was interrupted — retrying…"
                    let retry = await autoDownload(label: "Repairing model")
                    if ModelRepository.repairDecision(afterRetry: retry) == .load { break }

                    // We could not PROVE the snapshot is current — and we cannot prove it is
                    // broken either. The retry threw (offline / 429 / expired token / ENOSPC), or
                    // the Hub still disagrees, or the manifest lookup came back nil ("cannot
                    // tell"). None of that is evidence about the bytes on disk, and **we never
                    // destroy a model we cannot prove is broken**: force-wiping here would
                    // `removeDownload` the user's only copy on precisely the launch whose network
                    // already failed us, and drag the whole ~1 GB snapshot past the 25 MB gate
                    // this path exists to enforce. So: don't wipe, and don't load a snapshot we
                    // have positive reason to distrust. Say what to do and stop — the next launch
                    // re-checks, and the user's Download tap (and "Force re-download") still
                    // repairs it whenever they choose.
                    updateNotice = "Model update didn’t finish — tap Download to complete it."
                    status = "Model not loaded — the update didn’t finish."
                    return

                case .success(let result) where !result.stillStale.isEmpty:
                    // The fetch landed *nothing* (HubApi's offline/cancelled branches return
                    // success). Untouched snapshot → still loadable, still stale, re-checked
                    // next launch.
                    updateNotice = "Model update didn’t apply — using the model already on this device."

                case .success:
                    updateNotice = ""
                }
            }

            // Step 2: load with whatever snapshot is now on disk.
            status = ""
            if await autoLoad() == true { return }
            // Load failed — the local T3LM is likely the stale pre-host-mask
            // version that fails with `functionName must be nil`. Wipe + clean
            // re-download from HF, then load again. (One-time cost on upgrade.)
            status = "Existing T3LM didn't load — wiping local snapshot and re-downloading…"
            if case .failure = await autoDownload(force: true) {
                resolveModel()  // the wipe happened; re-sync `modelPath` with what's on disk
                return
            }
            _ = await autoLoad()
        }
    }

    /// Awaitable load wrapper for `tryAutoRun`. Returns true on success and
    /// leaves `model` set; on failure leaves `model == nil` and `busy == false`.
    @discardableResult
    private func autoLoad() async -> Bool {
        guard !modelPath.isEmpty else { return false }
        busy = true
        status = "Auto-loading model…"
        let start = Date()
        do {
            let modelURL = URL(fileURLWithPath: modelPath)
            model = try await ChatterboxCoreMLModel.load(
                from: modelURL,
                russianStress: await resolveRussianStress(),
                watermarker: await resolveWatermarker(modelDirectory: modelURL),
                computeUnits: pipelineComputeUnits)
            loadedVariant = model?.loadedVariant
            syncVariantToLoaded()
            loadInfo = String(format: "Model loaded in %.2fs.", Date().timeIntervalSince(start))
            status = "Ready to generate."
            busy = false
            return true
        } catch {
            loadInfo = ""
            status = "Auto-load failed: \(error.localizedDescription)"
            busy = false
            return false
        }
    }

    /// What autorun should do with what is on disk, and the Hub manifest that decided it —
    /// returned alongside so the update path can hand it straight to `download` and ask the
    /// Hub once, not twice.
    ///
    /// The decision itself lives in ``ModelRepository/launchPlan(modelOnDisk:dir:manifest:matching:autoApplyLimit:)``
    /// (pure, and unit-tested there); this only gathers the inputs. Costs at most
    /// `ModelRepository`'s 5 s lookup budget, and only when a model is already present —
    /// offline / unauthenticated / unreachable all answer "cannot tell" → `.load`.
    ///
    /// **Two dirs are never version-checked**, for the same reason: they are not snapshots we
    /// own, and a snapshot the app didn't create is not one it gets to silently replace.
    ///   - `source == .localFolder` — a folder the user chose (very often a locally-built
    ///     model, which differs from the Hub by construction).
    ///   - Any *discovered* dir that isn't the one `download` writes to — a Python-`hf_hub`
    ///     cache snapshot, or a `--local-dir` clone. "Repairing" one of those wouldn't even fix
    ///     the model we are about to load: `download` writes to `<base>/models/<repoId>`, a
    ///     **different directory**, so the user would get a surprise ~1 GB fetch and the stale
    ///     model would load anyway. (`ModelRepository.ownsSnapshot`. Can't arise on iOS.)
    private func launchPlan() async -> (ModelRepository.LaunchPlan, [String: Int]?) {
        guard hasT3LMOnDisk() else { return (.download, nil) }
        guard source == .hfHome else { return (.load, nil) }

        let dir = URL(fileURLWithPath: modelPath)
        // Not the dir we download into — a foreign layout (hf_hub cache / --local-dir clone).
        guard ModelRepository.ownsSnapshot(
            at: dir, hfHome: effectiveHFHome, repoId: variant.repoId)
        else { return (.load, nil) }

        // `busy` is held across the awaited lookup: Load / Download are gated on it, and a tap
        // must not race the update we may be about to start.
        busy = true
        defer { busy = false }
        status = "Checking for model updates…"
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let manifest = await ModelRepository.remoteManifest(
            repoId: variant.repoId, hfToken: token.isEmpty ? nil : token)
        // Only the files this variant actually downloads may condemn the snapshot (the
        // full-repo layouts also hold `README.md`, `onnx/`, …) — `matching:` is what enforces
        // that, so a model-card commit can't trigger a refetch.
        let plan = ModelRepository.launchPlan(
            modelOnDisk: true, dir: dir, manifest: manifest, matching: variant.runtimeGlobs)
        return (plan, manifest)
    }

    /// `537,254,468` → “537.3 MB”. The user is being asked to spend this, so they get to see
    /// it in the units their carrier bills in (`.file` is decimal, like the App Store's).
    private func formattedBytes(_ bytes: Int) -> String {
        Int64(bytes).formatted(.byteCount(style: .file))
    }

    /// True if the resolved model dir holds either a compiled or packaged T3LM.
    private func hasT3LMOnDisk() -> Bool {
        guard !modelPath.isEmpty else { return false }
        let dir = URL(fileURLWithPath: modelPath)
        let fm = FileManager.default
        return fm.fileExists(atPath: dir.appendingPathComponent("T3LM.mlmodelc").path)
            || fm.fileExists(atPath: dir.appendingPathComponent("T3LM.mlpackage").path)
    }

    /// Run the download path used by the Download button, but `await`-able so
    /// `tryAutoRun` can chain straight into load.
    ///
    /// Returns the full ``ModelRepository/DownloadResult`` rather than a `Bool`: a fetch can
    /// succeed and still not have landed what it went for (`HubApi.snapshot` returns success
    /// when it is offline or cancelled), so the caller has to be able to tell a real update
    /// from a no-op — and a *half-applied* commit from either.
    ///
    /// `force` wipes the snapshot and re-fetches it from clean (the recovery path when a
    /// present model fails to load — what we hold is provably unloadable). Without it
    /// nothing is deleted: `download` just re-runs the fetch, which is enough for `HubApi`
    /// to repair a file that changed upstream, and a failure leaves the snapshot exactly as
    /// it was.
    ///
    /// `manifest` passes on an already-fetched Hub listing so the *first* fetch of an update
    /// doesn't ask the Hub twice. **Omit it to re-read the Hub** — `download` skips the lookup
    /// entirely when it is handed one, so a caller that needs *current* truth (the
    /// partial-apply re-fetch) must pass `nil`.
    ///
    /// `label` names the phase in the progress line: the closure fires every few hundred ms and
    /// would otherwise stomp the caller's `status` with a fixed string milliseconds after it was
    /// set.
    private func autoDownload(
        force: Bool = false,
        manifest: [String: Int]? = nil,
        label: String = "Auto-downloading T3LM"
    ) async -> Result<ModelRepository.DownloadResult, Error> {
        busy = true
        defer { busy = false }
        let home = effectiveHFHome
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let result = try await ModelRepository.download(
                repoId: variant.repoId,
                hfHome: home,
                hfToken: token.isEmpty ? nil : token,
                matching: variant.runtimeGlobs,
                force: force,
                manifest: manifest
            ) { frac in
                Task { @MainActor in status = "\(label)… \(Int(frac * 100))%" }
            }
            modelPath = result.directory.path
            return .success(result)
        } catch {
            status = downloadFailureMessage(error)
            return .failure(error)
        }
    }

    /// A full disk is the one download failure with a different remedy, so it gets its own
    /// message — the generic one sends the user hunting a network problem they don't have.
    private func downloadFailureMessage(_ error: Error) -> String {
        ModelRepository.isOutOfSpace(error)
            ? "Not enough free space to update the model. Free up some space and try again."
            : "Download failed: \(error)"
    }

    /// Deletes the selected user voice's file and refreshes the picker.
    private func deleteSelectedVoice() {
        guard let url = selectedVoice.fileURL else { return }
        try? FileManager.default.removeItem(at: url)
        let removed = selectedVoice.displayName
        selectedVoice = .default
        refreshVoices()
        UserDefaults.standard.set(selectedVoice.id, forKey: Self.voiceKey)
        status = "Deleted voice “\(removed)”."
    }

    private func resolveModel() {
        switch source {
        case .localFolder:
            if let dir = localModelDir {
                modelPath = dir.path
                status = "Model folder selected. Tap Load."
            } else {
                modelPath = ""
                status = "Choose the model folder (the converter's out/ directory)."
            }
        case .hfHome:
            let found = effectiveHFHome.flatMap { ModelRepository.existingModelDirectory(hfHome: $0, repoId: variant.repoId) }
                ?? ModelRepository.existingModelDirectory(repoId: variant.repoId)
            if let found {
                modelPath = found.path
                status = "Model present. Tap Load."
            } else {
                modelPath = ""
                status = "No model found. Tap Download."
            }
        }
    }

    /// The Download button. Plain "Download" re-runs the snapshot fetch over whatever is on
    /// disk, which is all a stale file needs: `HubApi` re-downloads it when its remote etag
    /// no longer matches the one stored beside it. So re-downloading a broken model — the
    /// user's natural reaction to one — now actually repairs it, and nothing is deleted to
    /// do so.
    ///
    /// **This is the "yes" to the update autorun declined to make on its own.** The launch-time
    /// path applies an update only under ``ModelRepository/autoUpdateByteLimit`` and merely
    /// *offers* anything larger (`updateNotice`); tapping here is the user's consent, so there
    /// is no size gate on this path — whatever the Hub has, at whatever size, gets fetched.
    ///
    /// "Force re-download" still means what it says: wipe the snapshot, fetch from clean.
    private func downloadModel() {
        busy = true
        let home = effectiveHFHome
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let force = forceRedownload
        let selectedVariant = variant
        Task {
            do {
                if force {
                    model = nil  // the snapshot backing it is about to be deleted
                    loadInfo = ""
                }
                status = force ? "Clearing cached model and re-downloading…" : "Downloading model…"
                let result = try await ModelRepository.download(
                    repoId: selectedVariant.repoId,
                    hfHome: home,
                    hfToken: token.isEmpty ? nil : token,
                    matching: selectedVariant.runtimeGlobs,
                    force: force
                ) { frac in
                    Task { @MainActor in status = "Downloading model… \(Int(frac * 100))%" }
                }
                modelPath = result.directory.path
                if result.stillStale.isEmpty {
                    updateNotice = ""  // the offer (if any) has been taken
                    status = "Downloaded. Tap Load."
                } else {
                    // Succeeded without landing everything (offline/cancelled), or landed only
                    // part of a commit. Either way this snapshot is not what the Hub has, and a
                    // half-applied one must not be loaded at all.
                    updateNotice = result.isPartiallyApplied
                        ? "Update only partly applied — tap Download again (or Force re-download) before loading."
                        : "Update didn’t apply — check your connection and try again."
                    status = "Download incomplete."
                }
            } catch {
                status = downloadFailureMessage(error)
            }
            busy = false
        }
    }

    private func loadModel() {
        busy = true
        status = "Loading model…"
        let path = modelPath
        Task {
            let start = Date()
            do {
                let modelURL = URL(fileURLWithPath: path)
                model = try await ChatterboxCoreMLModel.load(
                    from: modelURL,
                    russianStress: await resolveRussianStress(),
                    watermarker: await resolveWatermarker(modelDirectory: modelURL),
                    computeUnits: pipelineComputeUnits)
                loadedVariant = model?.loadedVariant
                syncVariantToLoaded()
                let dt = Date().timeIntervalSince(start)
                loadInfo = String(format: "Model loaded in %.2fs.", dt)
                status = "Ready to generate."
            } catch {
                loadInfo = ""
                loadedVariant = nil
                status = "Load failed: \(error)"
            }
            busy = false
        }
    }

    /// The Russian stress source passed to `load`. For the multilingual model it
    /// downloads `ruaccent-coreml` using the SAME HF token + HF_HOME as the
    /// chatterbox model, so the two share one cache. Falls back to manual
    /// `+`/U+0301 marks if unavailable. The English models (turbo, nano) need no stresser.
    private func resolveRussianStress() async -> RussianStressing {
        guard variant == .multilingual else { return ManualRussianStress() }
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        status = "Loading Russian stress model…"
        return await RuAccentStress.makeOrFallback(
            downloadHFHome: effectiveHFHome,
            hfToken: token.isEmpty ? nil : token)
    }

    /// The watermarker passed to `load`. Perth (`perth-coreml`) marks every
    /// utterance the app produces — upstream chatterbox watermarks unconditionally,
    /// and there is deliberately no UI switch for it. Resolution is model dir first
    /// (so an offline install keeps marking), then `iliasaz/perth-coreml` with the
    /// SAME HF token + HF_HOME as the model, exactly like `resolveRussianStress`.
    /// Never fails the load: unavailable ⇒ unwatermarked audio and a logged error.
    private func resolveWatermarker(modelDirectory: URL) async -> AudioWatermarking {
        let token = hfToken.trimmingCharacters(in: .whitespacesAndNewlines)
        return await PerthWatermark.makeOrFallback(
            modelDirectory: modelDirectory,
            downloadHFHome: effectiveHFHome,
            hfToken: token.isEmpty ? nil : token,
            computeUnits: pipelineComputeUnits.watermark ?? .all)
    }

    /// A labeled slider row (`label … value`, `Slider`, and a one-line description
    /// of how the parameter affects the speech).
    @ViewBuilder
    private func sliderRow(_ label: String, value: Binding<Double>,
                           range: ClosedRange<Double>, step: Double, help: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .foregroundStyle(.secondary).monospacedDigit()
            }
            .font(.caption)
            Slider(value: value, in: range, step: step).disabled(busy)
            Text(help).font(.caption2).foregroundStyle(.secondary)
        }
    }

    /// Resets all generation parameters to the documented defaults.
    private func resetParameters() {
        temperature = 0.8; topP = 0.95; repetitionPenalty = 1.2; seed = 0
        topK = 1000; minP = 0.05; exaggeration = 0.5; cfgWeight = 0.5
        sentencePause = 0.25; commaPause = 0.25
    }

    /// Compute-unit placement passed to `ChatterboxCoreMLModel.load`. T3LM
    /// (prefill+decode) is pinned to the Neural Engine **unconditionally** — measured
    /// never worse than `.all`, and it removes the per-decode-step GPU submission that
    /// breaks background execution (background-mode plan §4). When "Run on Neural
    /// Engine" is on (the default, background-capable path) the synth encoder/CFM/
    /// vocoder are pinned to the ANE too; off is the foreground-only GPU synth A/B
    /// (slightly faster, but any GPU submission from the background fails).
    private var pipelineComputeUnits: PipelineComputeUnits {
        forceSynthANE ? .neuralEngine : PipelineComputeUnits(t3: .cpuAndNeuralEngine)
    }

    /// Builds `GenerationOptions` from the UI parameters, applying only those the
    /// active model supports (turbo + nano: top_k; multilingual: min_p/exaggeration/
    /// cfg_weight/language — see docs/parameters.md).
    private var generationOptions: GenerationOptions {
        let s: UInt64? = seed > 0 ? UInt64(seed) : nil
        // Playback pacing from the Parameters sliders (see GenerationOptions
        // .sentencePause / .commaPause). 0 = off; both models honor them.
        if variant == .multilingual {
            let lang = language.trimmingCharacters(in: .whitespaces)
            return GenerationOptions(
                temperature: Float(temperature), topK: 0, topP: Float(topP),
                repetitionPenalty: Float(repetitionPenalty), seed: s,
                minP: Float(minP), cfgWeight: Float(cfgWeight),
                exaggeration: Float(exaggeration), language: lang.isEmpty ? nil : lang,
                sentencePause: sentencePause, commaPause: commaPause)
        }
        return GenerationOptions(
            temperature: Float(temperature), topK: Int(topK), topP: Float(topP),
            repetitionPenalty: Float(repetitionPenalty), seed: s,
            sentencePause: sentencePause, commaPause: commaPause)
    }

    private func generate() {
        guard let model else { return }
        busy = true
        status = "Generating…"
        let prompt = text
        let options = generationOptions
        // `.userInitiated` so the pre-group stages (tokenize, prefill assembly) and the
        // stream consumer don't dip below the decode/synth legs' QoS — the QoS ladder
        // that keeps decode latency flat under ANE contention (background-mode plan §3).
        Task(priority: .userInitiated) {
            await runGeneration(model: model, prompt: prompt, options: options)
            busy = false
        }
    }

    /// The awaitable generation core (streams chunks → playback). Extracted from
    /// `generate()` so the on-device background-mode driver can `await` and repeat it.
    private func runGeneration(model: ChatterboxCoreMLModel, prompt: String, options: GenerationOptions) async {
        do {
            player.reset()
            // Bring the audio session + engine up NOW (not on first buffer) so
            // backgrounding during the pre-first-audio gap keeps us alive (§2).
            player.beginGeneration()
            defer { player.finishGeneration() }
            let start = Date()
            var chunks = 0
            var audioSeconds = 0.0
            var prefillTotal = 0.0   // CoreML T3LM prefill
            var decodeTotal = 0.0    // CoreML T3LM decode loop
            var synthTotal = 0.0     // CoreML S3 synth (encoder→CFM→vocoder)
            var tokenTotal = 0
            var decodeEngine = "CoreML"
            var stoppedByInterruption = false
            // Stream chunk-by-chunk: play each as soon as it's ready while
            // the model keeps generating the next (pipelined playback).
            for try await chunk in model.generateStream(prompt, voice: selectedVoice.conditioningURL, options: options) {
                chunks += 1
                prefillTotal += chunk.prefillTime
                decodeTotal += chunk.decodeTime
                synthTotal += chunk.synthTime
                tokenTotal += chunk.tokenCount
                decodeEngine = chunk.decodeBackend.rawValue
                let buffer = try AudioOutput.pcmBuffer(from: chunk.samples)
                try player.enqueue(buffer)
                audioSeconds += Double(buffer.frameLength) / buffer.format.sampleRate
                // Phase-stamped per-chunk timing at `.notice` — the always-visible
                // regression signal for the background/locked verification (§5):
                // decode ms/tok must stay flat across fg → bg → locked.
                let chunkMsPerTok = chunk.tokenCount > 0 ? chunk.decodeTime * 1000 / Double(chunk.tokenCount) : 0
                AppLog.playback.notice("[chunk \(chunks, privacy: .public)] phase=\(lifecycle.phase, privacy: .public) decode \(chunkMsPerTok, format: .fixed(precision: 1), privacy: .public) ms/tok (\(chunk.tokenCount, privacy: .public) tok) synth \(chunk.synthTime, format: .fixed(precision: 3), privacy: .public)s")
                if chunks == 1 {
                    let ttfa = Date().timeIntervalSince(start)
                    status = String(format: "Playing — first audio in %.2fs…", ttfa)
                } else {
                    status = "Playing chunk \(chunks)…"
                }
                // If a phone call or Siri took the audio session, pause HERE — at a chunk
                // boundary — instead of generating on into a session that can't play and
                // risking suspension mid-`MLModel.prediction` (§5). Awaiting stops
                // draining the package's bounded chunk channel, which back-pressures the
                // decode and synth legs, so the whole pipeline parks with us.
                if await player.awaitResumeIfInterrupted() == false {
                    stoppedByInterruption = true
                    break
                }
            }
            guard !stoppedByInterruption else {
                status = "Stopped after \(chunks) chunk(s) — the audio session was interrupted and could not resume."
                return
            }
            let total = Date().timeIntervalSince(start)
            let msPerTok = tokenTotal > 0 ? decodeTotal * 1000 / Double(tokenTotal) : 0
            let overhead = max(0, total - prefillTotal - decodeTotal - synthTotal)
            status = String(
                format: "Done: %d chunk(s), %.2fs audio.\nprefill (CoreML) %.2fs · decode (%@) %.2fs (%d tok, %.0f ms/tok) · synth (CoreML) %.2fs · overhead %.2fs · total %.2fs",
                chunks, audioSeconds, prefillTotal, decodeEngine as NSString, decodeTotal, tokenTotal, msPerTok, synthTotal, overhead, total)
        } catch {
            status = "Generate failed: \(error)"
        }
    }

}

#Preview {
    ContentView()
}
