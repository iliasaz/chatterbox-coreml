import Foundation
import ChatterboxCoreML

/// Which upstream demo application a bundled reference voice comes from — and
/// therefore which models it is offered for.
///
/// This is not decoration. Conds themselves are model-agnostic (the shared voice
/// encoders bake in no T3 weights), so any voice *loads* under any variant, and an
/// earlier version of this catalog showed them all everywhere on that basis. But
/// loading and working are different claims, and the second one is false for short
/// prompts on turbo: every reference under ~5 s of source audio degenerates there —
/// a one-sentence request comes back as 0.2 s of nothing or 27 s of babble — while
/// the same conds are perfectly well behaved on the multilingual model, whose
/// Perceiver resampler maps a variable-length prompt onto a fixed query set.
/// Upstream splits the sets for the same reason: their turbo app offers one long
/// English prompt, their multilingual app offers one prompt per language, ten of
/// which are under five seconds. Measured, per voice; see `Voices/README.md`.
enum VoiceDemoSet: Sendable {
    /// `gradio_tts_turbo_app.py` — one ~30 s English prompt. Shown for turbo/nano.
    case turbo
    /// `multilingual_app.py` — one prompt per supported language. Shown for multilingual.
    case multilingual

    func matches(_ variant: ModelRepository.Variant) -> Bool {
        switch variant {
        case .turbo, .nano: return self == .turbo
        case .multilingual: return self == .multilingual
        }
    }
}

/// A selectable reference voice. `default` uses the model's built-in
/// `default-conds.safetensors` (loaded from the model directory); named voices
/// load a `*-conds.safetensors` either bundled under `Voices/` (the reference set
/// below) or created on-device under `<Documents>/Voices/` (user clones via
/// `VoiceCloner`). The selected voice's `conditioningURL` is passed to
/// `ChatterboxCoreMLModel.generateStream(_:voice:options:)`.
struct Voice: Identifiable, Hashable {
    let id: String
    let displayName: String
    /// Bundle resource stem (e.g. `ru_m-conds`) for a shipped voice, else nil.
    let resourceName: String?
    /// Absolute file URL for a user-created voice (under Documents/Voices), else nil.
    let fileURL: URL?
    /// BCP-47 code of the language the reference recording is in, matching
    /// `SupportedLanguage.code`. `nil` for the model default and user clones, which
    /// belong to no particular language and are shown for every model.
    let language: String?
    /// Which upstream demo set this came from; `nil` = always shown.
    let demoSet: VoiceDemoSet?

    init(id: String, displayName: String, resourceName: String? = nil,
         fileURL: URL? = nil, language: String? = nil, demoSet: VoiceDemoSet? = nil) {
        self.id = id
        self.displayName = displayName
        self.resourceName = resourceName
        self.fileURL = fileURL
        self.language = language
        self.demoSet = demoSet
    }

    /// The built-in voice baked into the model directory.
    static let `default` = Voice(id: "default", displayName: "Default")

    /// Bundled reference voices: Resemble AI's own published demo prompts, converted
    /// to conds on-device with `VoiceCloner`. Anonymous speakers named by language and
    /// gender, exactly as upstream names them — no personas are invented here, and no
    /// identifiable person is cloned. Provenance and source URLs: `Voices/README.md`.
    ///
    /// Order mirrors `SupportedLanguage.all`: the validated Russian target first, then
    /// alphabetical by code. Japanese carries no gender because upstream's filename
    /// (`ja/ja_prompts1.flac`) records none, and guessing one would be inventing detail.
    static let references: [Voice] = [
        // Turbo / nano — `gradio_tts_turbo_app.py`'s default prompt (~30 s).
        Voice(id: "female_random_podcast", displayName: "English (female)",
              resourceName: "female_random_podcast-conds", language: "en", demoSet: .turbo),
        // Multilingual — `multilingual_app.py`'s per-language prompts.
        Voice(id: "ru_m", displayName: "Russian (male)", resourceName: "ru_m-conds", language: "ru", demoSet: .multilingual),
        Voice(id: "ar_f", displayName: "Arabic (female)", resourceName: "ar_f-conds", language: "ar", demoSet: .multilingual),
        Voice(id: "da_m1", displayName: "Danish (male)", resourceName: "da_m1-conds", language: "da", demoSet: .multilingual),
        Voice(id: "de_f1", displayName: "German (female)", resourceName: "de_f1-conds", language: "de", demoSet: .multilingual),
        Voice(id: "el_m", displayName: "Greek (male)", resourceName: "el_m-conds", language: "el", demoSet: .multilingual),
        Voice(id: "en_f1", displayName: "English (female)", resourceName: "en_f1-conds", language: "en", demoSet: .multilingual),
        Voice(id: "es_f1", displayName: "Spanish (female)", resourceName: "es_f1-conds", language: "es", demoSet: .multilingual),
        Voice(id: "fi_m", displayName: "Finnish (male)", resourceName: "fi_m-conds", language: "fi", demoSet: .multilingual),
        Voice(id: "fr_f1", displayName: "French (female)", resourceName: "fr_f1-conds", language: "fr", demoSet: .multilingual),
        Voice(id: "he_m1", displayName: "Hebrew (male)", resourceName: "he_m1-conds", language: "he", demoSet: .multilingual),
        Voice(id: "hi_f1", displayName: "Hindi (female)", resourceName: "hi_f1-conds", language: "hi", demoSet: .multilingual),
        Voice(id: "it_m1", displayName: "Italian (male)", resourceName: "it_m1-conds", language: "it", demoSet: .multilingual),
        Voice(id: "ja", displayName: "Japanese", resourceName: "ja-conds", language: "ja", demoSet: .multilingual),
        Voice(id: "ko_f", displayName: "Korean (female)", resourceName: "ko_f-conds", language: "ko", demoSet: .multilingual),
        Voice(id: "ms_f", displayName: "Malay (female)", resourceName: "ms_f-conds", language: "ms", demoSet: .multilingual),
        Voice(id: "nl_m", displayName: "Dutch (male)", resourceName: "nl_m-conds", language: "nl", demoSet: .multilingual),
        Voice(id: "no_f1", displayName: "Norwegian (female)", resourceName: "no_f1-conds", language: "no", demoSet: .multilingual),
        Voice(id: "pl_m", displayName: "Polish (male)", resourceName: "pl_m-conds", language: "pl", demoSet: .multilingual),
        Voice(id: "pt_m1", displayName: "Portuguese (male)", resourceName: "pt_m1-conds", language: "pt", demoSet: .multilingual),
        Voice(id: "sv_f", displayName: "Swedish (female)", resourceName: "sv_f-conds", language: "sv", demoSet: .multilingual),
        Voice(id: "sw_m", displayName: "Swahili (male)", resourceName: "sw_m-conds", language: "sw", demoSet: .multilingual),
        Voice(id: "tr_m", displayName: "Turkish (male)", resourceName: "tr_m-conds", language: "tr", demoSet: .multilingual),
        Voice(id: "zh_f2", displayName: "Chinese (female)", resourceName: "zh_f2-conds", language: "zh", demoSet: .multilingual),
    ]

    /// Directory holding user-created voices: `<Documents>/Voices/`.
    static var userVoicesDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Voices", isDirectory: true)
    }

    /// User-created voices discovered under `userVoicesDirectory`, sorted by name.
    /// File `<name>-conds.safetensors` → id `user.<name>`, display the de-slugged name.
    static func userVoices() -> [Voice] {
        let dir = userVoicesDirectory
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            return []
        }
        return entries
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasSuffix("-conds.safetensors") }
            .map { url -> Voice in
                let stem = url.lastPathComponent.replacingOccurrences(of: "-conds.safetensors", with: "")
                return Voice(id: "user.\(stem)", displayName: prettify(stem), resourceName: nil, fileURL: url)
            }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
    }

    /// Everything (default + every reference + user) — used for `named` lookup so a
    /// saved selection resolves regardless of the current model.
    static func allVoices() -> [Voice] { [.default] + references + userVoices() }

    /// Voices to show for `variant`: default + that variant's demo set + user voices
    /// (user clones are always shown — the caller recorded them and knows what they
    /// are). See ``VoiceDemoSet`` for why the reference set is filtered rather than
    /// shown whole.
    static func allVoices(for variant: ModelRepository.Variant) -> [Voice] {
        [.default] + references.filter { $0.demoSet?.matches(variant) ?? true } + userVoices()
    }

    /// Look up a voice by `id` across default/bundled/user, falling back to default.
    static func named(_ id: String?) -> Voice {
        allVoices().first { $0.id == id } ?? .default
    }

    /// URL of this voice's conditioning file, or nil for the built-in default
    /// (which `generateStream(voice:)` interprets as "use the model's default").
    var conditioningURL: URL? {
        if let fileURL { return fileURL }
        guard let resourceName else { return nil }
        return Bundle.main.url(forResource: resourceName, withExtension: "safetensors")
    }

    /// True for a user-created voice (deletable/renamable).
    var isUserVoice: Bool { fileURL != nil }

    /// Title-cases a file stem: "my_cool_voice" → "My Cool Voice".
    private static func prettify(_ stem: String) -> String {
        stem.replacingOccurrences(of: "_", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}

/// A language the multilingual model supports (code + display name), for the
/// language Picker. Mirrors `ChatterboxMultilingualTTS.SUPPORTED_LANGUAGES`
/// (23 languages); Russian is the validated/default target.
struct SupportedLanguage: Identifiable, Hashable {
    let code: String
    let name: String
    var id: String { code }

    static let all: [SupportedLanguage] = [
        .init(code: "ru", name: "Russian"),       // default / validated target first
        .init(code: "ar", name: "Arabic"),   .init(code: "da", name: "Danish"),
        .init(code: "de", name: "German"),   .init(code: "el", name: "Greek"),
        .init(code: "en", name: "English"),  .init(code: "es", name: "Spanish"),
        .init(code: "fi", name: "Finnish"),  .init(code: "fr", name: "French"),
        .init(code: "he", name: "Hebrew"),   .init(code: "hi", name: "Hindi"),
        .init(code: "it", name: "Italian"),  .init(code: "ja", name: "Japanese"),
        .init(code: "ko", name: "Korean"),   .init(code: "ms", name: "Malay"),
        .init(code: "nl", name: "Dutch"),    .init(code: "no", name: "Norwegian"),
        .init(code: "pl", name: "Polish"),   .init(code: "pt", name: "Portuguese"),
        .init(code: "sv", name: "Swedish"),  .init(code: "sw", name: "Swahili"),
        .init(code: "tr", name: "Turkish"),  .init(code: "zh", name: "Chinese"),
    ]
}
