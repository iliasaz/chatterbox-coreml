import SwiftUI
import AVFoundation
import UniformTypeIdentifiers
import ChatterboxCoreML

/// Records (or imports) a reference clip and clones it into a `*-conds.safetensors`
/// under `<Documents>/Voices/`, so it appears in the voice picker. Needs the
/// loaded model directory (where the 5 conditioning `.mlpackage`s live) and
/// `NSMicrophoneUsageDescription` for recording.
struct CreateVoiceView: View {
    /// Directory holding the conditioning models (the resolved model path).
    let modelDirectory: URL
    /// Called with the new voice's `id` after a successful clone.
    let onCreated: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @StateObject private var recorder = VoiceRecorder()
    @State private var name = ""
    @State private var importedURL: URL?
    @State private var importing = false
    @State private var busy = false
    @State private var status = ""

    /// The reference clip to clone: a fresh recording takes priority over an import.
    private var sourceURL: URL? { recorder.recordedURL ?? importedURL }

    /// A short passage to read aloud while recording (~25-30 s). Deliberately mixes
    /// a whisper, a question, an excited exclamation, tender dialogue, and a
    /// triumphant close so the clone captures a wide range of intonation and
    /// emotion — a flat, monotone read produces a flat, monotone voice.
    private static let readAloud = """
    “Wait… did you hear that?” Mara whispered, her eyes wide.

    A low growl rumbled from the cave — and then, all at once, a tiny dragon \
    tumbled out!

    “Oh! You’re just a baby,” she laughed, scooping him up. “But how on earth \
    did you get all the way out here, little one?”

    The dragon sniffed, sneezed a puff of smoke, and curled up in her arms.

    What an extraordinary, unforgettable morning this had turned out to be!
    """

    private var canCreate: Bool {
        !busy && sourceURL != nil &&
        !name.trimmingCharacters(in: .whitespaces).isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Read this aloud — with feeling!") {
                    Text(Self.readAloud)
                        .font(.callout)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Perform it — whisper the whisper, sound excited at the “!”, lift your voice at the “?”. The more expression you give, the more of your voice the clone captures.")
                        .font(.caption2).foregroundStyle(.secondary)
                }

                Section("Reference audio") {
                    Text("Tap Record and read the passage above (about 25–30 seconds), then tap Stop. Or import an existing clip of clear, expressive speech.")
                        .font(.caption).foregroundStyle(.secondary)

                    Button {
                        recorder.toggle()
                    } label: {
                        HStack {
                            Image(systemName: recorder.isRecording ? "stop.circle.fill" : "record.circle")
                            Text(recorder.isRecording
                                 ? String(format: "Stop (%.1fs)", recorder.elapsed)
                                 : (recorder.recordedURL == nil ? "Record" : "Re-record"))
                            if recorder.isRecording { Spacer(); ProgressView() }
                        }
                    }
                    .buttonStyle(.borderless)
                    .disabled(busy)

                    Button {
                        importing = true
                    } label: {
                        Label(importedURL == nil ? "Import audio file…" : "Imported: \(importedURL!.lastPathComponent)",
                              systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.borderless)
                    .disabled(busy || recorder.isRecording)

                    if let src = sourceURL {
                        Text("Using: \(src.lastPathComponent)")
                            .font(.caption2).foregroundStyle(.green)
                    }
                }

                Section("Name") {
                    TextField("Voice name (e.g. My Voice)", text: $name)
                        #if os(iOS)
                        .textInputAutocapitalization(.words)
                        #endif
                        .disabled(busy)
                }

                if !status.isEmpty {
                    Section {
                        HStack(spacing: 8) {
                            if busy { ProgressView() }
                            Text(status).font(.callout).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Create Voice")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { recorder.cancel(); dismiss() }.disabled(busy)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Create") { create() }.disabled(!canCreate)
                }
            }
            .fileImporter(isPresented: $importing,
                          allowedContentTypes: [.audio, .mpeg4Audio, .wav, .mp3],
                          allowsMultipleSelection: false) { result in
                if case .success(let urls) = result, let url = urls.first {
                    _ = url.startAccessingSecurityScopedResource()
                    importedURL = url
                    recorder.discard()      // an import overrides any prior recording
                }
            }
        }
    }

    private func create() {
        guard let src = sourceURL else { return }
        let slug = name.trimmingCharacters(in: .whitespaces)
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !slug.isEmpty else { status = "Please enter a valid name."; return }

        busy = true
        status = "Cloning voice…"
        Task {
            do {
                let dir = Voice.userVoicesDirectory
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let outURL = dir.appendingPathComponent("\(slug)-conds.safetensors")
                let cloner = try await VoiceCloner(modelDirectory: modelDirectory)
                try cloner.cloneVoice(from: src, to: outURL)
                status = "Done."
                onCreated("user.\(slug)")
                dismiss()
            } catch {
                status = "Failed: \(error.localizedDescription)"
            }
            busy = false
        }
    }
}

/// Minimal AVAudioRecorder wrapper: records mono AAC to a temp file, exposing
/// `isRecording` + elapsed time for the UI. The system prompts for mic access
/// (NSMicrophoneUsageDescription) on first record.
@MainActor
final class VoiceRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var elapsed: TimeInterval = 0
    @Published private(set) var recordedURL: URL?

    private var recorder: AVAudioRecorder?
    private var timer: Timer?
    private var startedAt: Date?

    func toggle() { isRecording ? stop() : start() }

    private func start() {
        Task { @MainActor in
            if await Self.requestPermission() { beginRecording() }
        }
    }

    private func beginRecording() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("voice_ref_\(UUID().uuidString).m4a")
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playAndRecord, mode: .default)
        try? session.setActive(true)
        #endif
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 24000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]
        do {
            let r = try AVAudioRecorder(url: url, settings: settings)
            r.record()
            recorder = r
            recordedURL = nil
            startedAt = Date()
            elapsed = 0
            isRecording = true
            timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    guard let self, let s = self.startedAt else { return }
                    self.elapsed = Date().timeIntervalSince(s)
                }
            }
        } catch {
            isRecording = false
        }
    }

    private func stop() {
        recorder?.stop()
        recordedURL = recorder?.url
        recorder = nil
        timer?.invalidate(); timer = nil
        isRecording = false
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false)
        #endif
    }

    /// Stops without keeping the file (used on cancel).
    func cancel() {
        if isRecording { stop() }
        discard()
    }

    /// Drops any recorded file reference (e.g. when an import supersedes it).
    func discard() {
        if let url = recordedURL { try? FileManager.default.removeItem(at: url) }
        recordedURL = nil
    }

    private static func requestPermission() async -> Bool {
        await withCheckedContinuation { cont in
            #if os(iOS)
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { cont.resume(returning: $0) }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { cont.resume(returning: $0) }
            }
            #else
            AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
            #endif
        }
    }
}
