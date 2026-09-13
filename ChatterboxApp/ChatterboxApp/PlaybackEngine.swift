import AVFoundation
import os
#if canImport(UIKit)
import UIKit
#endif

/// App-side unified-logging channels. The CoreML pipeline logs its own timings under
/// `com.chatterbox.coreml`; these are the *app* marks to grep for when verifying
/// background playback. Phase transitions and per-chunk
/// timings are logged at `.notice` so they surface in `idevicesyslog` without a
/// debugger and without enabling debug-level logging for the pipeline subsystem.
enum AppLog {
    static let subsystem = "com.iliasaz.ChatterboxApp"
    static let playback = Logger(subsystem: subsystem, category: "playback")
    static let lifecycle = Logger(subsystem: subsystem, category: "lifecycle")
}

/// Observes app/lock lifecycle and stamps a `.notice` MARK on every transition, so a
/// syslog timeline has precise phase boundaries to correlate with the pipeline's
/// decode/synth timings. Also takes a `beginBackgroundTask` assertion on background as
/// belt-and-suspenders next to the audio keep-alive (its expiry handler firing means
/// the keep-alive lapsed and iOS is about to suspend us — visible in the log).
///
/// iOS-only behavior; on macOS `install()` is a no-op (nothing suspends the app).
@MainActor
final class AppLifecycle {
    /// Coarse phase for stamping timing rows: `fg` (active), `bg` (backgrounded), or
    /// `locked` (protected data unavailable — the real locked regime, past the passcode
    /// grace window).
    private(set) var phase = "fg"
    private var installed = false
    private var observers: [NSObjectProtocol] = []
    #if canImport(UIKit)
    private var bgTask: UIBackgroundTaskIdentifier = .invalid
    #endif

    func install() {
        guard !installed else { return }
        installed = true
        #if canImport(UIKit)
        let c = NotificationCenter.default
        observers.append(c.addObserver(forName: UIApplication.didEnterBackgroundNotification,
                                       object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.didEnterBackground() }
        })
        observers.append(c.addObserver(forName: UIApplication.didBecomeActiveNotification,
                                       object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.didBecomeActive() }
        })
        observers.append(c.addObserver(forName: UIApplication.protectedDataWillBecomeUnavailableNotification,
                                       object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.setPhase("locked", event: "protectedDataWillBecomeUnavailable") }
        })
        observers.append(c.addObserver(forName: UIApplication.protectedDataDidBecomeAvailableNotification,
                                       object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.setPhase("bg", event: "protectedDataDidBecomeAvailable") }
        })
        #endif
    }

    #if canImport(UIKit)
    private func didEnterBackground() {
        setPhase("bg", event: "didEnterBackground")
        // Belt-and-suspenders next to the audio keep-alive. The handler only logs —
        // the audio session is what actually keeps us running; an expiry fire is the
        // signal that the keep-alive lapsed.
        let app = UIApplication.shared
        endBackgroundTask()
        bgTask = app.beginBackgroundTask(withName: "ChatterboxGeneration") { [weak self] in
            AppLog.lifecycle.notice("MARK bg task EXPIRING — audio keep-alive may have lapsed")
            MainActor.assumeIsolated { self?.endBackgroundTask() }
        }
    }

    private func didBecomeActive() {
        setPhase("fg", event: "didBecomeActive")
        endBackgroundTask()
    }

    private func endBackgroundTask() {
        guard bgTask != .invalid else { return }
        UIApplication.shared.endBackgroundTask(bgTask)
        bgTask = .invalid
    }
    #endif

    #if canImport(UIKit)
    private func setPhase(_ new: String, event: String) {
        phase = new
        AppLog.lifecycle.notice("MARK \(event, privacy: .public) → phase=\(new, privacy: .public)")
    }
    #endif
}

/// Plays 24 kHz PCM buffers through `AVAudioEngine`, enqueued back-to-back so streamed
/// chunks play gaplessly.
///
/// **Background/locked capability.** The
/// `audio` `UIBackgroundMode` keeps the app alive only *while an audio session is
/// actively rendering*, so:
///   - the session + engine come up when generation **starts** (`beginGeneration`),
///     not on the first synthesized buffer — otherwise the 1–10 s gap before first
///     audio (and any inter-chunk drain) would suspend a backgrounded app;
///   - a near-silent keep-alive tone renders continuously via an `AVAudioSourceNode`
///     (the mechanism proven to sustain backgrounded+locked execution for 15+ min in
///     `BGDecodeProbe`), so the output never goes idle mid-generation;
///   - everything is torn down once generation **and** playback both finish
///     (`finishGeneration` + the last buffer draining), so idle battery is unaffected.
///
/// The audio session is activated **once per generation**, never per buffer — repeated
/// per-chunk `setActive(true)` is a known playback CPU hog.
@MainActor
final class Player {
    private let engine = AVAudioEngine()
    private let node = AVAudioPlayerNode()
    private var keepAlive: AVAudioSourceNode?
    /// Phase accumulator for the keep-alive tone, boxed in a `Sendable` reference so the
    /// real-time render block can mutate it across the `@Sendable` boundary. Only the
    /// audio thread ever touches `value`, so the unchecked conformance is sound.
    private final class PhaseBox: @unchecked Sendable { var value = 0.0 }
    private let keepAlivePhase = PhaseBox()
    private var nodeAttached = false
    private var sessionActive = false
    /// Buffers scheduled on `node` that have not finished playing yet.
    private var pendingBuffers = 0
    /// True from `beginGeneration()` until `finishGeneration()`. While true the engine
    /// is kept rendering (via the keep-alive tone) even when no real audio is queued.
    private var generating = false
    /// Set while an audio-session interruption (phone call, Siri) is active: the system
    /// has deactivated our session, so incoming buffers are held here and flushed when
    /// the interruption ends with `.shouldResume`.
    private var interrupted = false
    private var backlog: [AVAudioPCMBuffer] = []
    /// Callers parked in ``awaitResumeIfInterrupted()`` for the duration of an
    /// interruption, keyed so a cancelled caller can be withdrawn without any risk of
    /// resuming a continuation twice.
    private var interruptionWaiters: [UInt64: CheckedContinuation<Bool, Never>] = [:]
    private var nextWaiterID: UInt64 = 0
    // `nonisolated(unsafe)`: set once in `init`, read once in the nonisolated `deinit`;
    // the token itself is not `Sendable` but its access is effectively single-threaded.
    nonisolated(unsafe) private var interruptionObserver: (any NSObjectProtocol)?

    init() {
        installInterruptionHandling()
    }

    deinit {
        if let interruptionObserver { NotificationCenter.default.removeObserver(interruptionObserver) }
    }

    /// Bring the audio session + engine up at generation start so backgrounding during
    /// the pre-first-audio gap doesn't suspend the app. Idempotent within a generation.
    func beginGeneration() {
        generating = true
        interrupted = false
        backlog.removeAll(keepingCapacity: true)
        do {
            try activateSession()
            startKeepAlive()
            if !engine.isRunning { try engine.start() }
            AppLog.playback.notice("beginGeneration — session active, engine rendering keep-alive")
        } catch {
            AppLog.playback.error("beginGeneration failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Appends a buffer to the playback queue. The player node is attached lazily on the
    /// first buffer (its format is known only then); the engine is already running from
    /// `beginGeneration`. During an interruption the buffer is held for later flush.
    func enqueue(_ buffer: AVAudioPCMBuffer) throws {
        if interrupted {
            backlog.append(buffer)
            return
        }
        // Fallback for a caller that skipped beginGeneration (e.g. a direct/serial use):
        // bring the engine up here. Session activation stays out of the per-buffer path.
        if !engine.isRunning {
            if !sessionActive { try activateSession() }
            startKeepAlive()
            try engine.start()
        }
        schedule(buffer)
    }

    /// Generation produced its last chunk. Keep rendering until the queue drains, then
    /// tear the session down.
    func finishGeneration() {
        generating = false
        teardownIfIdle()
    }

    /// Suspends the caller for as long as an audio-session interruption (phone call,
    /// Siri) is active, so the generation loop pauses **at a chunk boundary** instead of
    /// being suspended mid-`MLModel.prediction` (background-mode plan §5). Returns `true`
    /// to keep generating — either there was no interruption, or the session came back —
    /// and `false` when the session could not be reactivated, meaning the caller should
    /// stop cleanly at this boundary.
    ///
    /// Parking here is what actually pauses the *pipeline*: the package's
    /// `BoundedChunkChannel` back-pressures the decode and synth legs as soon as this
    /// consumer stops draining chunks.
    func awaitResumeIfInterrupted() async -> Bool {
        guard interrupted else { return true }
        AppLog.playback.notice("MARK generation paused at chunk boundary — session interrupted")
        let id = nextWaiterID
        nextWaiterID &+= 1
        return await withTaskCancellationHandler {
            await withCheckedContinuation { continuation in
                guard !Task.isCancelled else { return continuation.resume(returning: false) }
                interruptionWaiters[id] = continuation
            }
        } onCancel: {
            Task { @MainActor in self.withdrawWaiter(id) }
        }
    }

    /// Releases every caller parked in ``awaitResumeIfInterrupted()``.
    private func resolveInterruptionWaiters(_ resume: Bool) {
        guard !interruptionWaiters.isEmpty else { return }
        let parked = interruptionWaiters.values
        interruptionWaiters.removeAll(keepingCapacity: true)
        AppLog.playback.notice(
            "MARK generation \(resume ? "resumed" : "stopped", privacy: .public) after interruption")
        for continuation in parked { continuation.resume(returning: resume) }
    }

    /// Withdraws one cancelled caller. `removeValue` is what makes this safe: whichever
    /// of cancellation and interruption-resolution runs first takes the continuation, and
    /// the other finds nothing.
    private func withdrawWaiter(_ id: UInt64) {
        interruptionWaiters.removeValue(forKey: id)?.resume(returning: false)
    }

    /// Stops and clears any queued playback (call before a new generation).
    func reset() {
        if node.isPlaying { node.stop() }
        pendingBuffers = 0
        backlog.removeAll(keepingCapacity: true)
    }

    // MARK: - Internals

    private func schedule(_ buffer: AVAudioPCMBuffer) {
        if !nodeAttached {
            engine.attach(node)
            engine.connect(node, to: engine.mainMixerNode, format: buffer.format)
            nodeAttached = true
        }
        pendingBuffers += 1
        // `@Sendable` for the same reason as the keep-alive block: this completion
        // handler is invoked on an AVAudioEngine thread, not the main actor.
        node.scheduleBuffer(buffer, at: nil, options: [], completionCallbackType: .dataPlayedBack) { @Sendable [weak self] _ in
            Task { @MainActor in self?.bufferFinished() }
        }
        if !node.isPlaying { node.play() }
    }

    private func bufferFinished() {
        pendingBuffers = max(0, pendingBuffers - 1)
        teardownIfIdle()
    }

    private func teardownIfIdle() {
        guard !generating, pendingBuffers == 0, backlog.isEmpty else { return }
        stopKeepAlive()
        if node.isPlaying { node.stop() }
        if engine.isRunning { engine.stop() }
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        #endif
        sessionActive = false
        AppLog.playback.notice("playback idle — engine stopped, session deactivated")
    }

    private func activateSession() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
        #endif
        sessionActive = true
    }

    /// A continuous ~-60 dBFS 220 Hz tone into the mixer. Its only job is to keep the
    /// output actively rendering so the `audio` background mode holds; at 0.001
    /// amplitude it is inaudible under real speech and in inter-chunk gaps.
    private func startKeepAlive() {
        guard keepAlive == nil else { return }
        let sr = 44_100.0
        guard let fmt = AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1) else { return }
        let box = keepAlivePhase   // Sendable reference; the render block owns `value`.
        // `@Sendable` is load-bearing: this class is `@MainActor`, so without it the
        // render closure would inherit main-actor isolation and Swift would insert an
        // executor-isolation check that traps (EXC_BREAKPOINT) when the real-time audio
        // thread calls it. `@Sendable` makes the closure nonisolated (like the probe's
        // non-`@MainActor` KeepAliveAudio).
        let node = AVAudioSourceNode(format: fmt) { @Sendable _, _, frameCount, audioBufferList -> OSStatus in
            let abl = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let inc = 2.0 * Double.pi * 220.0 / sr
            var phase = box.value
            for frame in 0..<Int(frameCount) {
                let v = Float(sin(phase)) * 0.001
                phase += inc
                if phase > 2.0 * Double.pi { phase -= 2.0 * Double.pi }
                for buf in abl {
                    buf.mData?.assumingMemoryBound(to: Float.self)[frame] = v
                }
            }
            box.value = phase
            return noErr
        }
        engine.attach(node)
        engine.connect(node, to: engine.mainMixerNode, format: fmt)
        keepAlive = node
    }

    private func stopKeepAlive() {
        guard let node = keepAlive else { return }
        engine.detach(node)
        keepAlive = nil
    }

    // MARK: - Interruptions (§2/§5)

    private func installInterruptionHandling() {
        #if os(iOS)
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification, object: nil, queue: .main
        ) { [weak self] note in
            // Pull the Sendable primitives out here (on the main queue) so the
            // non-Sendable `Notification` never crosses into the actor-isolated method.
            let rawType = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            let rawOptions = note.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt
            MainActor.assumeIsolated { self?.handleInterruption(rawType: rawType, rawOptions: rawOptions) }
        }
        #endif
    }

    #if os(iOS)
    private func handleInterruption(rawType: UInt?, rawOptions: UInt?) {
        guard let rawType, let type = AVAudioSession.InterruptionType(rawValue: rawType) else { return }
        switch type {
        case .began:
            // The system has deactivated our session and paused the engine. Hold new
            // buffers until we can resume so generation isn't lost.
            interrupted = true
            AppLog.playback.notice("MARK audio interruption began")
        case .ended:
            let opts = rawOptions.map(AVAudioSession.InterruptionOptions.init(rawValue:)) ?? []
            let resumable = opts.contains(.shouldResume) && (generating || pendingBuffers > 0 || !backlog.isEmpty)
            AppLog.playback.notice("MARK audio interruption ended shouldResume=\(opts.contains(.shouldResume), privacy: .public) resuming=\(resumable, privacy: .public)")
            interrupted = false
            guard resumable else {
                // Not resumable: release the generation loop so it stops at its chunk
                // boundary rather than running on with nowhere to play (§5).
                resolveInterruptionWaiters(false)
                teardownIfIdle()
                return
            }
            do {
                try activateSession()
                startKeepAlive()
                if !engine.isRunning { try engine.start() }
                let held = backlog
                backlog.removeAll(keepingCapacity: true)
                for buffer in held { schedule(buffer) }
                if !node.isPlaying && pendingBuffers > 0 { node.play() }
                resolveInterruptionWaiters(true)
            } catch {
                AppLog.playback.error("interruption resume failed: \(error.localizedDescription, privacy: .public)")
                resolveInterruptionWaiters(false)
            }
        @unknown default:
            break
        }
    }
    #endif
}
