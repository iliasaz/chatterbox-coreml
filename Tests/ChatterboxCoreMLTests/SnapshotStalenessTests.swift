import Testing
import Foundation
@testable import ChatterboxCoreML

/// A throwaway directory standing in for a snapshot dir.
private func makeDir(_ prefix: String = "stale") throws -> URL {
    let dir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("\(prefix)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// Writes a file of exactly `bytes` bytes, creating parents.
private func write(_ bytes: Int, to url: URL) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(count: bytes).write(to: url)
}

/// Writes a file whose reported size is `bytes` **without allocating them** (APFS sparse
/// file). Needed because the sizes under test are real ones — a 211 MB nano weight — and the
/// only thing any of this reads is `st_size`.
private func writeSparse(_ bytes: Int, to url: URL) throws {
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    FileManager.default.createFile(atPath: url.path, contents: nil)
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: UInt64(bytes))
    try handle.close()
}

/// A relative path from the real repo, used across these tests.
private let cfmSpec = "S3CFM.mlpackage/Data/com.apple.CoreML/model.mlmodel"
private let t3lmWeight = "T3LM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"

/// Offline units for snapshot staleness (no network, no token).
///
/// The load-bearing property is the **safety invariant**: `staleFiles` may only name a file
/// on a positively-known content mismatch, because a non-empty answer is what makes the app
/// re-download. Every "cannot tell" — unreachable Hub, no token, malformed JSON, a dir we
/// don't have — must answer empty, or an airplane-mode launch would burn the user's launch
/// (and, on cellular, their data) chasing bytes it can't reach.
struct SnapshotStalenessTests {
    /// The core of the real bug: the file is present and complete, only its *size*
    /// betrays that it is an older build. (Byte counts are the actual ones — a
    /// pre-`f4af6b3` `S3CFM` `model.mlmodel` vs the fixed one on the Hub.)
    @Test func staleWhenALocalFileDiffersInSizeFromTheHub() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(712_370, to: dir.appendingPathComponent(cfmSpec))

        #expect(ModelRepository.staleFiles(dir: dir, manifest: [cfmSpec: 721_085]) == [cfmSpec])
        #expect(ModelRepository.staleFiles(dir: dir, manifest: [cfmSpec: 712_370]).isEmpty)
    }

    /// `existingModelDirectory` also resolves the Python-`huggingface_hub` cache and
    /// a `--local-dir` clone — layouts that hold the **full repo**, `README.md` and
    /// `.gitattributes` included. Those are files we never download, so they must not be
    /// able to condemn a snapshot: otherwise a model-card commit flips staleness and
    /// triggers an unprompted ~1 GB refetch of a model that is perfectly current.
    @Test func unmanagedFilesCannotCondemnASnapshot() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(721_085, to: dir.appendingPathComponent(cfmSpec))  // current
        try write(4_096, to: dir.appendingPathComponent("README.md"))  // full-repo layout
        try write(1_500, to: dir.appendingPathComponent(".gitattributes"))

        // The ONLY difference from the Hub is the model card. Not a file we fetch → not stale.
        let manifest = [cfmSpec: 721_085, "README.md": 9_999, ".gitattributes": 2_222]
        #expect(ModelRepository.staleFiles(dir: dir, manifest: manifest).isEmpty)

        // Sanity: the same check still catches a managed file, so the glob filter didn't
        // simply disable everything.
        #expect(ModelRepository.staleFiles(
            dir: dir, manifest: manifest.merging([cfmSpec: 712_370]) { _, b in b }) == [cfmSpec])
    }

    /// A hand-injected local build (a model copied into the app for testing) differs in size from
    /// the Hub by construction — which is exactly the "stale" signal. The sentinel says "these
    /// bytes are mine", so the version check must leave the dir entirely alone instead of
    /// helpfully overwriting the build the developer wanted to ear-check. (The case it really
    /// saves is a same-weight, *spec-only* build: small enough to land under the auto-apply
    /// limit, hence silently replaceable. A full package injection is over the limit, so it is
    /// only ever offered.)
    @Test func localBuildSentinelExemptsADir() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(712_370, to: dir.appendingPathComponent(cfmSpec))
        let manifest = [cfmSpec: 721_085]
        #expect(ModelRepository.staleFiles(dir: dir, manifest: manifest) == [cfmSpec])
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: manifest)
            == .update(files: [cfmSpec], bytes: 721_085))

        try Data().write(to: dir.appendingPathComponent(ModelRepository.localBuildSentinel))
        #expect(ModelRepository.staleFiles(dir: dir, manifest: manifest).isEmpty)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: manifest) == .load)
    }

    /// We fetch a glob **subset** of the repo, so a remote file we never asked for is
    /// simply absent locally. Absent must never read as stale — otherwise every install
    /// would re-download forever. (Absent is `HubApi`'s job: it fetches those.)
    @Test func absentLocalFileIsNotStale() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(100, to: dir.appendingPathComponent("tokenizer.json"))

        let remote = ["tokenizer.json": 100, "README.md": 4096, "onnx/model.onnx": 1_000_000]
        #expect(ModelRepository.staleFiles(dir: dir, manifest: remote).isEmpty)
    }

    /// The API's tree lists directories too, with `size: 0`. `remoteManifest` filters
    /// them out — but if one ever leaked through, sizing a *directory* (~96 B on APFS)
    /// against 0 would condemn every snapshot on every device. `regularFileSize` only
    /// answers for regular files, so a stray directory entry stays a no-op.
    @Test func directoryEntryIsNeverStale() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("S3CFM.mlpackage"), withIntermediateDirectories: true)

        #expect(ModelRepository.staleFiles(dir: dir, manifest: ["S3CFM.mlpackage": 0]).isEmpty)
    }

    /// The invariant. Each case is a way we could fail to reach (or read) the Hub — none
    /// may condemn the model. (`remoteManifest` funnels every one of them — offline, timeout,
    /// 401, 404, 5xx, malformed JSON — into the same `nil`, so pinning `manifest: nil` here
    /// pins them all, with no network.)
    @Test func cannotTellIsNeverStale() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(712_370, to: dir.appendingPathComponent(cfmSpec))

        // Cannot tell → never stale, and never a plan that spends anything.
        #expect(ModelRepository.staleFiles(dir: dir, manifest: nil).isEmpty)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: nil) == .load)
        // An empty tree is a broken answer, not "the repo has no files".
        #expect(ModelRepository.staleFiles(dir: dir, manifest: [:]).isEmpty)
        // A directory we don't even have: nothing to be stale about.
        let missing = dir.appendingPathComponent("nope")
        #expect(ModelRepository.staleFiles(dir: missing, manifest: ["tokenizer.json": 1]).isEmpty)
    }

    /// A Python `huggingface_hub` cache snapshot — one of the layouts
    /// `existingModelDirectory` discovers — is a farm of symlinks into `blobs/`.
    /// `attributesOfItem` has lstat semantics and would report the *link's* length
    /// (~85 B), condemning a perfectly good snapshot. Sizing must follow the link.
    @Test func symlinkedSnapshotIsSizedThroughTheLink() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let blob = dir.appendingPathComponent("blobs/abc123")
        try write(721_085, to: blob)

        let link = dir.appendingPathComponent(cfmSpec)
        try FileManager.default.createDirectory(
            at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: blob)

        // The link's own length is nowhere near 721,085 — if we lstat'd, this would be
        // "stale" and we'd move a good file out of a good model.
        #expect(ModelRepository.staleFiles(dir: dir, manifest: [cfmSpec: 721_085]).isEmpty)
        #expect(ModelRepository.staleFiles(dir: dir, manifest: [cfmSpec: 712_370]) == [cfmSpec])
    }
}

/// Offline units for the launch-time decision — `ModelRepository.launchPlan`, the seam the
/// app's `tryAutoRun` is a thin shell over (a SwiftUI `View` method can't be unit-tested, so
/// the *decision* was factored out to here and the view just switches on it).
///
/// Two invariants live here, and both are the kind that fail silently in the field:
///   1. **A failed update never denies the user their model** (`fetchIsRequired`). Only the
///      missing-model plan may end a launch without loading.
///   2. **Nothing large is downloaded unasked** (the ``ModelRepository/autoUpdateByteLimit``
///      gate). Byte counts below are the real ones, measured against the live repos.
struct LaunchPlanTests {
    /// The only plan that may abandon the launch on a download failure is the one with
    /// nothing on disk to fall back to. A device holding a complete-but-stale snapshot, on a
    /// flaky network / rate-limited / out of disk, must still end up with its model **loaded**.
    ///
    /// (The regression this pins is invisible: the app just quietly stops loading when the Hub
    /// is unreachable. `tryAutoRun` reads exactly this property to decide whether to `return`.)
    @Test func aFailedUpdateNeverDeniesTheUserTheirModel() throws {
        #expect(ModelRepository.LaunchPlan.download.fetchIsRequired)

        // Every plan that has a usable snapshot on disk: a failed fetch falls through to load.
        #expect(!ModelRepository.LaunchPlan.load.fetchIsRequired)
        #expect(!ModelRepository.LaunchPlan.update(files: [cfmSpec], bytes: 721_085).fetchIsRequired)
        #expect(!ModelRepository.LaunchPlan.offerUpdate(
            files: [t3lmWeight], bytes: 119_997_254).fetchIsRequired)
    }

    /// A missing model outranks everything — no Hub lookup can change the answer.
    @Test func noModelOnDiskAlwaysDownloads() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelRepository.launchPlan(
            modelOnDisk: false, dir: dir, manifest: [cfmSpec: 721_085]) == .download)
        #expect(ModelRepository.launchPlan(modelOnDisk: false, dir: dir, manifest: nil) == .download)
    }

    /// A snapshot that is current — or that we simply can't check — is just loaded.
    @Test func currentOrUnknowableSnapshotJustLoads() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(721_085, to: dir.appendingPathComponent(cfmSpec))

        #expect(ModelRepository.launchPlan(
            modelOnDisk: true, dir: dir, manifest: [cfmSpec: 721_085]) == .load)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: nil) == .load)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: [:]) == .load)
    }

    /// The reporter's device: a pre-`f4af6b3` `S3CFM` spec, an 8,715-byte content change that
    /// costs 721,085 B to fetch. Far under the limit → applied silently at launch, exactly as
    /// before the gate existed. This is the whole point of the feature, so the gate must not
    /// break it.
    @Test func aSpecFixIsSmallEnoughToApplySilently() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(712_370, to: dir.appendingPathComponent(cfmSpec))

        let manifest = [cfmSpec: 721_085]
        #expect(721_085 < ModelRepository.autoUpdateByteLimit)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: manifest)
            == .update(files: [cfmSpec], bytes: 721_085))
    }

    /// The negative control — and it is live TODAY. Both real repos were legitimately
    /// republished, so with an ungated auto-update every existing install would silently spend,
    /// on its very next launch (possibly on cellular):
    ///   - nano:         120 MB (`T3LM/weight.bin`, 211,678,982 → 119,997,254)
    ///   - multilingual: 537 MB (1,027,777,988 → 537,254,468)
    /// Both must be **offered**, not taken. The user's model still loads; the Download button
    /// (no size gate — that tap is the consent) applies it.
    ///
    /// Sizes are the ones measured against the live repos on 2026-07-12; `HubManifestTests`
    /// re-checks them against the Hub itself, so this offline test can't quietly rot.
    @Test func aLargeWeightRepublishIsOfferedNotTaken() throws {
        for (localBytes, remoteBytes) in [(211_678_982, 119_997_254),      // nano
                                          (1_027_777_988, 537_254_468)] {  // multilingual
            let dir = try makeDir()
            defer { try? FileManager.default.removeItem(at: dir) }
            try writeSparse(localBytes, to: dir.appendingPathComponent(t3lmWeight))

            let manifest = [t3lmWeight: remoteBytes]
            let plan = ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: manifest)
            #expect(remoteBytes > ModelRepository.autoUpdateByteLimit)
            #expect(plan == .offerUpdate(files: [t3lmWeight], bytes: remoteBytes))
            // The cost is the REMOTE size — the bytes that actually cross the network — not the
            // (here larger) local one, and not the delta.
            #expect(plan.bytes == remoteBytes)
        }
    }

    /// A plan's `bytes` totals **only managed, mismatched, locally-present** files — the same
    /// three filters `staleFiles` applies, for the same reasons. A number that over-counts would
    /// push a legitimate small fix over the gate and strand the user on a stale model; one that
    /// under-counts would spend their data unasked.
    @Test func planBytesTotalOnlyManagedMismatchedFiles() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(712_370, to: dir.appendingPathComponent(cfmSpec))       // mismatched → counts
        try write(3_561_468, to: dir.appendingPathComponent("tokenizer.json"))  // matches → 0
        try write(4_096, to: dir.appendingPathComponent("README.md"))     // unmanaged → 0

        let manifest = [
            cfmSpec: 721_085,
            "tokenizer.json": 3_561_468,
            "README.md": 9_999,             // a model-card commit must not cost a byte
            "onnx/model.onnx": 1_000_000,   // unmanaged AND absent locally
        ]
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: manifest)
            == .update(files: [cfmSpec], bytes: 721_085))
        // Cannot tell → 0, never a spend.
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: nil).bytes == 0)
        #expect(ModelRepository.launchPlan(modelOnDisk: true, dir: dir, manifest: [:]).bytes == 0)
    }

    /// The gate is a `<=` on the *total*, not a per-file rule: several small files that add up
    /// past the limit are offered, not taken.
    @Test func theGateIsOnTheTotalAndTheLimitItselfApplies() throws {
        let dir = try makeDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try write(1, to: dir.appendingPathComponent(cfmSpec))
        try write(1, to: dir.appendingPathComponent("tokenizer.json"))

        let limit = ModelRepository.autoUpdateByteLimit
        // Exactly at the limit: applied (the boundary belongs to the cheap side).
        #expect(ModelRepository.launchPlan(
            modelOnDisk: true, dir: dir,
            manifest: [cfmSpec: limit - 10, "tokenizer.json": 10])
            == .update(files: [cfmSpec, "tokenizer.json"], bytes: limit))
        // One byte over: offered.
        #expect(ModelRepository.launchPlan(
            modelOnDisk: true, dir: dir,
            manifest: [cfmSpec: limit - 10, "tokenizer.json": 11])
            == .offerUpdate(files: [cfmSpec, "tokenizer.json"], bytes: limit + 1))
    }

    /// Staleness is judged against the *discovered* `modelPath`, but `download` always writes to
    /// `<base>/models/<repoId>`. For a Python-`hf_hub` cache snapshot or a `--local-dir` clone
    /// those are different directories — so "repairing" one would fetch a fresh ~1 GB snapshot
    /// **somewhere else**, leave the original untouched, and load the stale model anyway. A
    /// foreign layout is not a snapshot we own: the app doesn't version-check it, and
    /// `removeDownload` won't delete it.
    @Test func onlyTheDirectoryWeDownloadIntoIsOurs() throws {
        let hfHome = try makeDir("hfhome")
        defer { try? FileManager.default.removeItem(at: hfHome) }
        let base = ModelRepository.base(forHFHome: hfHome)
        let repoId = ModelRepository.defaultNanoRepoId

        let ours = ModelRepository.downloadDirectory(hfHome: hfHome, repoId: repoId)
        #expect(ours == base.appending(component: "models").appending(path: repoId))
        #expect(ModelRepository.ownsSnapshot(at: ours, hfHome: hfHome, repoId: repoId))
        // Same dir, non-standard spelling (and, on macOS, /var → /private/var).
        #expect(ModelRepository.ownsSnapshot(
            at: ours.appending(path: "sub/.."), hfHome: hfHome, repoId: repoId))

        // The two layouts `existingModelDirectory` also discovers — neither is ours.
        #expect(!ModelRepository.ownsSnapshot(
            at: base.appending(path: repoId), hfHome: hfHome, repoId: repoId))  // --local-dir
        #expect(!ModelRepository.ownsSnapshot(
            at: base.appending(path: "models--iliasaz--chatterbox-nano-coreml/snapshots/abc123"),
            hfHome: hfHome, repoId: repoId))  // python hf_hub cache
        // …and not another repo's dir under the same base.
        #expect(!ModelRepository.ownsSnapshot(
            at: ours, hfHome: hfHome, repoId: ModelRepository.defaultRepoId))
    }

    /// `removeDownload` (the wipe behind "Force re-download") deletes the dir we **own**, not
    /// whatever `existingModelDirectory` happens to discover. Pointed at the discovered dir it
    /// would delete a user's `--local-dir` clone or their Python `hf_hub` cache — and then
    /// re-fetch into a *different* directory, so the deletion wouldn't even buy them the repair.
    @Test func removeDownloadNeverDeletesAForeignLayout() throws {
        let fm = FileManager.default
        let hfHome = try makeDir("hfhome")
        defer { try? fm.removeItem(at: hfHome) }
        let base = ModelRepository.base(forHFHome: hfHome)
        let repoId = ModelRepository.defaultNanoRepoId

        // A `--local-dir` clone: has the marker file, so `existingModelDirectory` finds it…
        let clone = base.appending(path: repoId)
        try write(1_024, to: clone.appendingPathComponent("default-conds.safetensors"))
        #expect(ModelRepository.existingModelDirectory(hfHome: hfHome, repoId: repoId) == clone)

        // …but it is not ours, so there is nothing for us to remove, and it survives.
        #expect(try ModelRepository.removeDownload(hfHome: hfHome, repoId: repoId) == false)
        #expect(fm.fileExists(atPath: clone.path))

        // Our own snapshot, by contrast, is removed.
        let ours = ModelRepository.downloadDirectory(hfHome: hfHome, repoId: repoId)
        try write(1_024, to: ours.appendingPathComponent("default-conds.safetensors"))
        #expect(try ModelRepository.removeDownload(hfHome: hfHome, repoId: repoId) == true)
        #expect(!fm.fileExists(atPath: ours.path))
        #expect(fm.fileExists(atPath: clone.path))  // still untouched
    }
}

/// Offline units for what a fetch *actually did*, which is not what "it didn't throw" says.
/// `HubApi.snapshot` returns success without downloading anything when it is offline
/// (`HubApi.swift:902`) or cancelled (`:966`), and applies a commit one file at a time.
struct DownloadResultTests {
    /// The distinction the whole recovery hinges on. Both of these have a non-empty
    /// `stillStale`, and treating them the same is a real bug in **either** direction:
    ///   - Force-wiping the *offline no-op* would delete the user's only copy of the model on
    ///     the one launch we already know cannot download a new one.
    ///   - Loading the *half-applied* one hands a new graph spec + an old weight to the ANE
    ///     compiler, which is the crash this entire fix exists to prevent.
    @Test func onlyAHalfAppliedFetchIsPartiallyApplied() throws {
        let dir = URL(fileURLWithPath: "/tmp/x")

        // Offline / cancelled before the first file: nothing landed. The snapshot is byte-for-
        // byte the working one it always was — still loadable, still stale, re-checked next
        // launch. NOT a reason to destroy it.
        let noOp = ModelRepository.DownloadResult(
            directory: dir, repaired: [], stillStale: [cfmSpec, t3lmWeight],
            checkedAgainstHub: true)
        #expect(!noOp.isPartiallyApplied)

        // Cancelled *between* two files of one commit: a new spec beside an old weight. Nothing
        // else catches this — the old weight is complete, so `incompleteMLPackage` is happy.
        // (Or: the Hub moved between our manifest lookup and our fetch, and the snapshot is in
        // fact current. The flag cannot tell those apart — which is exactly why it may not
        // authorise a wipe. See `PartialApplyRepairTests`.)
        let mixed = ModelRepository.DownloadResult(
            directory: dir, repaired: [cfmSpec], stillStale: [t3lmWeight],
            checkedAgainstHub: true)
        #expect(mixed.isPartiallyApplied)

        // A clean update, and a no-op on an already-current snapshot.
        #expect(!ModelRepository.DownloadResult(
            directory: dir, repaired: [cfmSpec], stillStale: [], checkedAgainstHub: true)
            .isPartiallyApplied)
        #expect(!ModelRepository.DownloadResult(
            directory: dir, repaired: [], stillStale: [], checkedAgainstHub: true)
            .isPartiallyApplied)
    }
}

/// Offline units for the one automatic decision that could cost a user their model:
/// ``ModelRepository/repairDecision(afterRetry:)`` — what autorun does once a partially-applied
/// fetch has been re-fetched plain.
///
/// This is the seam `ContentView.tryAutoRun` switches on (a SwiftUI `View` method cannot be
/// unit-tested, so the *decision* lives in `ModelRepository`, pure, and the view just calls it).
///
/// The invariant, and it is the one the whole design rests on: **we never destroy a model we
/// cannot prove is broken.** An earlier draft of this branch escalated a failed retry to
/// `autoDownload(force: true)` — a real `FileManager.removeItem` of the whole snapshot, with no
/// user tap — on evidence that proves nothing: a retry that *threw* (offline / 429 / expired
/// token / ENOSPC) says nothing whatever about the bytes on disk, and `isPartiallyApplied`
/// itself only means "disagrees with the manifest you were handed", which a snapshot that is
/// FULLY CURRENT at `main` also does when a commit lands between the lookup and the fetch.
struct PartialApplyRepairTests {
    private static let dir = URL(fileURLWithPath: "/tmp/x")

    private static func result(
        repaired: [String] = [], stillStale: [String] = [], checkedAgainstHub: Bool = true
    ) -> ModelRepository.DownloadResult {
        .init(directory: dir, repaired: repaired, stillStale: stillStale,
              checkedAgainstHub: checkedAgainstHub)
    }

    /// Only positive proof loads: the retry succeeded, it had **current** Hub truth to check
    /// against, and nothing is behind it.
    @Test func aVerifiedCleanRetryLoads() throws {
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result())) == .load)
        // The manifest we were handed was merely one commit old: the re-fetch, which re-reads
        // the Hub, finds the snapshot already current and repairs nothing. Load it — the old
        // code force-WIPED this perfectly good model.
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result(repaired: [cfmSpec]))) == .load)
    }

    /// **The regression this branch exists to prevent.** None of these is proof of a broken
    /// snapshot, so none of them may lead anywhere near a wipe — the decision is `askUser`, the
    /// snapshot stays on disk untouched, and the user's Download tap (or the next launch's
    /// re-check) repairs it.
    @Test func aFailedOrUnprovableRetryNeverDestroysTheSnapshot() throws {
        // (a) The retry THREW. Offline, rate-limited, token expired, disk full — every one of
        //     them is a statement about the network or the disk, not about the model.
        for error in [URLError(.notConnectedToInternet) as Error,
                      URLError(.timedOut),
                      NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))] {
            #expect(ModelRepository.repairDecision(afterRetry: .failure(error)) == .askUser)
        }

        // (b) The retry succeeded but the Hub STILL disagrees — the mix survived, or a fresh
        //     commit landed. Distrust it: don't load. But it still loads on the user's machine
        //     today, so we do not get to delete it either.
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result(repaired: [cfmSpec], stillStale: [t3lmWeight])))
            == .askUser)
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result(stillStale: [t3lmWeight]))) == .askUser)

        // (c) "Cannot tell": the retry's own manifest lookup came back nil (offline / no token /
        //     5xx), so `stillStale` is empty for LACK OF EVIDENCE, not because anything was
        //     verified. An empty list must not be mistaken for a clean bill of health.
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result(checkedAgainstHub: false))) == .askUser)
        #expect(ModelRepository.repairDecision(
            afterRetry: .success(Self.result(repaired: [cfmSpec], checkedAgainstHub: false)))
            == .askUser)
    }

    /// The strongest form of the invariant, and the reason `RepairDecision` is an enum at all:
    /// **there is no destructive outcome to reach.** A third case (`.forceRedownload`) is what
    /// the reviewer found here, past the 25 MB gate and with no user tap. If one is ever added,
    /// this fails.
    @Test func thereIsNoDestructiveDecisionToReach() throws {
        #expect(ModelRepository.RepairDecision.allCases == [.load, .askUser])
    }
}

/// Offline units for token resolution. `HubApi` finds the CLI token files itself, but a
/// raw `URLSession` call to the HF API cannot — so without this fallback the staleness
/// check silently 401s on an `hf auth login`-only machine (which is the dev machine) and
/// the whole feature is inert exactly where it is written.
struct ResolvedTokenTests {
    @Test func prefersExplicitThenEnvThenFiles() throws {
        let home = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tok-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: home) }
        let hfHome = home.appending(path: "HF_HOME")
        try FileManager.default.createDirectory(at: hfHome, withIntermediateDirectories: true)
        try Data(" hf_from_file\n".utf8).write(to: hfHome.appending(component: "token"))
        let env = ["HF_HOME": hfHome.path]

        // Explicit beats everything; env beats the files; the files are the last resort
        // (and get trimmed — a trailing newline in a Bearer header is a 401).
        #expect(ModelRepository.resolvedToken("hf_explicit", env: env, home: home) == "hf_explicit")
        #expect(ModelRepository.resolvedToken(
            nil, env: env.merging(["HF_TOKEN": "hf_env"]) { _, b in b }, home: home) == "hf_env")
        #expect(ModelRepository.resolvedToken(nil, env: env, home: home) == "hf_from_file")
    }

    @Test func readsHFTokenPathAndTheHomeCacheFiles() throws {
        let home = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tok-\(UUID().uuidString)")
        let cache = home.appending(path: ".cache/huggingface")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: home) }
        try Data("hf_from_cache".utf8).write(to: cache.appending(component: "token"))

        #expect(ModelRepository.resolvedToken(nil, env: [:], home: home) == "hf_from_cache")

        let explicitPath = home.appending(component: "custom-token")
        try Data("hf_from_path".utf8).write(to: explicitPath)
        #expect(ModelRepository.resolvedToken(
            nil, env: ["HF_TOKEN_PATH": explicitPath.path], home: home) == "hf_from_path")
    }

    /// A blank token must resolve to nil, not to an empty `Bearer ` header.
    @Test func blankIsNil() throws {
        let home = URL(fileURLWithPath: NSTemporaryDirectory())
        #expect(ModelRepository.resolvedToken("   ", env: ["HF_TOKEN": "\n"], home: home) == nil)
    }
}

/// Offline units for ``ModelRepository/isOutOfSpace(_:)``.
struct OutOfSpaceTests {
    /// A full disk gets its own message — its remedy differs from a network failure's, and
    /// the generic text sends the user hunting a problem they don't have. `URLSession` and
    /// Foundation each throw it in a different shape, and one of them buries it.
    @Test func outOfSpaceIsRecognisedThroughAnUnderlyingError() throws {
        let enospc = NSError(domain: NSPOSIXErrorDomain, code: Int(ENOSPC))
        let cocoa = NSError(domain: NSCocoaErrorDomain, code: NSFileWriteOutOfSpaceError)
        let wrapped = NSError(
            domain: NSCocoaErrorDomain, code: NSFileWriteUnknownError,
            userInfo: [NSUnderlyingErrorKey: enospc])

        #expect(ModelRepository.isOutOfSpace(enospc))
        #expect(ModelRepository.isOutOfSpace(cocoa))
        #expect(ModelRepository.isOutOfSpace(wrapped))
        #expect(!ModelRepository.isOutOfSpace(URLError(.notConnectedToInternet)))
    }
}
