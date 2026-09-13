import Testing
import Foundation
import Hub
@testable import ChatterboxCoreML

/// Live network test (opt-in) answering: does `HubApi.snapshot` fetch a
/// **newly-added** repo file into an *existing* cache, or does it consider the
/// snapshot already-complete and skip it?
///
/// This is the exact question behind "the iOS app downloaded but didn't get
/// `T3LM`". Run with auth + the flag:
///
///   CHATTERBOX_HUB_TEST=1 \
///     swift test --filter HubDownloadTests
///
/// Skipped unless `CHATTERBOX_HUB_TEST` is set (needs network; a token, if one is
/// set, resolves via `HF_TOKEN` or `$HF_HOME/token`).
struct HubDownloadTests {
    @Test func snapshotFetchesNewlyMatchedFileIntoExistingCache() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hubtest-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: tmp) }
        let base = tmp.appendingPathComponent("hub")

        let token = ProcessInfo.processInfo.environment["HF_TOKEN"]
        let api = HubApi(downloadBase: base, hfToken: token)
        let repo = Hub.Repo(id: ModelRepository.defaultRepoId)

        // Step 1 — populate the cache WITHOUT the model file (simulates a prior
        // download from an older app build whose globs didn't include T3LM).
        let dir1 = try await api.snapshot(from: repo, matching: ["added_tokens.json"])
        let t3lmManifest = "T3LM.mlpackage/Manifest.json"
        let before = fm.fileExists(atPath: dir1.appendingPathComponent(t3lmManifest).path)
        #expect(before == false)  // not fetched yet — only added_tokens.json was requested

        // Step 2 — re-run snapshot now that the patterns include the newly-added
        // model file. (Match only Manifest.json, not the 600 MB weight.bin, so the
        // test stays cheap — the per-file fetch mechanism is identical.)
        let dir2 = try await api.snapshot(
            from: repo, matching: ["added_tokens.json", t3lmManifest])
        let after = fm.fileExists(atPath: dir2.appendingPathComponent(t3lmManifest).path)

        // The answer to the user's question: does an existing cache pick up a
        // newly-matched file on a second snapshot call?
        #expect(after == true)
        #expect(dir1.path == dir2.path)  // same snapshot dir reused, not a fresh one
    }

    /// Reproduces the iPhone failure: an existing cache (old file set, no model),
    /// then an incremental download that must pull the **full** T3LM — including
    /// the ~606 MB LFS `weight.bin`. Asserts the weight file is complete, since a
    /// partial/missing weight compiles to a broken model (`ANECCompile FAILED`).
    /// Downloads ~600 MB; opt-in via CHATTERBOX_HUB_TEST_FULL.
    @Test func incrementalUpdatePullsCompleteWeightBin() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST_FULL"] != nil else { return }
        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hubfull-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: tmp) }
        let api = HubApi(downloadBase: tmp.appendingPathComponent("hub"),
                         hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"])
        let repo = Hub.Repo(id: ModelRepository.defaultRepoId)

        // 1) Prior cache WITHOUT the model (old app build's file set).
        _ = try await api.snapshot(from: repo, matching: ["added_tokens.json", "tokenizer.json"])
        // 2) Incremental download now that globs include the model package.
        let dir = try await api.snapshot(
            from: repo, matching: ["added_tokens.json", "tokenizer.json",
                                   "T3LM.mlpackage/*", "T3LM.mlpackage/**/*"])
        let weight = dir.appendingPathComponent("T3LM.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        let size = (try? fm.attributesOfItem(atPath: weight.path)[.size] as? Int) ?? nil
        #expect(size == 635_929_926)  // full weight.bin, not a partial/LFS-pointer
    }

    /// Verifies `download`'s self-heal: a truncated `weight.bin` (the iPhone
    /// "downloaded but won't load" state) is detected and repaired by a clean
    /// re-fetch, so Load never sees a broken model. Downloads ~600 MB twice;
    /// opt-in via CHATTERBOX_HUB_TEST_FULL.
    @Test func downloadSelfHealsTruncatedWeight() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST_FULL"] != nil else { return }
        let fm = FileManager.default
        let hfHome = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("selfheal-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: hfHome) }

        // Full clean download, then corrupt T3LM's weight to simulate a partial
        // incremental update.
        let dir = try await ModelRepository.download(hfHome: hfHome).directory
        let weight = dir.appendingPathComponent("T3LM.mlpackage/Data/com.apple.CoreML/weights/weight.bin")
        try Data(count: 200).write(to: weight)  // truncate to a "pointer-sized" stub
        #expect(ModelRepository.incompleteMLPackage(in: dir) == "T3LM.mlpackage")

        // Re-download: must notice the broken weight and re-fetch from clean.
        let dir2 = try await ModelRepository.download(hfHome: hfHome).directory
        #expect(ModelRepository.incompleteMLPackage(in: dir2) == nil)
        let size = ((try? fm.attributesOfItem(atPath: weight.path))?[.size] as? Int) ?? 0
        #expect(size == 635_929_926)
    }
}

/// Live network tests (opt-in) for `HubApi.snapshot`'s refresh policy — the two halves of
/// it, which are NOT the same and were once conflated into a wrong premise that drove a
/// whole round of unnecessary machinery.
///
/// `HubFileDownloader.download` (`HubApi.swift:786-820`) decides in this order:
///   1. local file + local `.metadata` + **`localCommitHash == remoteCommitHash`** → return
///      the local file, no content check at all;
///   2. otherwise compare the stored etag against the remote one (and, for LFS, the local
///      file's actual SHA-256) → **re-download when they differ.**
///
/// So corrupting a file *locally* while the remote is unchanged hits (1) and is NOT
/// repaired — `doesNotRepairLocallyCorruptedFile`. A file the remote *changed* hits (2) and IS
/// repaired — `repairsARemotelyChangedFile`. Only the second is the shipped bug's shape.
///
///   CHATTERBOX_HUB_TEST=1 \
///     swift test --filter HubStaleFileTests
struct HubStaleFileTests {
    /// **Measures exactly one thing: a file corrupted LOCALLY, with the REMOTE unchanged,
    /// is not re-fetched.** The commit hash beside it still matches the Hub's, so
    /// `snapshot` short-circuits before it ever looks at content (step 1 above).
    ///
    /// - Warning: This does **NOT** generalise to a remotely-changed file, and reading it
    ///   that way ("`snapshot` keys on existence, not content") is what produced the
    ///   fictional problem that `repairsARemotelyChangedFile` disproves. Local corruption
    ///   and an upstream commit are different inputs and get different behaviour. The only
    ///   thing pinned here: **we cannot repair local bit-rot by re-running a download** —
    ///   that needs an explicit wipe (`download(force:)`), which is what the app's
    ///   "Force re-download" is for.
    ///
    /// (Named for what it *pins*, not for what it hopes: the old name —
    /// `refreshesLocallyStaleFile` — asserted the opposite of its own `#expect(!refreshed)`,
    /// and misreading it cost two rounds of design.)
    @Test func doesNotRepairLocallyCorruptedFile() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let fm = FileManager.default
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hubstale-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: tmp) }
        let base = tmp.appendingPathComponent("hub")

        let api = HubApi(downloadBase: base, hfToken: ModelRepository.resolvedToken())
        let repo = Hub.Repo(id: ModelRepository.defaultNanoRepoId)
        // One small file is enough to characterise the refresh policy.
        let globs = ["tokenizer_config.json"]

        // 1. Full snapshot.
        let dir = try await api.snapshot(from: repo, matching: globs) { _ in }
        let file = dir.appendingPathComponent("tokenizer_config.json")
        let pristine = try Data(contentsOf: file)
        #expect(!pristine.isEmpty)

        // 2. Corrupt the LOCAL copy. The Hub's copy — and therefore the commit hash and the
        //    etag stored in `.cache/huggingface/download/…metadata` — is untouched.
        try Data("{\"stale\": true}".utf8).write(to: file)
        let staleSize = try Data(contentsOf: file).count
        #expect(staleSize != pristine.count)

        // 3. Re-snapshot into the same base.
        _ = try await api.snapshot(from: repo, matching: globs) { _ in }
        let after = try Data(contentsOf: file)

        let refreshed = (after == pristine)
        print("HUBSTALE: refreshed=\(refreshed) staleSize=\(staleSize) afterSize=\(after.count) pristineSize=\(pristine.count)")
        // The measured policy (2026-07-12): local commit hash == remote commit hash ⇒ the
        // local bytes are returned unexamined. Re-downloading cannot undo local corruption.
        #expect(!refreshed, "HubApi.snapshot now re-validates local content — the force-wipe path could be simplified")
        #expect(after == Data("{\"stale\": true}".utf8))
    }

    /// **The load-bearing test of the whole fix.** A file whose content changed *upstream*
    /// IS repaired by a plain `snapshot` — no wipe, no move-aside, no reconcile machinery —
    /// and the unchanged 140 MB sibling next to it is not re-downloaded.
    ///
    /// Reproduces the reporter's device exactly: stage the snapshot at `1bd603a46e8d` (the
    /// commit before the S3CFM `def1024` fix), so the on-disk `S3CFM` spec is the 712,370-byte
    /// build that SIGSEGV'd `bnns::GraphCompile` on the ANE, with genuine HubApi `.metadata`
    /// sidecars — then run the app's own `download()` against `main`.
    ///
    /// If this ever fails, the fix's premise is gone and the whole update path has to be
    /// rethought (that premise's *inverse* was believed for two rounds, on the strength of
    /// `doesNotRepairLocallyCorruptedFile` above, and it is wrong).
    @Test func repairsARemotelyChangedFile() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let fm = FileManager.default
        let hfHome = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hubrev-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: hfHome) }

        let repoId = ModelRepository.defaultRepoId
        let repo = Hub.Repo(id: repoId)
        // S3CFM alone: the changed spec (~712 KB) plus its unchanged weight (~147 MB), which
        // is the "not re-downloaded" evidence. Same package, same fetch, same etag rules as
        // the 600 MB T3LM would be.
        let globs = ["S3CFM.mlpackage/*", "S3CFM.mlpackage/**/*"]
        let spec = "S3CFM.mlpackage/Data/com.apple.CoreML/model.mlmodel"
        let weight = "S3CFM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"

        // 1. Be the reporter's phone: a real HubApi snapshot pinned to the pre-fix commit.
        let api = HubApi(downloadBase: ModelRepository.base(forHFHome: hfHome),
                         hfToken: ModelRepository.resolvedToken())
        let dir = try await api.snapshot(from: repo, revision: "1bd603a46e8d", matching: globs) { _ in }

        func attrs(_ path: String) throws -> [FileAttributeKey: Any] {
            try fm.attributesOfItem(atPath: dir.appendingPathComponent(path).path)
        }
        #expect(try attrs(spec)[.size] as? Int == 712_370)  // the graph that crashed the ANE
        let weightBefore = try attrs(weight)
        #expect(weightBefore[.size] as? Int == 146_892_256)
        let weightInode = weightBefore[.systemFileNumber] as? Int
        let weightMtime = weightBefore[.modificationDate] as? Date

        // 2. The check that gives `download()` a reason to run at all — and the size gate, which
        //    must NOT get in the way of the very repair this exists for: an 8.7 KB spec change
        //    costs 721,085 B to fetch, far under the 25 MB auto-apply limit, so the reporter's
        //    device still fixes itself silently at launch with no prompt.
        let manifest = try #require(await ModelRepository.remoteManifest(repoId: repoId))
        #expect(ModelRepository.staleFiles(dir: dir, manifest: manifest, matching: globs) == [spec])
        #expect(ModelRepository.launchPlan(
            modelOnDisk: true, dir: dir, manifest: manifest, matching: globs)
            == .update(files: [spec], bytes: 721_085))

        // 3. Just download. No invalidation of any kind.
        let result = try await ModelRepository.download(
            repoId: repoId, hfHome: hfHome, matching: globs, manifest: manifest)
        let dir2 = result.directory
        #expect(dir2.path == dir.path)

        // HubApi repaired the changed file by itself (remote etag ≠ stored etag)…
        #expect(try attrs(spec)[.size] as? Int == 721_085)
        #expect(ModelRepository.staleFiles(dir: dir2, manifest: manifest, matching: globs).isEmpty,
                "snapshot still behind the Hub")
        // …and said so: a fully-applied update, nothing left over, nothing to recover from.
        #expect(result.repaired == [spec])
        #expect(result.stillStale.isEmpty)
        #expect(!result.isPartiallyApplied)

        // …and left the 147 MB it did not need to touch alone: same inode, same mtime. (Which is
        // also the proof that nothing was wiped and re-fetched to achieve the repair.)
        #expect(try attrs(weight)[.systemFileNumber] as? Int == weightInode,
                "the unchanged weight was re-downloaded — the etag check is not doing its job")
        #expect(try attrs(weight)[.modificationDate] as? Date == weightMtime)
    }

    /// End-to-end: a fetch that lands *some* of the manifest it was fetched against and not the
    /// rest must **say so** (`isPartiallyApplied`) instead of silently handing the caller a
    /// snapshot that may be a NEW `model.mlmodel` beside an OLD `weight.bin`. Nothing else
    /// catches that shape: the old weight is complete, so `incompleteMLPackage` is perfectly
    /// happy with it, and the combination is exactly what SIGSEGV'd `bnns::GraphCompile` on the
    /// ANE.
    ///
    /// What the flag means is precisely "the snapshot disagrees with the manifest supplied" —
    /// **evidence of** a two-revision mix, not **proof** of one (an innocent commit landing
    /// between the caller's manifest lookup and its fetch raises it too). Hence the caller's
    /// only licensed response is to re-check against current Hub truth, never to destroy: see
    /// `ModelRepository.repairDecision(afterRetry:)` / `PartialApplyRepairTests`.
    ///
    /// How the half-application is produced here: stage the pre-fix revision (so the spec really
    /// is stale and really does get repaired), and hand `download` a manifest that *also* claims
    /// the weight is a different size. `HubApi` won't touch the weight — its etag is unchanged,
    /// which is precisely the case of a file the fetch does not land — so the post-fetch check
    /// sees one file repaired and one still off. Same bookkeeping, same code path, same result
    /// as a real cancellation between two files, without having to win a race with the network.
    @Test func aHalfAppliedFetchIsReportedNotSwallowed() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let fm = FileManager.default
        let hfHome = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("hubpartial-\(UUID().uuidString)")
        defer { try? fm.removeItem(at: hfHome) }

        let repoId = ModelRepository.defaultRepoId
        let globs = ["S3CFM.mlpackage/*", "S3CFM.mlpackage/**/*"]
        let spec = "S3CFM.mlpackage/Data/com.apple.CoreML/model.mlmodel"
        let weight = "S3CFM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"

        let api = HubApi(downloadBase: ModelRepository.base(forHFHome: hfHome),
                         hfToken: ModelRepository.resolvedToken())
        let dir = try await api.snapshot(
            from: Hub.Repo(id: repoId), revision: "1bd603a46e8d", matching: globs) { _ in }

        // The real manifest, plus one file the fetch cannot possibly reconcile.
        var manifest = try #require(await ModelRepository.remoteManifest(repoId: repoId))
        manifest[weight] = 146_892_257  // one byte off the real 146,892,256
        #expect(ModelRepository.staleFiles(dir: dir, manifest: manifest, matching: globs)
            == [spec, weight])

        let result = try await ModelRepository.download(
            repoId: repoId, hfHome: hfHome, matching: globs, manifest: manifest)

        // The fetch really did repair the spec…
        #expect(try fm.attributesOfItem(
            atPath: dir.appendingPathComponent(spec).path)[.size] as? Int == 721_085)
        #expect(result.repaired == [spec])
        // …and really did leave the other file behind — which is surfaced, not swallowed. The
        // download itself still succeeded (it must NOT become a hard failure of an otherwise-good
        // fetch); the caller reads this and retries a PLAIN update, which deletes nothing.
        #expect(result.stillStale == [weight])
        #expect(result.isPartiallyApplied)
        #expect(result.directory.path == dir.path)  // and nothing was moved or deleted
    }
}

/// Live network tests (opt-in, `CHATTERBOX_HUB_TEST=1`) for the Hub manifest lookup — the
/// cheap size comparison that tells the app its snapshot is behind the Hub and that it
/// should therefore call `download()` at all.
///
/// Deliberately run **without** `HF_TOKEN`, so they also prove the token-FILE fallback
/// (`hf auth login` → `$HF_HOME/token`): against a *private* repo, without it
/// `remoteManifest` 401s, returns nil, and the whole feature is silently inert.
///
///   CHATTERBOX_HUB_TEST=1 \
///     swift test --filter HubManifestTests
struct HubManifestTests {
    /// The footgun that would wipe every device: if `remoteManifest` reported an LFS
    /// file's **pointer** length (~131–134 B) instead of its real size, every weight
    /// would look stale on every launch and every install would re-download forever.
    ///
    /// (The HF API in fact reports the real size in BOTH `size` and `lfs.size`, with the
    /// pointer length in a separate `pointerSize` field — verified here rather than
    /// assumed, because getting it backwards is unrecoverable in the field.)
    @Test func manifestReportsRealLFSSizesNotPointerSizes() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let repoId = ModelRepository.defaultNanoRepoId
        let manifest = try #require(
            await ModelRepository.remoteManifest(repoId: repoId),
            "no manifest for \(repoId) — network, or the token-file fallback is broken")

        // The big LFS weights, at their real sizes. A pointer is ~130 bytes.
        let weight = try #require(manifest["T3LM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"])
        #expect(weight > 100_000_000, "LFS weight reported as \(weight) B — that's a pointer, not the blob")
        #expect(manifest["S3CFM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"] ?? 0 > 100_000_000)
        // No entry may be pointer-sized, and directories (`size: 0`) must not appear at
        // all — a 0-byte "file" compared against a real local directory reads as stale.
        #expect(manifest.values.allSatisfy { $0 > 0 })
        #expect(manifest["S3CFM.mlpackage"] == nil, "directory entries must be filtered out")
        // The spec file at the heart of the crash, at its fixed size.
        #expect(manifest["S3CFM.mlpackage/Data/com.apple.CoreML/model.mlmodel"] == 721_085)
    }

    /// The negative control, against the LIVE repos. Both nano and multilingual were
    /// legitimately republished with a smaller `T3LM/weight.bin`, so an ungated auto-update
    /// would, on the very next launch of every existing install, silently pull the whole new
    /// weight — on cellular, with no confirmation and no way to defer. This is what the
    /// `autoUpdateByteLimit` gate exists to stop, and this test measures it against the Hub
    /// rather than trusting the numbers hardcoded in `LaunchPlanTests`.
    ///
    /// Staged with a sparse file (no bytes are written or fetched): only `st_size` is ever read.
    @Test func aRepublishedWeightIsOfferedNotAutoDownloaded() async throws {
        guard ProcessInfo.processInfo.environment["CHATTERBOX_HUB_TEST"] != nil else { return }

        let fm = FileManager.default
        let weightPath = "T3LM.mlpackage/Data/com.apple.CoreML/weights/weight.bin"
        // The size each variant's snapshot holds TODAY on a device installed before the
        // republish — i.e. what the reviewers' negative control found in the field.
        let installed: [(ModelRepository.Variant, Int, Int)] = [
            (.nano, 211_678_982, 119_997_254),
            (.multilingual, 1_027_777_988, 537_254_468),
        ]

        for (variant, localSize, expectedRemote) in installed {
            let dir = URL(fileURLWithPath: NSTemporaryDirectory())
                .appendingPathComponent("republish-\(UUID().uuidString)")
            defer { try? fm.removeItem(at: dir) }
            let weight = dir.appendingPathComponent(weightPath)
            try fm.createDirectory(at: weight.deletingLastPathComponent(),
                                   withIntermediateDirectories: true)
            fm.createFile(atPath: weight.path, contents: nil)
            let handle = try FileHandle(forWritingTo: weight)
            try handle.truncate(atOffset: UInt64(localSize))  // sparse: no bytes on disk
            try handle.close()

            let manifest = try #require(await ModelRepository.remoteManifest(repoId: variant.repoId))
            #expect(manifest[weightPath] == expectedRemote,
                    "\(variant.rawValue): the Hub now has \(manifest[weightPath] ?? -1) B, not \(expectedRemote) — the auto-update gate's numbers need re-checking")

            let plan = ModelRepository.launchPlan(
                modelOnDisk: true, dir: dir, manifest: manifest, matching: variant.runtimeGlobs)
            print("REPUBLISH: \(variant.rawValue) local=\(localSize) remote=\(expectedRemote) plan=\(plan)")
            // Offered — NOT downloaded. The stale model still loads; Download applies the update.
            #expect(plan == .offerUpdate(files: [weightPath], bytes: expectedRemote))
            #expect(!plan.fetchIsRequired, "a launch must never be abandoned over an update")
        }
    }
}
