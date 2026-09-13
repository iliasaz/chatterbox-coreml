import Testing
import Foundation
import CoreML
@testable import ChatterboxCoreML

/// Unit coverage for the persistent compiled-model cache (issue #23):
/// `compiledModel(at:cacheDir:)` compiles a `.mlpackage` once and reuses the
/// `.mlmodelc` on later launches so the ~52 s ANE AOT recompile is not paid every
/// launch. These exercise the keying / warm-hit logic without a real CoreML model
/// (a real cold compile is covered end-to-end by `PipelineTests` with a model dir).
@Suite("compiledModel persistent cache (issue #23)")
struct CompiledModelCacheTests {
    private func tempDir() throws -> URL {
        let d = FileManager.default.temporaryDirectory
            .appendingPathComponent("cbx-cache-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    /// A directory that *looks* like a `.mlpackage` (right extension, holds files so
    /// the content stamp is non-trivial) but is NOT a loadable CoreML model — so any
    /// code path that actually calls `MLModel.compileModel` on it would throw.
    private func fakePackage(in dir: URL, name: String = "Fake", weightBytes: Int = 16) throws -> URL {
        let pkg = dir.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        let weights = pkg.appendingPathComponent("Data/com.apple.CoreML/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
        try Data(repeating: 0xab, count: weightBytes).write(to: weights.appendingPathComponent("weight.bin"))
        try Data("{}".utf8).write(to: pkg.appendingPathComponent("Manifest.json"))
        return pkg
    }

    @Test("a .mlmodelc input is returned unchanged (already compiled)")
    func mlmodelcPassthrough() async throws {
        let url = URL(fileURLWithPath: "/nonexistent/T3LM.mlmodelc")
        let out = try await ChatterboxCoreMLModel.compiledModel(at: url, cacheDir: nil)
        #expect(out == url)
    }

    @Test("content stamp is stable when unchanged and changes when a file changes")
    func versionStampInvalidates() throws {
        let dir = try tempDir(); defer { try? FileManager.default.removeItem(at: dir) }
        let pkg = try fakePackage(in: dir, weightBytes: 16)
        let s1 = ChatterboxCoreMLModel.packageVersionStamp(pkg)
        #expect(s1 == ChatterboxCoreMLModel.packageVersionStamp(pkg))   // deterministic
        // Re-export / re-download changes weight.bin → stamp must change so the
        // persisted compile invalidates.
        let wb = pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        try Data(repeating: 0xcd, count: 32).write(to: wb)
        #expect(ChatterboxCoreMLModel.packageVersionStamp(pkg) != s1)
    }

    @Test("cache key separates two model dirs holding same-named packages")
    func keyDiffersBySourcePath() throws {
        let dirA = try tempDir(); defer { try? FileManager.default.removeItem(at: dirA) }
        let dirB = try tempDir(); defer { try? FileManager.default.removeItem(at: dirB) }
        // Same name + identical content, different parent dirs (turbo vs multilingual).
        let a = try fakePackage(in: dirA, name: "T3LM", weightBytes: 16)
        let b = try fakePackage(in: dirB, name: "T3LM", weightBytes: 16)
        let ka = ChatterboxCoreMLModel.compiledModelCacheNames(for: a)
        let kb = ChatterboxCoreMLModel.compiledModelCacheNames(for: b)
        #expect(ka.key != kb.key)                   // distinct artifacts…
        #expect(ka.prunePrefix != kb.prunePrefix)   // …so pruning one never evicts the other
    }

    @Test("warm cache hit returns the persisted .mlmodelc without recompiling")
    func warmHitSkipsCompile() async throws {
        let root = try tempDir(); defer { try? FileManager.default.removeItem(at: root) }
        let cacheDir = root.appendingPathComponent("cache", isDirectory: true)
        let pkg = try fakePackage(in: root, name: "Warm")
        // Pre-seed the exact stable artifact a prior cold launch would have produced —
        // including the `coremldata.bin` completeness sentinel.
        let key = ChatterboxCoreMLModel.compiledModelCacheNames(for: pkg).key
        let stable = cacheDir.appendingPathComponent("\(key).mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: stable, withIntermediateDirectories: true)
        try Data("seed".utf8).write(to: stable.appendingPathComponent("coremldata.bin"))
        // Reaching a non-throwing return == the warm branch fired: had it fallen
        // through to MLModel.compileModel, this fake package would have thrown.
        let out = try await ChatterboxCoreMLModel.compiledModel(at: pkg, cacheDir: cacheDir)
        #expect(out.path == stable.path)
    }

    @Test("completeness sentinel distinguishes a full vs a partial .mlmodelc")
    func completenessSentinel() throws {
        let fm = FileManager.default
        let dir = try tempDir(); defer { try? fm.removeItem(at: dir) }
        // Full: has the coremldata.bin sentinel every compiled .mlmodelc carries.
        let full = dir.appendingPathComponent("full.mlmodelc", isDirectory: true)
        try fm.createDirectory(at: full, withIntermediateDirectories: true)
        try Data().write(to: full.appendingPathComponent("coremldata.bin"))
        // Partial: dir present but no sentinel (an interrupted compile). The warm path
        // must NOT reuse this — `compiledModel` gates its warm hit on this predicate
        // and evicts + recompiles when it's false (self-heal).
        let partial = dir.appendingPathComponent("partial.mlmodelc", isDirectory: true)
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)
        try Data("junk".utf8).write(to: partial.appendingPathComponent("model.mil"))
        #expect(ChatterboxCoreMLModel.isCompleteCompiledModel(full))
        #expect(!ChatterboxCoreMLModel.isCompleteCompiledModel(partial))
    }

    @Test("prune removes stale same-source artifacts and keeps foreign packages")
    func pruneScopedToSource() throws {
        let fm = FileManager.default
        let cacheDir = try tempDir(); defer { try? fm.removeItem(at: cacheDir) }
        func seed(_ name: String) throws -> URL {
            let u = cacheDir.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
            try fm.createDirectory(at: u, withIntermediateDirectories: true)
            return u
        }
        let stale = try seed("T3LM-aaaa-oldstamp")     // same source, old version
        let keep = try seed("T3LM-aaaa-newstamp")      // same source, current version
        let otherVariant = try seed("T3LM-bbbb-stamp") // other variant's same-named T3LM
        let foreign = try seed("S3Encoder-cccc-stamp") // a different package
        ChatterboxCoreMLModel.pruneStaleSiblings(in: cacheDir, prunePrefix: "T3LM-aaaa-", keep: keep)
        #expect(!fm.fileExists(atPath: stale.path))         // pruned
        #expect(fm.fileExists(atPath: keep.path))           // kept
        #expect(fm.fileExists(atPath: otherVariant.path))   // untouched (different path hash)
        #expect(fm.fileExists(atPath: foreign.path))        // untouched (different stem)
    }

    /// Regression for the per-launch recompile: the cache key must be INDEPENDENT of
    /// the resolved absolute model path. The same repo can be discovered at the flat
    /// `…/models/<org>/<name>/…` layout or the HF-cache
    /// `…/models--<org>--<name>/snapshots/<sha>/…` layout, and at `/var` vs
    /// `/private/var` — all must share ONE compiled artifact (else the warm `.mlmodelc`
    /// is never found and the ANE AOT compile is paid every launch). A *different*
    /// repo must still get a distinct prefix so pruning one variant never evicts the
    /// other.
    @Test("compiled-cache key is path-independent for the same repo, distinct across repos")
    func keyStableAcrossLayoutsSameRepo() {
        func prefix(_ p: String) -> String {
            ChatterboxCoreMLModel.compiledModelCacheNames(for: URL(fileURLWithPath: p)).prunePrefix
        }
        let container = "/mobile/Containers/Data/Application/587ADCF8/Library/Caches/huggingface/hub"
        let turboFlat = "/var\(container)/models/iliasaz/chatterbox-turbo-coreml/T3LM.mlpackage"
        let turboSnapA = "/private/var\(container)/models--iliasaz--chatterbox-turbo-coreml/snapshots/aaaa/T3LM.mlpackage"
        let turboSnapB = "/var\(container)/models--iliasaz--chatterbox-turbo-coreml/snapshots/bbbb/T3LM.mlpackage"
        let mtlSnap = "/var\(container)/models--iliasaz--chatterbox-multilingual-coreml/snapshots/cccc/T3LM.mlpackage"
        #expect(prefix(turboFlat) == prefix(turboSnapA))    // flat ≡ HF snapshot
        #expect(prefix(turboSnapA) == prefix(turboSnapB))   // snapshot sha + /var flip don't matter
        #expect(prefix(turboFlat) != prefix(mtlSnap))       // turbo ≠ multilingual (no cross-evict)
        // And the model-identity itself collapses the layouts to the stable repo+file.
        #expect(ChatterboxCoreMLModel.modelIdentity(for: URL(fileURLWithPath: turboFlat))
            == "iliasaz/chatterbox-turbo-coreml/T3LM.mlpackage")
        #expect(ChatterboxCoreMLModel.modelIdentity(for: URL(fileURLWithPath: turboSnapA))
            == "iliasaz/chatterbox-turbo-coreml/T3LM.mlpackage")
    }

    /// All THREE variants ship a file literally named `T3LM.mlpackage`, so the keying
    /// has to separate them by *repo*, not filename. A shared prune prefix would make
    /// every launch evict the other variants' persisted `.mlmodelc` — i.e. re-pay the
    /// ~52 s ANE AOT compile per launch, exactly the regression issue #23 fixed.
    @Test("turbo / nano / multilingual cache keys + prune prefixes are pairwise distinct")
    func keysDistinctAcrossAllThreeVariants() {
        func names(_ p: String) -> (key: String, prunePrefix: String) {
            ChatterboxCoreMLModel.compiledModelCacheNames(for: URL(fileURLWithPath: p))
        }
        let container = "/mobile/Containers/Data/Application/587ADCF8/Library/Caches/huggingface/hub"
        let turbo = "/var\(container)/models/iliasaz/chatterbox-turbo-coreml/T3LM.mlpackage"
        let nano = "/var\(container)/models--iliasaz--chatterbox-nano-coreml/snapshots/dddd/T3LM.mlpackage"
        let mtl = "/var\(container)/models--iliasaz--chatterbox-multilingual-coreml/snapshots/cccc/T3LM.mlpackage"

        #expect(ChatterboxCoreMLModel.modelIdentity(for: URL(fileURLWithPath: nano))
            == "iliasaz/chatterbox-nano-coreml/T3LM.mlpackage")   // repo, not path
        let all = [turbo, nano, mtl]
        #expect(Set(all.map { names($0).prunePrefix }).count == 3)   // pairwise distinct
        #expect(Set(all.map { names($0).key }).count == 3)

        // Same for the converter's local output dirs, which sit side by side on a dev
        // box (`out/` = turbo, `out-nano/` = nano) and hit `modelIdentity`'s
        // parent-dir fallback — no `models/` or `models--` component to key off.
        let outTurbo = "/Users/x/chatterbox-coreml/out/T3LM.mlpackage"
        let outNano = "/Users/x/chatterbox-coreml/out-nano/T3LM.mlpackage"
        #expect(names(outTurbo).prunePrefix != names(outNano).prunePrefix)
        #expect(names(outTurbo).prunePrefix != names(turbo).prunePrefix)   // local ≠ downloaded
    }
}
