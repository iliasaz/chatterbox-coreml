import Testing
import Foundation
import CoreML
@testable import ChatterboxCoreML

/// Probe (opt-in): can the host write into a decode `MLState` buffer via
/// `withMultiArray` and then predict? If yes, the fast ANE-clean batched
/// `T3Prefill` can seed the stateful `T3Decode` directly (no ANE-hostile
/// stateful-prefill function needed). Run:
///   CHATTERBOX_STATE_PROBE=/path/to/out/T3Decode.mlpackage \
///     swift test --filter StateSeedProbe
struct StateSeedProbe {
    @Test func hostCanSeedDecodeStateThenPredict() async throws {
        #if arch(arm64)
        guard let pkg = ProcessInfo.processInfo.environment["CHATTERBOX_STATE_PROBE"] else { return }
        let cu = ProcessInfo.processInfo.environment["CHATTERBOX_DECODE_CU"]?.lowercased()
        let units: MLComputeUnits = (cu == "ane" || cu == "cpuandne") ? .cpuAndNeuralEngine
            : (cu == "cpu" ? .cpuOnly : .cpuAndGPU)

        let url = URL(fileURLWithPath: pkg)
        let compiled = url.pathExtension == "mlpackage" ? try await MLModel.compileModel(at: url) : url
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        let model = try MLModel(contentsOf: compiled, configuration: cfg)
        let maxSeq = model.modelDescription.inputDescriptionsByName["update_mask"]?
            .multiArrayConstraint?.shape.first?.intValue ?? 1536
        // Hidden width from the graph, NEVER `Constants.gpt2Hidden` — nano's decode
        // takes (1, 1, 768), turbo's (1, 1, 1024).
        let hidden = model.modelDescription.inputDescriptionsByName["inputs_embeds"]?
            .multiArrayConstraint?.shape.last?.intValue ?? Constants.gpt2Hidden

        let state = model.makeState()
        // Seed BOTH buffers in SEPARATE scopes — exactly as CoreMLDecoder.seedState
        // does. Nesting these traps (EXC_BREAKPOINT); this guards that regression.
        for name in ["keyCache", "valueCache"] {
            state.withMultiArray(for: name) { arr in
                // KV row = layers·heads·head_dim, read off the buffer: turbo 24576,
                // nano 9216. Computing it from `Constants` (turbo's 24576) on nano
                // stays INSIDE the buffer — (maxSeq, 9216) is far larger than the 3
                // rows written — but strides across the wrong rows, so the seed lands
                // as garbage instead of overrunning. Silent corruption, not a crash.
                let rowStride = arr.shape.last?.intValue ?? (arr.count / max(maxSeq, 1))
                let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
                for r in 0..<3 { for c in 0..<rowStride { p[r * rowStride + c] = Float16(0.01) } }
            }
        }
        // One decode predict at position 3.
        let um = try MLMultiArray(shape: [NSNumber(value: maxSeq), 1], dataType: .float16)
        let am = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: maxSeq)], dataType: .float16)
        let umP = um.dataPointer.bindMemory(to: Float16.self, capacity: maxSeq)
        let amP = am.dataPointer.bindMemory(to: Float16.self, capacity: maxSeq)
        for i in 0..<maxSeq { umP[i] = 0; amP[i] = i <= 3 ? 0 : Float16(-1e4) }
        umP[3] = 1
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLMultiArray.float16([Float](repeating: 0.02, count: hidden), shape: [1, 1, hidden]),
            "position_ids": MLMultiArray.int32([3], shape: [1, 1]),
            "update_mask": um,
            "attn_mask": am,
        ])
        let out = try await model.prediction(from: provider, using: state, options: MLPredictionOptions())
        let logits = out.featureValue(for: "logits")!.multiArrayValue!
        #expect(logits.count == Constants.speechVocabSize)   // 6563 — shared by turbo + nano
        FileHandle.standardError.write(Data("[probe] withMultiArray seed + predict OK on units=\(units.rawValue)\n".utf8))
        #endif
    }
}
