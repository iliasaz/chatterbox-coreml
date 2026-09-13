import Testing
import Foundation
@testable import ChatterboxCoreML

/// Pure unit tests for the cross-chunk pipeline hand-off (issue #22). No CoreML
/// models — exercises `BoundedChunkChannel`'s FIFO order, backpressure, finish,
/// failure, and the producer/consumer wiring that `produce` uses, with mock
/// index-tagged `DecodedChunk`s.
struct ChunkPipelineTests {
    /// Minimal conditioning value (never inspected here — only carried through).
    private static let conds = Conditionals(
        speakerEmb: [0], condPromptSpeechTokens: [], genEmbedding: [0],
        promptTokens: [], promptFeat: [])

    /// A `DecodedChunk` whose tokens encode its index, so the consumer can verify
    /// identity end-to-end (no reorder, no drop, no duplication).
    private static func chunk(_ i: Int) -> DecodedChunk {
        DecodedChunk(
            index: i,
            result: T3Result(tokens: [i], prefillLength: 0, prefillTime: 0, decodeTime: 0),
            conds: conds)
    }

    private enum MockError: Error, Equatable { case boom }

    /// Drives the same producer(child)/consumer(drain) shape as `produce`, over a
    /// real `BoundedChunkChannel`, and returns the consumed indices in order.
    private func runPipeline(chunkCount: Int, capacity: Int) async throws -> [Int] {
        let channel = BoundedChunkChannel(capacity: capacity)
        let producer = Task {
            for i in 0..<chunkCount { await channel.send(Self.chunk(i)) }
            await channel.finish()
        }
        var out: [Int] = []
        while let dc = try await channel.receive() {
            out.append(dc.result.tokens[0])   // tokens == [index]
            #expect(dc.index == out.count - 1) // arrives strictly in order
        }
        await producer.value
        return out
    }

    @Test(arguments: [
        (count: 1, cap: 1), (count: 2, cap: 2), (count: 3, cap: 2),
        (count: 5, cap: 2), (count: 5, cap: 1), (count: 8, cap: 3),
    ])
    func deliversEveryChunkInOrder(count: Int, cap: Int) async throws {
        let out = try await runPipeline(chunkCount: count, capacity: cap)
        #expect(out == Array(0..<count))   // ordered, none dropped/duplicated
    }

    @Test func emptyStreamFinishesWithNoChunks() async throws {
        let out = try await runPipeline(chunkCount: 0, capacity: 2)
        #expect(out.isEmpty)
    }

    @Test func finishAfterBufferingDrainsThenReturnsNil() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        await channel.send(Self.chunk(0))
        await channel.send(Self.chunk(1))
        await channel.finish()
        #expect(try await channel.receive()?.index == 0)
        #expect(try await channel.receive()?.index == 1)
        #expect(try await channel.receive() == nil)   // drained + finished
    }

    /// The bounded buffer must stall the producer when full: with capacity 1, a
    /// second `send` cannot complete until a `receive` frees the slot.
    @Test func fullChannelSuspendsProducer() async throws {
        actor Latch { var fired = false; func set() { fired = true }; func get() -> Bool { fired } }
        let channel = BoundedChunkChannel(capacity: 1)
        let latch = Latch()

        await channel.send(Self.chunk(0))             // fills the single slot
        let producer = Task {
            await channel.send(Self.chunk(1))         // must block until a receive
            await latch.set()
        }
        try await Task.sleep(nanoseconds: 30_000_000) // let the producer reach suspension
        #expect(await latch.get() == false)           // still blocked — backpressure

        #expect(try await channel.receive()?.index == 0)  // frees the slot → wakes producer
        #expect(try await channel.receive()?.index == 1)
        await producer.value
        #expect(await latch.get() == true)
    }

    @Test func failWakesSuspendedReceiver() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        let receiver = Task { try await channel.receive() }   // suspends (empty)
        try await Task.sleep(nanoseconds: 20_000_000)
        await channel.fail(MockError.boom)
        await #expect(throws: MockError.boom) { _ = try await receiver.value }
    }

    @Test func failWakesSuspendedSenderWithoutHanging() async throws {
        let channel = BoundedChunkChannel(capacity: 1)
        await channel.send(Self.chunk(0))                     // full
        let sender = Task { await channel.send(Self.chunk(1)) } // suspends (full)
        try await Task.sleep(nanoseconds: 20_000_000)
        await channel.fail(MockError.boom)
        await sender.value                                    // returns, does not hang
    }

    @Test func failTakesPrecedenceOverBufferedChunks() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        await channel.send(Self.chunk(0))
        await channel.fail(MockError.boom)
        await #expect(throws: MockError.boom) { _ = try await channel.receive() }
    }

    /// A producer that fails mid-stream must surface the error to the consumer
    /// (mirrors `produce`'s decode-error → `channel.fail` → synth-throws path).
    @Test func producerFailurePropagatesToConsumer() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        let producer = Task {
            await channel.send(Self.chunk(0))
            await channel.fail(MockError.boom)   // decode error after one chunk
        }
        var received: [Int] = []
        var thrown: Error?
        do {
            while let dc = try await channel.receive() { received.append(dc.index) }
        } catch { thrown = error }
        await producer.value
        #expect(thrown as? MockError == .boom)
        // The first chunk may or may not be drained before the failure is seen;
        // either way no out-of-order or post-failure chunk leaks through.
        #expect(received == [] || received == [0])
    }

    /// Records what the mock synth consumer saw (order-preserving, off-actor safe).
    private actor Recorder {
        private(set) var indices: [Int] = []
        func add(_ i: Int) { indices.append(i) }
    }

    /// Mirrors `ChatterboxCoreMLModel.produce`'s overlap shape (two child tasks +
    /// channel, each catch → `fail` → rethrow) so the cancellation/error tests
    /// exercise the *real* termination contract, not a simplified stand-in.
    /// Cancelling the returned `Task` must unwind BOTH legs without hanging — even
    /// when the decoder is parked in `send` (full buffer) and the consumer in
    /// `receive` (empty) cannot both happen at once, so a parked side is always
    /// matched by a live side that resumes it (via `receive`'s drain or `fail`).
    @Test(.timeLimit(.minutes(1)))
    func cancelledOverlapPipelineUnwindsCleanly() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        let rec = Recorder()
        let pipeline = Task {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {                      // decode — fast, unbounded
                    do {
                        var i = 0
                        while true {
                            try Task.checkCancellation()
                            await channel.send(Self.chunk(i)); i += 1
                        }
                    } catch { await channel.fail(error); throw error }
                }
                group.addTask {                      // synth — slow consumer
                    do {
                        while let dc = try await channel.receive() {
                            try Task.checkCancellation()
                            await rec.add(dc.index)
                            try await Task.sleep(nanoseconds: 10_000_000)
                        }
                    } catch { await channel.fail(error); throw error }
                }
                try await group.waitForAll()
            }
        }
        // Let the buffer fill so the decoder parks in send() before we cancel.
        try await Task.sleep(nanoseconds: 50_000_000)
        pipeline.cancel()
        await #expect(throws: CancellationError.self) { try await pipeline.value }
        #expect(await rec.indices.count >= 1)        // made progress before cancel
    }

    private struct DecodeFailure: Error {}

    /// A decode error mid-stream must abort BOTH tasks and surface to the consumer,
    /// with no chunk at/after the failing index ever emitted (no out-of-order or
    /// post-failure leak) — the produce()-level analogue of the channel `fail` path.
    @Test(.timeLimit(.minutes(1)))
    func decodeErrorAbortsBothOverlapTasks() async throws {
        let channel = BoundedChunkChannel(capacity: 2)
        let rec = Recorder()
        let pipeline = Task {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {                      // decode — throws on chunk 2
                    do {
                        for i in 0..<10 {
                            try Task.checkCancellation()
                            if i == 2 { throw DecodeFailure() }
                            await channel.send(Self.chunk(i))
                        }
                        await channel.finish()
                    } catch { await channel.fail(error); throw error }
                }
                group.addTask {                      // synth
                    do {
                        while let dc = try await channel.receive() {
                            try Task.checkCancellation()
                            await rec.add(dc.index)
                        }
                    } catch { await channel.fail(error); throw error }
                }
                try await group.waitForAll()
            }
        }
        await #expect(throws: DecodeFailure.self) { try await pipeline.value }
        let got = await rec.indices
        #expect(got.allSatisfy { $0 < 2 })           // nothing at/after the failure leaked
        #expect(Set(got).count == got.count)         // no duplicates
        #expect(got == got.sorted())                 // in order
    }
}
