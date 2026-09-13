import Foundation
import Testing
@testable import ChatterboxCoreML

/// The Objective-C exception trap around CoreML predictions.
///
/// The device crash these guard (2026-07-26) was an ANE `E5RT: No memory object bound
/// to port` raised as an `NSGenericException` out of `MLModel.prediction(from:)`.
/// Swift has no `catch` for `NSException`, so it unwound past the pipeline's
/// `do/catch` into `std::terminate` and killed the app. Without the trap, the first
/// test here would not fail — **the test process would die**, which is precisely the
/// behavior being fixed.
struct PredictionTrapTests {
    @Test func objcExceptionBecomesASwiftError() throws {
        // Verbatim shape of the device fault: an NSGenericException with an E5RT reason.
        let error = #expect(throws: ChatterboxError.self) {
            try trapping("S3Encoder") {
                NSException(
                    name: .genericException,
                    reason: "Failed to add operation to E5 stream. E5RT: No memory object bound to port. (2)",
                    userInfo: nil
                ).raise()
                return 0
            }
        }
        guard case .prediction(let label, let reason) = error else {
            Issue.record("expected .prediction, got \(String(describing: error))")
            return
        }
        #expect(label == "S3Encoder")
        #expect(reason.contains("No memory object bound to port"))
        // And the description a device log would show names the faulting stage.
        #expect(error?.description.contains("S3Encoder prediction failed") == true)
    }

    @Test func successfulWorkReturnsItsValueUntouched() throws {
        #expect(try trapping("S3Vocoder") { [1.5, 2.5] as [Float] } == [1.5, 2.5])
    }

    @Test func swiftErrorsPropagateUnconverted() {
        // Only NSException is converted; a normal `throws` failure keeps its identity
        // so existing error handling (and its messages) are unchanged.
        let error = #expect(throws: ChatterboxError.self) {
            try trapping("S3CFM") {
                throw ChatterboxError.invalidModelOutput("missing output 'velocity'")
            }
        }
        guard case .invalidModelOutput(let message) = error else {
            Issue.record("expected .invalidModelOutput, got \(String(describing: error))")
            return
        }
        #expect(message == "missing output 'velocity'")
    }

    @Test func trapIsReusableAfterCatching() throws {
        // A trapped fault must not wedge the trap: the pipeline keeps synthesizing
        // later utterances after one fails.
        _ = try? trapping("S3Encoder") {
            NSException(name: .genericException, reason: "boom", userInfo: nil).raise()
        }
        #expect(try trapping("S3Encoder") { 42 } == 42)
    }
}
