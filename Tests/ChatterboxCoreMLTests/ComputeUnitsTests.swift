import CoreML
import Testing
@testable import ChatterboxCoreML

/// Default per-stage placement. The S3Encoder stays on the CPU in both the default and the
/// all-Neural-Engine (background) placement: on iOS 27 its RangeDim graph on the ANE exhausts
/// memory and the host app crashes (see `SynthRunner.encoderDefault`).
struct ComputeUnitsTests {
    @Test func encoderDefaultsToCPU() {
        #expect(SynthRunner.encoderDefault == .cpuOnly)
    }

    @Test func neuralEnginePresetKeepsEncoderOnCPU() {
        #expect(PipelineComputeUnits.neuralEngine.encoder == .cpuOnly)
    }

    @Test func neuralEnginePresetPinsEveryOtherStageToTheANE() {
        let units = PipelineComputeUnits.neuralEngine
        #expect(units.t3 == .cpuAndNeuralEngine)
        #expect(units.cfm == .cpuAndNeuralEngine)
        #expect(units.vocoder == .cpuAndNeuralEngine)
        #expect(units.watermark == .cpuAndNeuralEngine)
    }
}
