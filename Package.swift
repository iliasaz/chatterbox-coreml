// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ChatterboxCoreML",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "ChatterboxCoreML", targets: ["ChatterboxCoreML"]),
        .executable(name: "chatterbox-cli", targets: ["ChatterboxCLI"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers.git",
            from: "1.3.0"
        ),
        .package(
            url: "https://github.com/jkrukowski/swift-safetensors.git",
            from: "0.1.1"
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            from: "0.8.1"
        ),
        // Russian neural stress, from `main` of `iliasaz/ruaccent-coreml`. Both
        // packages pin swift-transformers from 1.3.0, so SPM dedups to one copy.
        // See docs/ruaccent-integration.md.
        .package(url: "https://github.com/iliasaz/ruaccent-coreml.git", branch: "main"),
        // Perth-Net Implicit audio watermarking (`PerthWatermarker`). Upstream
        // chatterbox watermarks every utterance it returns, so we do too — see
        // `Watermarker.swift`. No transitive dependencies (CoreML + Accelerate only).
        .package(url: "https://github.com/iliasaz/perth-coreml.git", branch: "main"),
    ],
    targets: [
        // The package's only Objective-C. `@try/@catch` has no Swift equivalent, and
        // CoreML `@throw`s some accelerator faults (ANE `E5RT`) as `NSException`,
        // which unwinds past every Swift `do/catch` and terminates the process. See
        // `CBXExceptionTrap.h` / `PredictionTrap.swift`.
        .target(name: "ChatterboxExceptionTrap"),
        .target(
            name: "ChatterboxCoreML",
            dependencies: [
                "ChatterboxExceptionTrap",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Safetensors", package: "swift-safetensors"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                // Product name `RUAccentCoreML`; package identity `ruaccent-coreml`.
                .product(name: "RUAccentCoreML", package: "ruaccent-coreml"),
                // Product name `PerthCoreML`; package identity `perth-coreml`.
                .product(name: "PerthCoreML", package: "perth-coreml"),
            ]
        ),
        .executableTarget(
            name: "ChatterboxCLI",
            dependencies: ["ChatterboxCoreML"]
        ),
        .testTarget(
            name: "ChatterboxCoreMLTests",
            dependencies: [
                "ChatterboxCoreML",
                // Used by the (opt-in, network) Hub incremental-download test.
                .product(name: "Hub", package: "swift-transformers"),
                // The opt-in end-to-end watermark test detects with `PerthWatermarker`.
                .product(name: "PerthCoreML", package: "perth-coreml"),
            ],
            // Test fixtures loaded by path (`#filePath`), not as SwiftPM resources —
            // some committed, some generated locally and gitignored. Excluding
            // silences the "unhandled files" build warning.
            exclude: ["Fixtures"]
        ),
    ]
)
