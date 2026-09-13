import XCTest

/// Regression coverage for the macOS "empty window" bug: a `.disabled()` modifier
/// applied to a `Section` (wrapping it in `ModifiedContent`) made `Form` render
/// **nothing** on macOS (iOS tolerated it). These tests assert the Form's sections
/// actually render. Run on macOS (where the bug lived) and iOS.
///
/// Launched with `CHATTERBOX_UI_TEST` (skips the Keychain prompt, which blocks
/// automation) and `CHATTERBOX_NO_AUTORUN` (no model download/load during the test).
final class ChatterboxAppUITests: XCTestCase {
    @MainActor
    private func launchedApp() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["CHATTERBOX_UI_TEST"] = "1"
        app.launchEnvironment["CHATTERBOX_NO_AUTORUN"] = "1"
        app.launch()
        return app
    }

    /// The Form must render its sections (the bug made the whole Form empty).
    /// Checks the Voice section header and the Parameters' Temperature row — both
    /// disappear when the Form collapses. (Rows far below the fold aren't asserted:
    /// iOS `List` lazily materializes off-screen rows, so they're absent from the
    /// AX tree until scrolled; macOS renders them all.)
    @MainActor
    func testFormRendersSections() {
        let app = launchedApp()
        XCTAssertTrue(app.staticTexts["Temperature"].waitForExistence(timeout: 20),
                      "Parameters section did not render — Form is empty.\n\(app.debugDescription)")
        XCTAssertTrue(app.staticTexts["Voice"].exists, "missing Voice section")
    }

    /// The model picker exposes every variant (turbo + nano + multilingual). The
    /// segments carry the short name (`segmentLabel`), not the prose `displayName` —
    /// three "… (English)" labels truncate in a segmented control on iPhone.
    @MainActor
    func testModelPickerPresent() {
        let app = launchedApp()
        for segment in ["Turbo", "Nano", "Multilingual"] {
            XCTAssertTrue(app.buttons[segment].waitForExistence(timeout: 20)
                          || app.staticTexts[segment].exists,
                          "Model picker missing the \(segment) segment.\n\(app.debugDescription)")
        }
    }
}
