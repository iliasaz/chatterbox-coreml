import SwiftUI

@main
struct ChatterboxApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 480, minHeight: 380)
        }
        #if os(macOS)
        .windowResizability(.contentMinSize)
        #endif
    }
}
