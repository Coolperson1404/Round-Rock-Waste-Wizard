//
//  ContentView.swift
//  Waste Wizard - Round Rock
//
//  Created by Shresht Hunnolli on 8/15/26.
//

import SwiftUI
import WebKit
import Vision
import CoreML

/// Serves the bundled web app over a custom scheme instead of `file://`.
///
/// WKWebView gives a `file://` page an opaque origin, so `fetch()` / XHR to a
/// neighbouring bundled file fails ("Load failed") even when the web view was
/// given read access to the directory. That blocks TensorFlow.js from reading
/// the bundled MobileNet weights, and it makes localStorage unreliable.
/// A custom scheme gives the page a real origin, so both work normally.
final class BundleSchemeHandler: NSObject, WKURLSchemeHandler {

    static let scheme = "wwapp"
    static let indexURL = URL(string: "\(scheme)://app/index.html")!

    private let queue = DispatchQueue(label: "BundleSchemeHandler.io", qos: .userInitiated)
    private let lock = NSLock()
    private var liveTasks = Set<ObjectIdentifier>()

    private static let mimeTypes: [String: String] = [
        "html": "text/html; charset=utf-8",
        "js":   "text/javascript; charset=utf-8",
        "mjs":  "text/javascript; charset=utf-8",
        "css":  "text/css; charset=utf-8",
        "json": "application/json; charset=utf-8",
        "bin":  "application/octet-stream",
        "png":  "image/png",
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "gif":  "image/gif",
        "svg":  "image/svg+xml",
        "ico":  "image/x-icon",
        "woff": "font/woff",
        "woff2": "font/woff2"
    ]

    private func isLive(_ task: WKURLSchemeTask) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return liveTasks.contains(ObjectIdentifier(task))
    }

    func webView(_ webView: WKWebView, start urlSchemeTask: WKURLSchemeTask) {
        lock.lock()
        liveTasks.insert(ObjectIdentifier(urlSchemeTask))
        lock.unlock()

        let requestURL = urlSchemeTask.request.url

        queue.async { [weak self] in
            guard let self else { return }

            // Resources are bundled flat, so only the file name matters.
            // Using the last path component also rules out directory traversal.
            var name = (requestURL?.path as NSString?)?.lastPathComponent ?? ""
            if name.isEmpty { name = "index.html" }

            guard let root = Bundle.main.resourceURL else {
                self.finish(urlSchemeTask, error: URLError(.fileDoesNotExist))
                return
            }
            let fileURL = root.appendingPathComponent(name)

            guard FileManager.default.fileExists(atPath: fileURL.path),
                  let data = try? Data(contentsOf: fileURL),
                  let url = requestURL else {
                self.finish(urlSchemeTask, error: URLError(.fileDoesNotExist))
                return
            }

            let ext = fileURL.pathExtension.lowercased()
            let mime = Self.mimeTypes[ext] ?? "application/octet-stream"
            let response = HTTPURLResponse(
                url: url,
                statusCode: 200,
                httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": mime,
                    "Content-Length": String(data.count),
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "no-cache"
                ]
            )!

            DispatchQueue.main.async {
                // The web view may have cancelled the task while we were reading;
                // calling back into a stopped task traps.
                guard self.isLive(urlSchemeTask) else { return }
                urlSchemeTask.didReceive(response)
                urlSchemeTask.didReceive(data)
                urlSchemeTask.didFinish()
                self.forget(urlSchemeTask)
            }
        }
    }

    func webView(_ webView: WKWebView, stop urlSchemeTask: WKURLSchemeTask) {
        forget(urlSchemeTask)
    }

    private func forget(_ task: WKURLSchemeTask) {
        lock.lock()
        liveTasks.remove(ObjectIdentifier(task))
        lock.unlock()
    }

    private func finish(_ task: WKURLSchemeTask, error: Error) {
        DispatchQueue.main.async {
            guard self.isLive(task) else { return }
            task.didFailWithError(error)
            self.forget(task)
        }
    }
}

/// Exposes Apple's Vision framework to the web app.
///
/// The web page has no usable alternative on iOS: WKWebView does not implement
/// `BarcodeDetector`, and the pure-JS fallback only reads UPC-A codes that are
/// close to level and in focus. Vision handles every symbology at real-world
/// angles, and additionally gives text recognition and image classification
/// with no model to download.
///
/// Uses `WKScriptMessageHandlerWithReply`, so `postMessage` returns a Promise
/// on the JavaScript side rather than needing a callback registry.
final class VisionBridge: NSObject, WKScriptMessageHandlerWithReply {

    static let name = "vision"

    private let queue = DispatchQueue(label: "VisionBridge.work", qos: .userInitiated)

    /// Loads a compiled Core ML model (`<name>.mlmodelc`) from the bundle, or
    /// nil when it isn't bundled. Xcode compiles a dropped-in `<name>.mlmodel`
    /// into the bundle automatically.
    private static func loadModel(_ name: String) -> VNCoreMLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let ml = try? MLModel(contentsOf: url),
              let vn = try? VNCoreMLModel(for: ml) else { return nil }
        return vn
    }

    /// Loaded once each.
    private static let materialModel = loadModel("WasteMaterial")
    private static let itemModel = loadModel("WasteItems")

    func userContentController(_ userContentController: WKUserContentController,
                               didReceive message: WKScriptMessage,
                               replyHandler: @escaping (Any?, String?) -> Void) {

        guard let body = message.body as? [String: Any],
              let dataURL = body["image"] as? String,
              let cgImage = Self.decodeImage(dataURL) else {
            replyHandler(nil, "vision: could not decode the image")
            return
        }

        let ops = Set((body["ops"] as? [String]) ?? ["barcode", "text", "classify"])

        queue.async {
            var result: [String: Any] = ["barcodes": [], "text": [], "classes": []]
            var errors: [String: String] = [:]

            // Each request is performed on its own. Vision's perform([…]) throws
            // on the first failure and discards the rest, so batching would let
            // one unavailable request throw away perfectly good results from the
            // others.
            //
            // Newest revision first, then older ones: the newest algorithms want
            // a Neural Engine backend that not every environment provides (the
            // Simulator reports "Could not create inference context" for barcode
            // detection and "Failed to create espresso context" for
            // classification). Older revisions often have a CPU path.
            func run(_ op: String, revisions: [Int], _ make: () -> VNRequest) -> [VNObservation] {
                var lastError: String?
                for revision in revisions {
                    let request = make()
                    if request.revision != revision, type(of: request).supportedRevisions.contains(revision) {
                        request.revision = revision
                    }
                    do {
                        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up, options: [:])
                        try handler.perform([request])
                        return request.results ?? []
                    } catch {
                        lastError = "rev \(revision): \(error.localizedDescription)"
                    }
                }
                if let lastError { errors[op] = lastError }
                return []
            }

            if ops.contains("barcode") {
                let revisions = VNDetectBarcodesRequest.supportedRevisions.sorted(by: >)
                let found = run("barcode", revisions: revisions) { VNDetectBarcodesRequest() }
                    .compactMap { $0 as? VNBarcodeObservation }

                // Vision reports one observation per detected instance, so a
                // single printed barcode comes back a dozen times. Keep the
                // best-confidence reading of each distinct payload.
                var best: [String: VNBarcodeObservation] = [:]
                for o in found {
                    guard let value = o.payloadStringValue else { continue }
                    if let existing = best[value], existing.confidence >= o.confidence { continue }
                    best[value] = o
                }
                result["barcodes"] = best
                    .sorted { $0.value.confidence > $1.value.confidence }
                    .map { ["value": $0.key,
                            "symbology": $0.value.symbology.rawValue,
                            "confidence": Double($0.value.confidence)] }
            }

            if ops.contains("text") {
                let revisions = VNRecognizeTextRequest.supportedRevisions.sorted(by: >)
                result["text"] = run("text", revisions: revisions) {
                    let request = VNRecognizeTextRequest()
                    request.recognitionLevel = .accurate
                    // Packaging text is codes and brand names, not prose; language
                    // correction actively hurts things like "PET 1" and "HDPE".
                    request.usesLanguageCorrection = false
                    return request
                }
                .compactMap { obs -> [String: Any]? in
                    guard let o = obs as? VNRecognizedTextObservation,
                          let best = o.topCandidates(1).first else { return nil }
                    // Vision's origin is bottom-left; flip to top-left so the
                    // web side can reason in the same space as a canvas.
                    let b = o.boundingBox
                    return ["value": best.string,
                            "confidence": Double(best.confidence),
                            "box": ["x": Double(b.minX), "y": Double(1 - b.maxY),
                                    "w": Double(b.width), "h": Double(b.height)]]
                }
            }

            // Runs a bundled classifier and reports its top `top` labels under
            // `key`, or records "model not bundled" (and sets no key) if absent.
            func classify(_ op: String, model: VNCoreMLModel?, key: String, top: Int) {
                guard let model else {
                    errors[op] = "model not bundled"
                    return
                }
                let revisions = VNCoreMLRequest.supportedRevisions.sorted(by: >)
                let found = run(op, revisions: revisions) {
                    let r = VNCoreMLRequest(model: model)
                    r.imageCropAndScaleOption = .centerCrop
                    return r
                }
                result[key] = found
                    .compactMap { $0 as? VNClassificationObservation }
                    .sorted { $0.confidence > $1.confidence }
                    .prefix(top)
                    .map { ["value": $0.identifier, "confidence": Double($0.confidence)] }
            }

            if ops.contains("material") {
                // Purpose-trained on waste materials. It only has to separate six
                // materials rather than a thousand ImageNet classes, which is why
                // it avoids errors like calling an aluminium can "hair spray" and
                // routing it to hazardous waste.
                classify("material", model: Self.materialModel, key: "materials", top: 3)
            }

            if ops.contains("item") {
                // Trained on the app's own item categories (WasteItems.mlmodel).
                classify("item", model: Self.itemModel, key: "items", top: 5)
            }

            if ops.contains("featureprint") {
                // A 768-float embedding of the image. The web layer stores these
                // against corrections and compares them with a plain L2 distance,
                // which reproduces VNFeaturePrintObservation.computeDistance
                // exactly, so no round trip is needed to compare two images.
                let revisions = VNGenerateImageFeaturePrintRequest.supportedRevisions.sorted(by: >)
                let found = run("featureprint", revisions: revisions) { VNGenerateImageFeaturePrintRequest() }
                if let o = found.first as? VNFeaturePrintObservation {
                    result["featureprint"] = [
                        "data": o.data.base64EncodedString(),
                        "elementCount": o.elementCount
                    ]
                }
            }

            if ops.contains("classify") {
                // Returns well over a thousand labels, almost all near zero.
                // Send only what could plausibly matter.
                let revisions = VNClassifyImageRequest.supportedRevisions.sorted(by: >)
                let classes = run("classify", revisions: revisions) { VNClassifyImageRequest() }
                    .compactMap { $0 as? VNClassificationObservation }
                    .filter { $0.confidence >= 0.05 }
                    .sorted { $0.confidence > $1.confidence }
                    .prefix(15)
                    .map { ["value": $0.identifier, "confidence": Double($0.confidence)] }
                result["classes"] = Array(classes)
            }

            if !errors.isEmpty { result["errors"] = errors }
            replyHandler(result, nil)
        }
    }

    private static func decodeImage(_ dataURL: String) -> CGImage? {
        // Accepts a full "data:image/jpeg;base64,…" URL or bare base64.
        let base64 = dataURL.range(of: ",").map { String(dataURL[$0.upperBound...]) } ?? dataURL
        guard let data = Data(base64Encoded: base64, options: .ignoreUnknownCharacters),
              let image = UIImage(data: data) else { return nil }
        return image.cgImage
    }
}

struct HTMLWebView: UIViewRepresentable {
    let fileName: String

    class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        // Held here so they live as long as the web view.
        let schemeHandler = BundleSchemeHandler()
        let visionBridge = VisionBridge()

        private func isExternal(_ url: URL) -> Bool {
            let s = url.absoluteString.lowercased()
            return s.hasPrefix("tel:") || s.hasPrefix("telprompt:") || s.hasPrefix("maps:")
                || s.contains("maps.apple.com") || s.contains("maps.google.com")
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            if let url = navigationAction.request.url, isExternal(url) {
                UIApplication.shared.open(url, options: [:], completionHandler: nil)
                decisionHandler(.cancel)
                return
            }
            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
            if let url = navigationAction.request.url {
                if isExternal(url) {
                    UIApplication.shared.open(url, options: [:], completionHandler: nil)
                } else if navigationAction.targetFrame == nil {
                    webView.load(navigationAction.request)
                }
            }
            return nil
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true
        config.setURLSchemeHandler(context.coordinator.schemeHandler,
                                   forURLScheme: BundleSchemeHandler.scheme)
        config.userContentController.addScriptMessageHandler(
            context.coordinator.visionBridge,
            contentWorld: .page,
            name: VisionBridge.name
        )

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.isOpaque = false
        webView.backgroundColor = .systemBackground
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator

        let url = URL(string: "\(BundleSchemeHandler.scheme)://app/\(fileName).html")
            ?? BundleSchemeHandler.indexURL
        webView.load(URLRequest(url: url))

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
    }
}

struct ContentView: View {
    var body: some View {
        HTMLWebView(fileName: "index")
            .ignoresSafeArea()
    }
}

#Preview {
    ContentView()
}
