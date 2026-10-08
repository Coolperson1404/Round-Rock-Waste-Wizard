// Run a Core ML classifier through Vision over a labelled test folder and dump raw predictions.
//
//   swift ml/predict.swift --model <path>.mlmodel --test <dir> --out <predictions.json> [--allow-skips]
//
// <dir> holds one subfolder per class; the parent folder name is the truth label.
// Output: [{"file", "truth", "top": [{"label", "confidence"}]}] with the top 5 per image.
import CoreML
import Foundation
import Vision

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

var modelPath: String?
var testPath: String?
var outPath: String?
var allowSkips = false

var args = CommandLine.arguments.dropFirst().makeIterator()
while let flag = args.next() {
    switch flag {
    case "--model": modelPath = args.next()
    case "--test": testPath = args.next()
    case "--out": outPath = args.next()
    case "--allow-skips": allowSkips = true
    default: fail("unknown argument: \(flag)")
    }
}
guard let modelPath = modelPath, let testPath = testPath, let outPath = outPath else {
    fail("usage: swift ml/predict.swift --model <path>.mlmodel --test <dir> --out <predictions.json> [--allow-skips]")
}

let imageExts: Set<String> = ["jpg", "jpeg", "png", "heic", "bmp", "webp"]
let fm = FileManager.default
let testURL = URL(fileURLWithPath: testPath)

let compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: compiledURL))

var isDir: ObjCBool = false
guard fm.fileExists(atPath: testPath, isDirectory: &isDir), isDir.boolValue else {
    fail("test directory not found: \(testPath)")
}

let classDirs = ((try? fm.contentsOfDirectory(at: testURL, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
    .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
    .filter { !$0.lastPathComponent.hasPrefix(".") }
    .sorted { $0.lastPathComponent < $1.lastPathComponent }

guard !classDirs.isEmpty else { fail("no class folders in test directory: \(testPath)") }

var results: [[String: Any]] = []
var skipped = 0
for dir in classDirs {
    let truth = dir.lastPathComponent
    let files = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { imageExts.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    for file in files {
        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .centerCrop // same as the app
        let handler = VNImageRequestHandler(url: file, orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            FileHandle.standardError.write(Data("skip \(file.path): \(error)\n".utf8))
            skipped += 1
            continue
        }
        let observations = (request.results as? [VNClassificationObservation]) ?? []
        let top = observations
            .sorted { $0.confidence > $1.confidence }
            .prefix(5)
            .map { ["label": $0.identifier, "confidence": Double($0.confidence)] as [String: Any] }
        results.append(["file": file.path, "truth": truth, "top": top])
    }
}

if skipped > 0 { print("skipped \(skipped) unreadable image(s)") }
guard !results.isEmpty else { fail("no predictions produced from \(testPath)") }
if skipped > 0 && !allowSkips {
    fail("refusing to write predictions: \(skipped) image(s) skipped (pass --allow-skips to accept)")
}

let outURL = URL(fileURLWithPath: outPath)
try fm.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
try data.write(to: outURL)
print("wrote \(results.count) predictions to \(outURL.path)")
