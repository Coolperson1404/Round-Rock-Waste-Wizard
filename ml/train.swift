// Train an image classifier with Create ML from labelled folders.
//
//   swift ml/train.swift --train <dir> --model-out <path>.mlmodel --iterations <n>
//
// <dir> holds one subfolder per class (as written by ml/prepare_data.py).
// Writes the model and <path>.labels.txt (sorted class names, one per line),
// and prints training / validation accuracy (1 - classificationError).
//
// Training runs as a resumable Create ML session in <path>.session/ (the one-shot
// MLImageClassifier initializer crashes with "Failed to create CVPixelBufferPool"
// past ~7k images). If interrupted or failed, re-running with the same images and
// iterations resumes from the session; it is deleted once the model is written.
import Combine
import CreateML
import CryptoKit
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

setvbuf(stdout, nil, _IOLBF, 0)

// Bump when ModelParameters below change, so stale sessions are not resumed.
let parametersVersion = "params-v1:split=automatic;aug=crop,rotation,blur,exposure,flip;scenePrint=2;logistic"

var trainPath: String?
var modelOut: String?
var iterations = 25

var args = CommandLine.arguments.dropFirst().makeIterator()
while let flag = args.next() {
    switch flag {
    case "--train": trainPath = args.next()
    case "--model-out": modelOut = args.next()
    case "--iterations":
        guard let raw = args.next(), let n = Int(raw), n > 0 else { fail("--iterations needs a positive integer") }
        iterations = n
    default: fail("unknown argument: \(flag)")
    }
}
guard let trainPath = trainPath, let modelOut = modelOut else {
    fail("usage: swift ml/train.swift --train <dir> --model-out <path>.mlmodel --iterations <n>")
}
guard modelOut.hasSuffix(".mlmodel") else { fail("--model-out must end in .mlmodel") }

let trainURL = URL(fileURLWithPath: trainPath)
let modelURL = URL(fileURLWithPath: modelOut)
let labelsURL = URL(fileURLWithPath: String(modelOut.dropLast(".mlmodel".count)) + ".labels.txt")
let sessionURL = URL(fileURLWithPath: String(modelOut.dropLast(".mlmodel".count)) + ".session")

let fm = FileManager.default
let entries = (try? fm.contentsOfDirectory(at: trainURL, includingPropertiesForKeys: [.isDirectoryKey])) ?? []
let labels = entries
    .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
    .map { $0.lastPathComponent }
    .filter { !$0.hasPrefix(".") }
    .sorted()
guard labels.count >= 2 else { fail("need at least 2 class folders in \(trainPath), found \(labels.count)") }

// Create ML only auto-splits validation data from 50+ training images; below that
// validationMetrics reports classificationError 1.0, which is not a real score.
// The same listing feeds the session fingerprint: relative path | size | mtime.
let imageExts: Set<String> = ["jpg", "jpeg", "png", "heic", "bmp", "webp"]
var imageCount = 0
var fingerprintLines: [String] = []
if let subpaths = try? fm.subpathsOfDirectory(atPath: trainURL.path) {
    for rel in subpaths where imageExts.contains((rel as NSString).pathExtension.lowercased()) {
        imageCount += 1
        let attrs = try? fm.attributesOfItem(atPath: trainURL.appendingPathComponent(rel).path)
        let size = (attrs?[.size] as? NSNumber)?.int64Value ?? -1
        let mtime = (attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        fingerprintLines.append("\(rel)|\(size)|\(String(format: "%.3f", mtime))")
    }
}
fingerprintLines.sort()
let fingerprintInput = fingerprintLines.joined(separator: "\n") + "\niterations=\(iterations)\n\(parametersVersion)\n"
let fingerprint = SHA256.hash(data: Data(fingerprintInput.utf8)).map { String(format: "%02x", $0) }.joined()

do {
    try fm.createDirectory(at: modelURL.deletingLastPathComponent(), withIntermediateDirectories: true)
} catch {
    fail("cannot create \(modelURL.deletingLastPathComponent().path): \(error.localizedDescription)")
}

let fingerprintURL = sessionURL.appendingPathComponent("fingerprint.txt")
let existingFingerprint = (try? String(contentsOf: fingerprintURL, encoding: .utf8))?
    .trimmingCharacters(in: .whitespacesAndNewlines)
do {
    if fm.fileExists(atPath: sessionURL.path), existingFingerprint == fingerprint {
        print("Resuming session in \(sessionURL.path)")
    } else {
        if fm.fileExists(atPath: sessionURL.path) { try fm.removeItem(at: sessionURL) }
        try fm.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        try (fingerprint + "\n").write(to: fingerprintURL, atomically: true, encoding: .utf8)
    }
} catch {
    fail("cannot prepare session directory \(sessionURL.path): \(error.localizedDescription)")
}

let parameters = MLImageClassifier.ModelParameters(
    validation: .split(strategy: .automatic),
    maxIterations: iterations,
    augmentation: [.crop, .rotation, .blur, .exposure, .flip],
    algorithm: .transferLearning(featureExtractor: .scenePrint(revision: 2), classifier: .logisticRegressor)
)
let sessionParameters = MLTrainingSessionParameters(
    sessionDirectory: sessionURL,
    reportInterval: 500,
    checkpointInterval: 500,
    iterations: iterations
)

print("Training \(labels.count) classes from \(trainPath) (\(iterations) iterations)")
let job: MLJob<MLImageClassifier>
do {
    job = try MLImageClassifier.train(
        trainingData: MLImageClassifier.DataSource.labeledDirectories(at: trainURL),
        parameters: parameters,
        sessionParameters: sessionParameters
    )
} catch {
    fail("training failed: \(error.localizedDescription)\n(session kept at \(sessionURL.path); re-run to resume)")
}

func accuracy(_ metrics: MLClassifierMetrics) -> String {
    String(format: "%.4f", 1.0 - metrics.classificationError)
}

var subscriptions = Set<AnyCancellable>()
var finished = false

job.progress.publisher(for: \.fractionCompleted)
    .throttle(for: .seconds(10), scheduler: RunLoop.main, latest: true)
    .sink { fraction in
        let phase = (job.progress.userInfo[ProgressUserInfoKey("phase")] as? MLPhase).map { String(describing: $0) } ?? "starting"
        print("progress: \(phase) \(String(format: "%.1f", fraction * 100))%")
    }
    .store(in: &subscriptions)

job.result.sink(
    receiveCompletion: { completion in
        if case .failure(let error) = completion {
            fail("training failed: \(error.localizedDescription)\n(session kept at \(sessionURL.path); re-run to resume)")
        }
    },
    receiveValue: { classifier in
        let isoDate = ISO8601DateFormatter()
        isoDate.formatOptions = [.withFullDate]
        let metadata = MLModelMetadata(
            author: "Round Rock Waste Wizard",
            shortDescription: "Waste item classifier (\(labels.count) classes)",
            version: isoDate.string(from: Date())
        )
        do {
            try classifier.write(to: modelURL, metadata: metadata)
            try (labels.joined(separator: "\n") + "\n").write(to: labelsURL, atomically: true, encoding: .utf8)
        } catch {
            fail("could not write the model: \(error.localizedDescription)\n(session kept at \(sessionURL.path); re-run to resume)")
        }
        print("training accuracy: \(accuracy(classifier.trainingMetrics))")
        if imageCount >= 50 {
            print("validation accuracy: \(accuracy(classifier.validationMetrics))")
        } else {
            print("validation accuracy: n/a (\(imageCount) training images; Create ML needs 50 to hold out a validation split)")
        }
        print("wrote \(modelURL.path)")
        print("wrote \(labelsURL.path)")
        try? fm.removeItem(at: sessionURL)
        finished = true
    }
).store(in: &subscriptions)

while !finished { RunLoop.main.run(until: Date().addingTimeInterval(1)) }
