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

// The model configuration. Both the ModelParameters and the session fingerprint are
// built from these constants, so a changed setting can never resume a stale session.
// String(describing:) loses information (every scenePrint revision prints "ScenePrint"),
// so the fingerprint uses explicit names that sit next to the values they describe.
let validationStrategy = MLImageClassifier.ModelParameters.ValidationData.split(strategy: .automatic)
let validationName = "split-automatic"
let augmentations: [(name: String, option: MLImageClassifier.ImageAugmentationOptions)] = [
    ("crop", .crop), ("rotation", .rotation), ("blur", .blur), ("exposure", .exposure), ("flip", .flip),
]
let augmentationOptions = augmentations.reduce(into: MLImageClassifier.ImageAugmentationOptions()) { $0.formUnion($1.option) }
let scenePrintRevision = 2
let featureExtractor = MLImageClassifier.FeatureExtractorType.scenePrint(revision: scenePrintRevision)
let classifierKind = MLImageClassifier.ModelParameters.ClassifierType.logisticRegressor
let classifierName = "logisticRegressor"

/// The parameter component of the session fingerprint. `revision` is a parameter only so
/// `--print-fingerprint` can show that different revisions yield different fingerprints.
func parametersDescription(scenePrintRevision revision: Int = scenePrintRevision) -> String {
    [
        "validation=\(validationName)",
        "augmentation=\(augmentations.map { $0.name }.joined(separator: ","))",
        "featureExtractor=scenePrint-r\(revision)",
        "classifier=\(classifierName)",
    ].joined(separator: ";")
}

var trainPath: String?
var modelOut: String?
var iterations = 25
var printFingerprint = false  // debug aid: print the session fingerprint and exit

var args = CommandLine.arguments.dropFirst().makeIterator()
while let flag = args.next() {
    switch flag {
    case "--train": trainPath = args.next()
    case "--model-out": modelOut = args.next()
    case "--print-fingerprint": printFingerprint = true
    case "--iterations":
        guard let raw = args.next(), let n = Int(raw), n > 0 else { fail("--iterations needs a positive integer") }
        iterations = n
    default: fail("unknown argument: \(flag)")
    }
}
guard let trainPath = trainPath, let modelOut = modelOut ?? (printFingerprint ? "unused.mlmodel" : nil) else {
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
func sessionFingerprint(scenePrintRevision revision: Int = scenePrintRevision) -> String {
    let input = fingerprintLines.joined(separator: "\n")
        + "\niterations=\(iterations)\n\(parametersDescription(scenePrintRevision: revision))\n"
    return SHA256.hash(data: Data(input.utf8)).map { String(format: "%02x", $0) }.joined()
}
let fingerprint = sessionFingerprint()

if printFingerprint {
    print("fingerprint: \(fingerprint)")
    print("fingerprint scenePrint-r1: \(sessionFingerprint(scenePrintRevision: 1))")
    print("fingerprint scenePrint-r2: \(sessionFingerprint(scenePrintRevision: 2))")
    exit(0)
}

// Never let the session dir (which may be deleted) overlap the training images.
let sessionKeptNote = "(session kept at \(sessionURL.path); re-run to resume, or delete it to force a fresh start)"
do {
    // resolvingSymlinksInPath() skips a path whose last component does not exist yet (a new
    // session dir), so resolve the deepest existing ancestor and re-append the rest.
    func canonicalPath(_ url: URL) -> String {
        var existing = url.standardizedFileURL
        var rest: [String] = []
        while !fm.fileExists(atPath: existing.path), existing.path != "/" {
            rest.insert(existing.lastPathComponent, at: 0)
            existing = existing.deletingLastPathComponent()
        }
        return rest.reduce(existing.resolvingSymlinksInPath()) { $0.appendingPathComponent($1) }.path
    }
    let train = canonicalPath(trainURL)
    let session = canonicalPath(sessionURL)
    if session == train || session.hasPrefix(train + "/") || train.hasPrefix(session + "/") {
        fail("refusing to use session directory \(sessionURL.path): it overlaps the training directory \(trainPath)")
    }
}

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
        if fm.fileExists(atPath: sessionURL.path) {
            // Only delete a directory that is ours (has a fingerprint) or empty.
            let contents = try fm.contentsOfDirectory(atPath: sessionURL.path)
            if !contents.isEmpty && !contents.contains("fingerprint.txt") {
                fail("refusing to delete \(sessionURL.path): it is not a training session (no fingerprint.txt). Remove or move it, or pick another --model-out")
            }
            try fm.removeItem(at: sessionURL)
        }
        try fm.createDirectory(at: sessionURL, withIntermediateDirectories: true)
        try (fingerprint + "\n").write(to: fingerprintURL, atomically: true, encoding: .utf8)
    }
} catch {
    fail("cannot prepare session directory \(sessionURL.path): \(error.localizedDescription)")
}

let parameters = MLImageClassifier.ModelParameters(
    validation: validationStrategy,
    maxIterations: iterations,
    augmentation: augmentationOptions,
    algorithm: .transferLearning(featureExtractor: featureExtractor, classifier: classifierKind)
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
    fail("training failed: \(error.localizedDescription)\n\(sessionKeptNote)")
}

func accuracy(_ metrics: MLClassifierMetrics) -> String {
    String(format: "%.4f", 1.0 - metrics.classificationError)
}

var subscriptions = Set<AnyCancellable>()

job.progress.publisher(for: \.fractionCompleted)
    .throttle(for: .seconds(10), scheduler: DispatchQueue.main, latest: true)
    .sink { fraction in
        let phase = (job.progress.userInfo[ProgressUserInfoKey("phase")] as? MLPhase).map { String(describing: $0) } ?? "starting"
        print("progress: \(phase) \(String(format: "%.1f", fraction * 100))%")
    }
    .store(in: &subscriptions)

// Everything below runs on the main queue; dispatchMain() parks the main thread
// (no polling) and the sinks exit the process explicitly.
var gotResult = false
job.result.receive(on: DispatchQueue.main).sink(
    receiveCompletion: { completion in
        if case .failure(let error) = completion {
            fail("training failed: \(error.localizedDescription)\n\(sessionKeptNote)")
        }
        if !gotResult { fail("training ended without a result\n\(sessionKeptNote)") }
    },
    receiveValue: { classifier in
        gotResult = true
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
            fail("could not write the model: \(error.localizedDescription)\n\(sessionKeptNote)")
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
        exit(0)
    }
).store(in: &subscriptions)

dispatchMain()
