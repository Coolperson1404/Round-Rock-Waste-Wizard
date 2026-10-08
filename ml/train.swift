// Train an image classifier with Create ML from labelled folders.
//
//   swift ml/train.swift --train <dir> --model-out <path>.mlmodel --iterations <n>
//
// <dir> holds one subfolder per class (as written by ml/prepare_data.py).
// Writes the model and <path>.labels.txt (sorted class names, one per line),
// and prints training / validation accuracy (1 - classificationError).
import CreateML
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

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
let imageExts: Set<String> = ["jpg", "jpeg", "png", "heic", "bmp", "webp"]
var imageCount = 0
if let walker = fm.enumerator(at: trainURL, includingPropertiesForKeys: nil) {
    for case let url as URL in walker where imageExts.contains(url.pathExtension.lowercased()) {
        imageCount += 1
    }
}

try fm.createDirectory(at: modelURL.deletingLastPathComponent(), withIntermediateDirectories: true)

let parameters = MLImageClassifier.ModelParameters(
    validation: .split(strategy: .automatic),
    maxIterations: iterations,
    augmentation: [.crop, .rotation, .blur, .exposure, .flip],
    algorithm: .transferLearning(featureExtractor: .scenePrint(revision: 2), classifier: .logisticRegressor)
)

print("Training \(labels.count) classes from \(trainPath) (\(iterations) iterations)")
let classifier = try MLImageClassifier(
    trainingData: MLImageClassifier.DataSource.labeledDirectories(at: trainURL),
    parameters: parameters
)

let isoDate = ISO8601DateFormatter()
isoDate.formatOptions = [.withFullDate]
let metadata = MLModelMetadata(
    author: "Round Rock Waste Wizard",
    shortDescription: "Waste item classifier (\(labels.count) classes)",
    version: isoDate.string(from: Date())
)
try classifier.write(to: modelURL, metadata: metadata)
try (labels.joined(separator: "\n") + "\n").write(to: labelsURL, atomically: true, encoding: .utf8)

func accuracy(_ metrics: MLClassifierMetrics) -> String {
    String(format: "%.4f", 1.0 - metrics.classificationError)
}
print("training accuracy: \(accuracy(classifier.trainingMetrics))")
if imageCount >= 50 {
    print("validation accuracy: \(accuracy(classifier.validationMetrics))")
} else {
    print("validation accuracy: n/a (\(imageCount) training images; Create ML needs 50 to hold out a validation split)")
}
print("wrote \(modelURL.path)")
print("wrote \(labelsURL.path)")
