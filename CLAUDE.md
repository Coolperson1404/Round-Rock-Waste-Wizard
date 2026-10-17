## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### Known gap in this repo (verified 2026-08-22)

graphify extracts the Swift well — `VisionBridge`, `BundleSchemeHandler`,
`HTMLWebView` and their methods all resolve with accurate line numbers, so
`explain`/`path`/`query` are genuinely useful for the native layer.

It extracts **no JavaScript**, because essentially the whole web app lives in
inline `<script>` blocks inside `www/index.html` (~5,000 lines) and the
extractor does not parse script tags in HTML. Verified missing from the graph:
`handlePhoto`, `matchOnDevicePrediction`, `readPackagingText`, `deliverScanResult`,
`analyzeImageMaterialSignatures`, `Corrections`, `VisualMemory`, `callClaudeVision`.
`graphify query "how does the app classify a photo"` returns the SwiftUI entry
point and `Scene` — nothing about classification.

So, for this repo:
- Native/Swift questions → use graphify first, as above.
- Anything about scanning, classification, the item rules, the correction
  system or the cloud calls → **grep `www/index.html` directly**; the graph does
  not contain it and will point the wrong way.

`.graphifyignore` excludes the vendored minified bundles (`tf.min.js`,
`mobilenet.min.js`, `leaflet.js`), the bundled model weights, the generated
asset catalogues, and the two byte-identical copies of `index.html` that
`scripts/sync-targets.sh` maintains. Without it the graph was 882 nodes whose
top community hubs were mangled minifier symbols (`zxe`, `D8`, `hde`); with it
the graph is 114 nodes of real code.
