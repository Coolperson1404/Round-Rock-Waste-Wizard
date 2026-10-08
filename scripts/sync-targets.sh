#!/usr/bin/env bash
# Copies the web app from www/ (the single source of truth) into every target.
# Run after editing www/index.html or refreshing the bundled model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT="$ROOT/Waste Wizard - Round Rock/Waste Wizard - Round Rock"

# tf.min.js / mobilenet.min.js / the model are loaded by relative path and must
# sit flat beside index.html — Xcode flattens bundled resources, so a
# subdirectory would break every path at runtime.
ASSETS=(tf.min.js mobilenet.min.js mobilenet-model.json
        group1-shard1of4.bin group1-shard2of4.bin
        group1-shard3of4.bin group1-shard4of4.bin)

cp "$ROOT/www/index.html" "$ROOT/RoundRockWasteWizard_5.html"
cp "$ROOT/www/index.html" "$SWIFT/index.html"

for f in "${ASSETS[@]}"; do
  cp "$ROOT/www/$f" "$ROOT/$f"
  cp "$ROOT/www/$f" "$SWIFT/$f"
done

# Capacitor's ios/android public dirs are generated, and are gitignored.
( cd "$ROOT" && npx cap copy )

echo "Synced www/ -> standalone, SwiftUI app, iOS and Android."
