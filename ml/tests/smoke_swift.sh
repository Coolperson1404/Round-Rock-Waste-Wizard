#!/usr/bin/env bash
# End-to-end smoke test for ml/train.swift and ml/predict.swift on a tiny 2-class slice.
# Not part of `npm test` (needs macOS + Create ML). Writes only under a temp dir.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

DATA="$WORK/smoke"
MODEL="$WORK/model/Smoke.mlmodel"
LABELS="$WORK/model/Smoke.labels.txt"
PREDS="$WORK/predictions.json"
mkdir -p "$WORK/model"

echo "== prepare"
python3 "$REPO/ml/prepare_data.py" --config "$REPO/ml/tests/fixtures/smoke.json" \
  --out "$DATA" --root "$REPO"

echo "== prune to 20 train / 5 test per class"
prune() { # dir, keep
  local dir="$1" keep="$2"
  ls "$dir" | sort | tail -n +"$((keep + 1))" | while IFS= read -r f; do rm -f "$dir/$f"; done
}
for label in glass paper; do
  prune "$DATA/train/$label" 20
  prune "$DATA/test/$label" 5
done

SESSION="$WORK/model/Smoke.session"

echo "== train (with a stale session dir whose fingerprint does not match)"
mkdir -p "$SESSION"
echo "junk-fingerprint" > "$SESSION/fingerprint.txt"
swift "$REPO/ml/train.swift" --train "$DATA/train" --model-out "$MODEL" --iterations 5

echo "== predict"
swift "$REPO/ml/predict.swift" --model "$MODEL" --test "$DATA/test" --out "$PREDS"

echo "== assert"
python3 -c '
import json, sys
preds = json.load(open(sys.argv[1]))
assert len(preds) == 10, f"expected 10 predictions, got {len(preds)}"
for p in preds:
    assert len(p["top"]) == 2, p
    assert p["truth"] in ("glass", "paper"), p["truth"]
labels = open(sys.argv[2]).read()
assert labels == "glass\npaper\n", repr(labels)
' "$PREDS" "$LABELS"
[ ! -e "$SESSION" ] || { echo "session dir should be deleted after a successful train: $SESSION" >&2; exit 1; }

echo "== refuse unsafe session dirs (exit 1, nothing deleted)"
expect_refusal() { # description, expected-substring, then the train args
  local what="$1" want="$2"; shift 2
  local err rc=0
  err="$(swift "$REPO/ml/train.swift" "$@" 2>&1 >/dev/null)" || rc=$?
  [ "$rc" -eq 1 ] || { echo "$what: expected exit 1, got $rc" >&2; exit 1; }
  case "$err" in *"$want"*) ;; *) echo "$what: stderr lacks '$want': $err" >&2; exit 1 ;; esac
}
# A pre-existing, non-empty session dir without fingerprint.txt is not ours to delete.
mkdir -p "$WORK/foreign/Foreign.session"
echo keep > "$WORK/foreign/Foreign.session/precious.txt"
expect_refusal "foreign session dir" "not a training session" \
  --train "$DATA/train" --model-out "$WORK/foreign/Foreign.mlmodel" --iterations 5
[ -f "$WORK/foreign/Foreign.session/precious.txt" ] || { echo "foreign session dir was modified" >&2; exit 1; }
# A session dir inside the training dir would be scanned as images and could delete them.
expect_refusal "session inside train dir" "overlaps the training directory" \
  --train "$DATA/train" --model-out "$DATA/train/glass/Inside.mlmodel" --iterations 5
[ ! -e "$DATA/train/glass/Inside.session" ] || { echo "session was created inside the train dir" >&2; exit 1; }

echo "SMOKE OK"
