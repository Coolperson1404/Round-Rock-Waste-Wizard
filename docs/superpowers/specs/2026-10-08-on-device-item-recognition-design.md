# On-device item recognition — design

Date: 2026-10-08 · Status: draft for review

## Problem

Scanning an everyday object with no barcode and no readable label goes wrong
in two ways:

1. **Confident wrong answers** — the wrong item, and often the wrong bin.
2. **No answer** — the scan lands on "What did you scan?" too often.

Both come from the same stage. After barcode and packaging text, the main
visual step is MobileNet v2 (ImageNet, 1,000 generic classes) plus
hand-written rules (`IMAGENET_TO_ROUND_ROCK`, `matchOnDevicePrediction`) that
map ImageNet labels to our 116 items. That model was never trained on waste.
The purpose-trained `WasteMaterial.mlmodel` has only 6 classes and only runs
after MobileNet has already failed. The 10-class, ~12,000-photo
`Garbage Dataset/` is not used anywhere.

## Decisions (from brainstorming)

| Question | Answer |
|---|---|
| What fails | Wrong bin / wrong item, and failing to recognise anything |
| Cloud allowed for recognition? | **No — fully on-device.** The existing opt-in cloud mode is left as-is, not extended |
| Platforms | **iPhone app only.** Browser and Android keep today's behaviour |
| Bar for "smart" | **Right bin and right item** |
| Approach | **A** — a Create ML item classifier, run through Apple Vision |

## Design

### 1. Two Core ML models, both trained with Create ML

Both models are transfer-learned on Apple's built-in `scenePrint` (revision 2)
feature extractor with a logistic-regression head. Each file stays small
(well under 1 MB) and runs on the Neural Engine through `VNCoreMLRequest`, the
same way the material model runs today.

- **`WasteItems.mlmodel`** (new). This is the item classifier: 41 everyday item
  classes plus `not_waste`. Each item class maps to exactly one `ITEMS` entry.
  Items that look alike and go in the same bin are merged into one class (a
  cereal box answers as "Cardboard box"). Items that look alike but go in
  *different* bins get separate classes so the model learns the difference:
  pizza box vs cardboard box, broken glass vs glass bottle, foam vs cardboard
  egg carton, aerosol can vs aluminum can. The full taxonomy lives in
  `ml/sources.json`.
- **`WasteMaterial.mlmodel`** (retrained). This is the fallback, retrained
  from 6 classes to all 10 Garbage Dataset classes, which adds battery,
  biological, clothes and shoes.

### 2. Training data

| Source | Used for |
|---|---|
| `Garbage Dataset/standardized_384` (local) | The whole material model. For the item model: battery, food scraps, clothing (clothes + shoes), cardboard box, paper |
| TACO litter dataset (COCO-format boxes, cropped to one item per image) | Most packaging classes |
| **Your own iPhone photos** in `ml/data/own/<label>/` | Every class. This is the only source for 12 classes (e.g. milk jug, K-cup, foil tin, paint can, light bulbs, `not_waste`) |
| **Your own iPhone photos** in `ml/data/own-test/<label>/` | **The item test set.** It is never trained on, and shows different physical objects from the training photos |

Downloading public datasets needs your explicit OK each time.

**Minimums per class:** 40 training images and 10 test images. A class that
falls short is left out of the build and does not ship. It keeps working
through the existing fallbacks.

Realistic photo cost for v1 is about **900 phone photos**: 40 training photos
for each of the 12 own-photo-only classes, plus 10 test photos for each of the
42 classes. Classes can be added in batches.

### 3. Pipeline order in `handlePhoto` (iOS)

```
Barcode → packaging text            (unchanged, most reliable)
Cloud mode, if the user opted in    (unchanged)
NEW  Item model                     answer, or ask with model candidates, or "not_waste" retake hint
     MobileNet + rules              still runs when the item model has no opinion; capped to "ask" on iOS
     Material model (10 classes)    unchanged position, more classes
     Chooser                        candidates now led by the item model's top 3
```

- The native Vision pass gains an `item` op. It returns the top 5
  `{value, confidence}` as `items`.
- **Decision rule** (in `www/waste-models.js`, a pure, unit-tested module):
  - Top item confidence ≥ `ITEM_ACCEPT` → **answer**.
  - Between `ITEM_SUGGEST` (0.35) and `ITEM_ACCEPT` → **ask**: the existing
    "Which one is it?" chooser, with the model's top 3 listed first.
  - Top is `not_waste` at ≥ `ITEM_SUGGEST` → a retake hint.
  - Anything else → no opinion, and the scan falls through.
- **`ITEM_ACCEPT`** is not guessed. It is the smallest threshold at which,
  on the test set, at most **2 %** of asserted answers are in the wrong bin.
  It is then clamped to at least `CONFIRM_BELOW` (0.60), so the app's
  existing ask-below-0.60 logic stays consistent.
- When the item model ran on iOS, MobileNet answers are capped to 0.59. They
  become a suggestion in the chooser rather than an assertion. This targets
  the confident-wrong-answer problem directly.
- Corrections and visual memory (`deliverScanResult`) still run first on
  every result. User fixes still win.

### 4. Measuring "smarter"

`ml/report.py` scores raw predictions against the own-photo test set:

- top-1 item accuracy
- bin accuracy
- a threshold sweep showing coverage (the share of photos answered without
  asking) against the wrong-bin rate among answered photos

The same test photos are run through the **current** `WasteMaterial.mlmodel`
as the baseline. A 20-photo device checklist is recorded before and after.

**Ship targets for v1**, on the own-photo test set:

| Metric | Target |
|---|---|
| Item top-1 accuracy | ≥ 80 % |
| Bin accuracy | ≥ 92 % |
| Wrong bin among answered photos | ≤ 2 % (enforced by `ITEM_ACCEPT`) |
| Coverage at `ITEM_ACCEPT` | ≥ 65 % |

**Material model target:** ≥ 90 % top-1 on a hash-split 15 % holdout of the
Garbage Dataset.

If a target is missed, the model does not ship. We add photos for the worst
classes (listed in the report) and retrain.

### 5. Error handling

- **No model in the bundle, or the model fails to load:** the Swift side
  reports `errors.item` and returns no `items`. The JS side treats that as
  "item model did not run", and today's pipeline runs unchanged, including an
  uncapped MobileNet.
- **The model emits a label the app has no mapping for** (the model and app
  are out of sync): that prediction is skipped and the next one is considered.
  It never throws.
- **Simulator:** Vision may lack a Core ML inference context there. Behaviour
  is the same as "no model".

## Out of scope

- Browser and Android builds.
- Object detection (several items in one photo).
- Extending the cloud path.
- The other ~75 `ITEMS` entries. Rare items keep using MobileNet, the material
  model, the chooser and correction memory.
- An in-app photo-collection tool.
