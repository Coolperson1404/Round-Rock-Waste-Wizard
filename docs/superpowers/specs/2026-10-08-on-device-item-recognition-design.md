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
| **Your own iPhone photos** in `ml/data/own-test-unlisted/<bin>/` | **The open-set test.** Items outside the taxonomy that look like listed items, filed by their real bin (e.g. a motor-oil bottle under `hhw`). Test only, never trained on |

Downloading public datasets needs your explicit OK each time.

**Minimums per class:** 40 training images and 10 test images. A class short
of *training* photos can be left out of training (`--drop-underfilled`), but
that does not make it safe. The item model is a closed-set classifier: shown an
item it was not trained on, it answers with the nearest look-alike, often
confidently, and the existing fallbacks never run. So a dropped class's test
photos stay in the test set and count in every metric. The unlisted look-alikes
are measured there too. A class short only of *test* photos is never dropped:
it keeps its training data and is listed as `undermeasured` in `summary.json`
(`underfilled` lists both kinds). Dropping a train-short `hhw` or `center`
class needs explicit consent (`--allow-drop-hazard`); without it the build
refuses. Each class's training set is capped at 500 images (`maxTrain`). Your
own photos (`ml/data/own/<label>/`) are always kept; the rest of the cap is
filled from the other sources, chosen deterministically by SHA-1, and
`summary.json` lists the classes that were capped (`capped`).

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
  - Top is `not_waste` at ≥ `ITEM_SUGGEST` → a retake hint. The chooser
    still lists the model's top 3 item guesses first.
  - **Hazard safety net:** confidences are softmax probabilities that sum
    to 1, so a hazardous runner-up can never reach `ITEM_SUGGEST` next to an
    answer. Instead: when the top item is not `hhw` and an `hhw` item at rank
    2 or 3 scores ≥ `HAZARD_RUNNER_UP` (0.10), the verdict is an ask
    (confidence clipped below `CONFIRM_BELOW`) and names that item as
    `hazard`. The hazardous item is always one of the candidates.
    `ml/report.py` applies the same rule when run with `--hazard-net` (its
    `HAZARD_RUNNER_UP` is tested equal to the app's), so a photo the app
    would ask about is not counted as answered. The net is opt-in and used
    only for item-model runs: the material fallback has no such net, so
    material and baseline reports are scored without it.
  - Anything else → no opinion, and the scan falls through.
- **`ITEM_ACCEPT`** is not guessed. It is the smallest grid threshold at or
  above `CONFIRM_BELOW` (0.60) at which, on the test set, at least one photo
  is answered and at most **2 %** of answered photos are in the wrong bin. The
  app ships exactly that threshold, so the rate there is what users see; a
  worse rate at a higher, unshipped threshold does not block it.
- When the item model ran on iOS, MobileNet answers are capped to 0.59. They
  become a suggestion in the chooser rather than an assertion. This targets
  the confident-wrong-answer problem directly.
- The chooser candidates and the correction key are separate signals. The
  chooser always gets the item model's top 3 guesses when it ran, whatever
  the verdict. The correction key is `itm:<top item>`, set only when the
  verdict is answer or ask (never for `not_waste` or no opinion).
- When the item model ran, the material-model fallback is capped to 0.59 as
  well, like MobileNet.
- Corrections and visual memory (`deliverScanResult`) still run first on
  every result. User fixes still win. Like MobileNet's `cls:` key, `itm:` is
  broad: taught once, it overrides only when the pipeline is unsure (< 0.70)
  and is asked about below `CONFIRM_BELOW`. Taught twice, it is used outright,
  with one exception: a broad key is never asserted when the incoming result
  is `hhw` and the learned item is not, or when the result is a
  hazard-downgraded ask. The app asks instead ("Is this the same one
  again?"), listing the learned item, the incoming item and the hazardous
  item. A hazardous incoming result always contributes a hazardous
  candidate, even when it carries only a display name such as "Battery
  item". Specific keys (`bc:`, `txt:`) and visual memory are unchanged.

### 4. Measuring "smarter"

`ml/report.py` scores raw predictions against the own-photo test set:

- top-1 item accuracy
- bin accuracy
- a threshold sweep showing coverage (the share of photos answered without
  asking) against the wrong-bin rate among answered photos
- hazardous misses at the shipped threshold: `hhw` photos answered into
  another bin at `ITEM_ACCEPT`. This gate runs whenever `ITEM_ACCEPT` can be
  computed: any miss exits 3 unless `--allow-hazard-misses` is passed, which
  prints a loud warning. Drop-off (`center`) misses are reported alongside,
  for information only.
- the open set: how many unlisted look-alikes are answered at `ITEM_ACCEPT`,
  and how many of those land in the wrong bin. Unlisted photos count toward
  bin accuracy, the wrong-bin rate and hazardous misses. They do not count
  toward item accuracy or coverage, which are about the known items. When
  coverage is a required target, at least **5** `unlisted__hhw` photos are
  needed (the report records `open_set.hhw_n`); fewer exits 1 unless
  `--allow-no-open-set` is passed.

The same test photos are run through the **current** `WasteMaterial.mlmodel`
as the baseline. A 20-photo device checklist is recorded before and after.

**Ship targets for v1**, on the own-photo test set:

| Metric | Target |
|---|---|
| Item top-1 accuracy | ≥ 80 % |
| Bin accuracy | ≥ 92 % |
| Wrong bin among answered photos | ≤ 2 % (enforced by `ITEM_ACCEPT`) |
| Coverage at `ITEM_ACCEPT` | ≥ 65 % |
| Hazardous (`hhw`) photos answered into another bin at `ITEM_ACCEPT` | 0 |

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
