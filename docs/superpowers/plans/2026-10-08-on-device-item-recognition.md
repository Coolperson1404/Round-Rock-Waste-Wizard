# On-device Item Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace MobileNet-plus-rules as the iPhone app's main visual recogniser with a purpose-trained Core ML item classifier (41 everyday items plus `not_waste`), retrain the material fallback to 10 classes, and prove the gain on a held-out test set of the user's own photos.

**Architecture:**
- **Training** (offline, on the Mac): `ml/prepare_data.py` builds class folders, `ml/train.swift` and `ml/predict.swift` use Create ML and Vision, and `ml/report.py` scores the results.
- **Native app:** the `.mlmodel` files ship in the SwiftUI target, and `VisionBridge` gains an `item` op.
- **Web layer:** a new pure module, `www/waste-models.js`, turns model output into answer / ask / not-waste decisions. `handlePhoto` in `www/index.html` calls it before MobileNet.

**Tech Stack:**
- Create ML: `MLImageClassifier`, transfer learning on `scenePrint(revision: 2)` with a logistic regressor.
- Vision `VNCoreMLRequest`.
- Swift 5 / WKWebView.
- Python 3 with Pillow, tested with stdlib `unittest`.
- Node 26, tested with `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-08-on-device-item-recognition-design.md`

## Global Constraints

- iPhone app only (`Waste Wizard - Round Rock/`). The browser and Capacitor builds must behave exactly as today when `WasteModels` or native Vision is absent.
- Recognition is fully on-device. Add no network calls to the app. Do not change the cloud code path.
- `www/index.html` is the single source of truth. Run `scripts/sync-targets.sh` after editing it. Web assets sit flat beside `index.html` because Xcode flattens bundle resources.
- Constants:
  - `ITEM_SUGGEST = 0.35`
  - `CONFIRM_BELOW = 0.60` (existing; do not change)
  - max wrong-bin rate among answered photos `0.02`
  - `minTrain = 40`, `minTest = 10`
  - prepared images: longest side 384 px
  - hash holdout: 15 %
  - COCO crops: 10 % padding, sides under 64 px dropped
  - Vision crop option: `.centerCrop`
- Ship gates:
  - item model: top-1 ≥ 0.80, bin ≥ 0.92, coverage ≥ 0.65 at the effective accept threshold
  - material model: top-1 ≥ 0.90
- Python uses the stdlib plus Pillow only (`unittest`, no pytest). JS tests use `node --test`. Neither may add dependencies.
- **Downloading any dataset needs the user's explicit OK** (name the source and size first). `ml/data/` and `ml/build/` are gitignored.
- Work on branch `feat/on-device-item-model`. `main` already has unrelated *staged* Xcode files, so every commit names its paths: `git commit -m "…" -- <paths>`.
- After code changes, run `graphify update .` (CLAUDE.md).

## Review Focus

1. **Model and app drift.** The installed model emits a label `ITEM_LABELS` doesn't know, or an `ITEMS` entry is renamed. The expected behaviour is that the unknown label is skipped and a CI test fails on the rename. Pinned in Task 1 (`labels.test.js`) and Task 2 ("skips unknown labels").
2. **No model or no inference context** (Simulator, model missing from the bundle). The expected behaviour is that today's pipeline runs unchanged and MobileNet is *not* capped. Pinned in Task 2 (`itemModelRan`) and Task 9 (Swift reports `errors.item`).
3. **Hazardous item read as recyclable** (aerosol → aluminum can, battery → trash). This is the costliest error. The report counts `hazard_misses` separately. Pinned in Task 5.
4. **Same photo in train and test.** This inflates every metric. The expected behaviour is that prepare fails loudly. Pinned in Task 3 (`test_leak_detected_exits_2`).
5. **Non-numeric or NaN confidence from the bridge.** The expected behaviour is that it's ignored and nothing throws. Pinned in Task 2 ("ignores non-finite confidences").

---

## File Structure

| Path | Responsibility |
|---|---|
| `www/waste-models.js` (new) | Pure interpretation of item and material model output. Label → item maps and thresholds. UMD (`window.WasteModels` / `module.exports`) |
| `ml/sources.json` (new) | Item-model taxonomy: label → bin plus data sources |
| `ml/materials.json` (new) | Material-model classes → bin plus Garbage Dataset sources |
| `ml/prepare_data.py` (new) | Config → `out/{train,test}/<label>/*.jpg`, `summary.json`, leakage check |
| `ml/train.swift` (new) | Create ML training → `.mlmodel` plus `.labels.txt` |
| `ml/predict.swift` (new) | Vision inference over a test dir → `predictions.json` |
| `ml/report.py` (new) | Metrics, threshold sweep, targets gate |
| `ml/PHOTO_GUIDE.md` (new) | How the user takes training and test photos |
| `ml/models/` (new, tracked) | `<Model>.labels.txt` and `<Model>.report.json` for each shipped model |
| `ml/tests/` (new) | `labels.test.js`, `waste-models.test.js`, `helpers/items.js`, `test_prepare_data.py`, `test_report.py`, `smoke_swift.sh`, `fixtures/smoke.json` |
| `Waste Wizard - Round Rock/Waste Wizard - Round Rock/ContentView.swift` | `VisionBridge`: shared model loader and classify helper; new `item` op |
| `Waste Wizard - Round Rock/Waste Wizard - Round Rock/WasteItems.mlmodel` (new) | Shipped item model (auto-included by the synchronized group) |
| `…/WasteMaterial.mlmodel` | Replaced by the 10-class retrain |
| `www/index.html` | Script tag, `NativeVision`, `handlePhoto` item step, MobileNet cap, `buildCandidates`, `matchMaterialPrediction` delegation |
| `scripts/sync-targets.sh`, `.gitignore`, `package.json` | Ship `waste-models.js`; ignore data, build output and asset copies; `npm test` |

---

### Task 1: Taxonomy, label maps and consistency tests

**Files:**
- Create: `ml/sources.json`, `ml/materials.json`, `www/waste-models.js` (constants only), `ml/tests/helpers/items.js`, `ml/tests/labels.test.js`
- Modify: `package.json` (`scripts.test`), `.gitignore`

**Interfaces:**
- Produces: `WasteModels.ITEM_LABELS: {[label]: string|null}` (only `not_waste` is `null`), `WasteModels.NOT_WASTE = 'not_waste'`, `WasteModels.MATERIAL_TO_ITEM: {[label]: {item, name}}`, `WasteModels.MATERIAL_MIN_CONFIDENCE = 0.60`, `WasteModels.ITEM_SUGGEST = 0.35`, `WasteModels.ITEM_ACCEPT = 0.60` (placeholder value; Task 8 sets the real one).
- Produces: `loadItems(): Array<{n: string, cat: string}>` in `ml/tests/helpers/items.js`. It parses the `const ITEMS = [ … ];` block of `www/index.html` with the regex `/\{\s*n:\s*'([^']*)',\s*cat:\s*'([a-z]+)'/g`.
- Produces: config format, shared by Tasks 3 and 5:

```json
{ "model": "WasteItems", "ownPhotos": true, "testPercent": 15, "minTrain": 40, "minTest": 10,
  "classes": { "<label>": { "bin": "<CATS id | none>", "sources": [
    { "type": "folder", "path": "<repo-relative dir>", "split": "train|test|hash" },
    { "type": "coco", "annotations": "ml/data/taco/data/annotations.json",
      "images": "ml/data/taco/data", "categories": ["<TACO name>"], "split": "train" } ] } } }
```

`ownPhotos: true` implicitly adds `ml/data/own/<label>` (train) and `ml/data/own-test/<label>` (test) to every class.

**Item taxonomy.** Copy these into `ml/sources.json` and `ITEM_LABELS` exactly. Sources: **G** = `Garbage Dataset/standardized_384/<dir>` (split `train`), **T** = TACO categories (split `train`), and every class also gets own photos.

| label | ITEMS name | bin | G / T sources |
|---|---|---|---|
| cardboard_box | Cardboard box | recycle | G cardboard; T Corrugated carton |
| pizza_box | Pizza box | trash | T Pizza box |
| egg_carton_cardboard | Egg carton (cardboard) | recycle | T Egg carton |
| egg_carton_foam | Egg carton (foam) | center | — |
| paper_tube | Paper towel / toilet paper tube | recycle | T Toilet tube |
| paper | Junk mail, letters, envelopes | recycle | G paper; T Normal paper, Magazine paper |
| paper_bag | Paper bag | recycle | T Paper bag |
| paper_plate | Paper plate | trash | — |
| tissue | Napkin / paper towel / tissue | trash | T Tissues |
| drink_carton | Milk or juice carton | trash | T Drink carton |
| paper_cup | Coffee cup (paper, with lining) | trash | T Paper cup |
| plastic_bottle | Water or soda bottle | recycle | T Clear plastic bottle |
| milk_jug | Milk jug | recycle | — |
| plastic_tub | Yogurt / butter / cream cheese tub | recycle | T Spread tub |
| detergent_bottle | Detergent or cleaning bottle | recycle | T Other plastic bottle |
| plastic_lid | Plastic lid | recycle | T Plastic lid, Plastic bottle cap |
| plastic_bag | Plastic grocery bag | other | T Single-use carrier bag, Polypropylene bag |
| plastic_film | Plastic wrap / bubble wrap | trash | T Plastic film |
| plastic_cup_cutlery | Plastic cutlery, straws, cups | trash | T Disposable plastic cup, Plastic utensils, Plastic straw |
| chip_bag | Chip bag / candy wrapper | trash | T Crisp packet, Other plastic wrapper |
| styrofoam_food | Styrofoam cup or takeout box | center | T Foam cup, Foam food container |
| styrofoam_packaging | Styrofoam packaging (TV, furniture) | center | T Styrofoam piece |
| glass_bottle | Glass bottle (wine, beer, liquor) | recycle | T Glass bottle |
| glass_jar | Pasta sauce jar | recycle | T Glass jar |
| broken_glass | Broken glass | trash | T Broken glass |
| drinking_glass | Drinking glass, Pyrex, ceramic dish | trash | T Glass cup |
| aluminum_can | Aluminum can | recycle | T Drink can |
| food_can | Soup or vegetable can | recycle | T Food Can |
| aluminum_foil | Aluminum foil | recycle | T Aluminium foil |
| foil_tin | Foil baking tin | recycle | — |
| aerosol_can | Aerosol can (hairspray, spray paint) | hhw | T Aerosol |
| paint_can | Paint can | hhw | — |
| reusable_bottle | Reusable bottle, flask or travel mug | center | — |
| battery | Alkaline battery (AA, AAA, 9V) | hhw | G battery; T Battery |
| light_bulb | Regular light bulb / LED | trash | — |
| fluorescent_bulb | Fluorescent light bulb / tube | center | — |
| clothing | Clothing / shoes / textiles | other | G clothes, G shoes; T Shoe |
| food_scraps | Food scraps / leftovers | trash | G biological; T Food waste |
| kcup | K-Cup / coffee pod | trash | — |
| cell_phone | Cell phone | other | — |
| laptop | Computer, laptop, tablet | other | — |
| not_waste | *(null)* | none | — |

**Materials.** `ml/materials.json` has `ownPhotos: false`. Each of the 10 Garbage Dataset dirs is a `folder` source with split `hash`.

| label | bin | MATERIAL_TO_ITEM item | name |
|---|---|---|---|
| battery | hhw | Alkaline battery (AA, AAA, 9V) | Battery |
| biological | trash | Food scraps / leftovers | Food waste |
| cardboard | recycle | Cardboard box | Cardboard |
| clothes | other | Clothing / shoes / textiles | Clothing |
| glass | recycle | Glass bottle (wine, beer, liquor) | Glass |
| metal | recycle | Aluminum can | Metal |
| paper | recycle | Junk mail, letters, envelopes | Paper |
| plastic | recycle | Water or soda bottle | Plastic |
| shoes | other | Clothing / shoes / textiles | Shoes |
| trash | trash | Plastic cutlery, straws, cups | Non-recyclable waste |

The existing six entries keep their current item and name values from `www/index.html:3644-3651`.

- [ ] **Step 1: Write the failing tests** in `ml/tests/labels.test.js`:
  - `every mapped item exists in ITEMS`: each non-null `ITEM_LABELS` value equals some `loadItems()[i].n`.
  - `only not_waste is unmapped`: `Object.entries(ITEM_LABELS).filter(([,v]) => v === null).map(([k]) => k)` deep-equals `['not_waste']`.
  - `sources.json classes match ITEM_LABELS`: sorted keys are equal, and there are 42 of them.
  - `sources.json bin equals the mapped item's cat`: for every label, `bin === ITEMS[n].cat`. For `not_waste`, `bin === 'none'`.
  - `materials.json matches MATERIAL_TO_ITEM`: equal key sets (10). Every `.item` exists in ITEMS. Every material `bin` equals that item's `cat`.
  - `installed item model labels are all mapped`: if `ml/models/WasteItems.labels.txt` exists, every non-empty line is a key of `ITEM_LABELS`. Otherwise `t.skip('no model installed yet')`.
  - `loadItems parses all 116 items`: `loadItems().length === 116`.
- [ ] **Step 2: Run `node --test ml/tests/`.** Expected: FAIL, because `Cannot find module '../../www/waste-models.js'`.
- [ ] **Step 3: Create the two JSON configs and `www/waste-models.js`** with the constants above, wrapped as `(function (root) { const WasteModels = {…}; if (typeof module === 'object' && module.exports) module.exports = WasteModels; else root.WasteModels = WasteModels; })(typeof self !== 'undefined' ? self : this);`.
- [ ] **Step 4: Set up tooling.** Set `package.json` `scripts.test` to `node --test ml/tests/ && python3 -m unittest discover -s ml/tests -p 'test_*.py'`. Append these lines to `.gitignore`:
  - `ml/data/`
  - `ml/build/`
  - `/waste-models.js`
  - `Waste Wizard - Round Rock/Waste Wizard - Round Rock/waste-models.js`
- [ ] **Step 5: Run `node --test ml/tests/`.** Expected: PASS. The installed-model test is SKIP.
- [ ] **Step 6: Commit:** `git add ml/sources.json ml/materials.json www/waste-models.js ml/tests package.json .gitignore && git commit -m "feat(ml): item and material taxonomy with consistency tests" -- ml/sources.json ml/materials.json www/waste-models.js ml/tests package.json .gitignore`

### Task 2: Decision logic in `waste-models.js`

**Files:**
- Modify: `www/waste-models.js`
- Test: `ml/tests/waste-models.test.js`

**Interfaces:**
- Consumes: the Task 1 constants and `loadItems()`.
- Produces:
  - `WasteModels.itemModelRan(native: object|null) -> boolean`. True only when `native.items` is a non-empty array.
  - `WasteModels.interpretItems(predictions: Array<{value, confidence}>, items: ITEMS[], opts?: {confirmBelow=0.60, accept=ITEM_ACCEPT, suggest=ITEM_SUGGEST}) -> ItemVerdict|null`
  - `ItemVerdict = { kind: 'answer'|'ask'|'not_waste', item: ITEMS entry|null, confidence: number, candidates: ITEMS entry[], why: string, badge: string }`. `candidates` holds up to 3 distinct entries, top first.
  - `WasteModels.interpretMaterials(predictions, items) -> {item, match, cat, why, tip, badge, badgeIcon, confidence}|null`. Same return shape and copy as today's `matchMaterialPrediction` (`www/index.html:3656-3674`), using `MATERIAL_TO_ITEM` and `MATERIAL_MIN_CONFIDENCE`.

**Rules for `interpretItems`:**
1. Coerce each confidence with `Number()`. Drop non-finite confidences, and drop labels that aren't in `ITEM_LABELS` or whose item is missing from `items`. Sort what remains descending.
2. If nothing remains, or the top confidence is below `suggest`, return `null`.
3. If the top label is `not_waste`, return `{kind: 'not_waste', item: null, candidates: []}`.
4. Compute `effectiveAccept = Math.max(accept, confirmBelow)`.
5. If the top confidence is ≥ `effectiveAccept`, return `kind: 'answer'` with the raw confidence. Otherwise return `kind: 'ask'` with `confidence = Math.min(top, confirmBelow - 0.01)`.
6. `why` is `` `On-device item model recognised this as ${item.n}.` `` and `badge` is `` `Item model: ${item.n} (${Math.round(conf*100)}%)` ``.

- [ ] **Step 1: Write the failing tests** using `loadItems()`. Every test that expects `kind 'answer'` passes `opts {accept: 0.60}`, so it keeps passing after Task 8 tunes `ITEM_ACCEPT`.
  - `answers a confident can`: `[{value:'aluminum_can',confidence:0.91},{value:'food_can',confidence:0.05}]` gives `kind 'answer'`, `item.n 'Aluminum can'`, `confidence 0.91`, and candidate names `['Aluminum can','Soup or vegetable can']`.
  - `asks when unsure`: `aerosol_can 0.5, aluminum_can 0.4` gives `kind 'ask'`, `item.n 'Aerosol can (hairspray, spray paint)'`, `confidence 0.5`.
  - `clips ask confidence below confirmBelow`: `opts {accept:0.8}` with `aluminum_can 0.7` gives `kind 'ask'` and `confidence 0.59`.
  - `accept is never below confirmBelow`: `opts {accept:0.4}` with `aluminum_can 0.5` gives `kind 'ask'`.
  - `not_waste`: `[{value:'not_waste',confidence:0.8}]` gives `{kind:'not_waste', item:null, candidates:[]}`.
  - `no opinion below suggest`: `aluminum_can 0.2` gives `null`.
  - `skips unknown labels`: `hovercraft 0.9, aluminum_can 0.7` gives `kind 'answer'`, `item.n 'Aluminum can'`.
  - `ignores non-finite confidences`: `[{value:'aluminum_can',confidence:NaN}]` gives `null`. `confidence:'0.9'` gives `kind 'answer'`.
  - `sorts unsorted input`: `food_can 0.1, aluminum_can 0.8` gives `item.n 'Aluminum can'`.
  - `null / [] / non-array` each give `null`.
  - `itemModelRan`: `null` gives false, `{items:[], errors:{item:'model not bundled'}}` gives false, `{items:[{value:'x',confidence:0.1}]}` gives true.
  - `interpretMaterials`: `[{value:'battery',confidence:0.8}]` gives `match 'Alkaline battery (AA, AAA, 9V)'`, `cat 'hhw'`, `badge 'Material: Battery (80%)'`. `0.5` gives `null`. An unknown label gives `null`.
  - `ITEM_ACCEPT is within [0.60, 0.95]`.
- [ ] **Step 2: Run `node --test ml/tests/waste-models.test.js`.** Expected: FAIL, because `interpretItems is not a function`.
- [ ] **Step 3: Implement `itemModelRan`, `interpretItems` and `interpretMaterials`** in `www/waste-models.js` per the rules above.
- [ ] **Step 4: Run `npm test`.** Expected: all JS tests pass. The Python discovery finds 0 tests, which is OK.
- [ ] **Step 5: Commit** `www/waste-models.js` and `ml/tests/waste-models.test.js`: `feat(models): item/material decision logic`.

### Task 3: `prepare_data.py` and the photo guide

**Files:**
- Create: `ml/prepare_data.py`, `ml/PHOTO_GUIDE.md`
- Test: `ml/tests/test_prepare_data.py`

**Interfaces:**
- Consumes: the Task 1 config format.
- Produces: the CLI `python3 ml/prepare_data.py --config <json> --out <dir> [--drop-underfilled] [--check]`. Output layout is `<out>/{train,test}/<label>/<tag>_<stem>[_<annId>].jpg` plus `<out>/summary.json` = `{"counts": {label: {"train": n, "test": n}}, "underfilled": [label…], "unreadable": n}`. Exit codes: `0` ok, `2` leakage, `1` config error. `--check` validates the config (including COCO category names) and writes nothing.
- Functions, all importable for tests:
  - `hash_split(rel_path: str, test_percent: int) -> 'train'|'test'`: SHA-1 of the path, `int(hex, 16) % 100 < test_percent`.
  - `crop_box(img, bbox: [x, y, w, h], pad=0.10) -> Image|None`: clamped to the image; `None` if either side is under 64 px.
  - `collect(config, root: Path) -> list[Sample]`: `Sample` is a namedtuple `(label, split, src: Path, bbox, tag)`. A missing folder contributes 0 samples. A missing COCO file, or an unknown COCO category, raises `ValueError` and lists the available names.
  - `load_image(path) -> Image`: applies `ImageOps.exif_transpose`, converts to RGB, downsizes the longest side to 384 (never upscales). `.heic` files go through `sips -s format jpeg` to a temp file first.
  - `find_leaks(samples) -> list[tuple[Path, Path]]`: pairs of train and test samples whose source file bytes have equal SHA-256.
  - `summarize(counts, min_train, min_test) -> dict`.

- [ ] **Step 1: Write failing tests**, using tiny PIL images in `tempfile` dirs:
  - `test_hash_split_deterministic_and_proportional`: the same input gives the same output. Over `f"img_{i}.jpg"` for 1000 values of i, the test fraction is in `[0.12, 0.18]` at 15.
  - `test_crop_box_pads_and_rejects_small`: a 200×200 image with `[50,50,100,100]` gives size `(120,120)`. `[0,0,30,30]` gives `None`.
  - `test_unknown_coco_category_raises`: a COCO json with category `Drink can` and config category `Soda can` raises `ValueError` whose message contains `Drink can`.
  - `test_own_photos_implicit_and_optional`: files in `ml/data/own/x` and `ml/data/own-test/x` land in train and test respectively, and a class with no dirs gives 0 with no error.
  - `test_exif_orientation_applied`: a 100×50 JPEG with EXIF orientation 6 loads as 50×100.
  - `test_leak_detected_exits_2`: identical bytes in `own/x` and `own-test/x` make `main([...])` return 2.
  - `test_drop_underfilled`: a class with 3 train images under `minTrain=40` is listed in `summary["underfilled"]`, and with `--drop-underfilled` has no output dir.
- [ ] **Step 2: Run `python3 -m unittest ml.tests.test_prepare_data -v`** (add an empty `ml/__init__.py` and `ml/tests/__init__.py` if discovery needs them). Expected: FAIL, because `ModuleNotFoundError`.
- [ ] **Step 3: Implement `ml/prepare_data.py`** with `main(argv) -> int`, and `if __name__ == '__main__': sys.exit(main(sys.argv[1:]))`.
- [ ] **Step 4: Run `npm test`.** Expected: PASS.
- [ ] **Step 5: Write `ml/PHOTO_GUIDE.md`.** It should cover:
  - **Folder layout:** `ml/data/own/<label>/` for training and `ml/data/own-test/<label>/` for testing, with the label list copied from `ml/sources.json`.
  - **Framing:** one item per photo, filling at least half the frame.
  - **Variety:** different counters, floors and hands, day and lamp light, at least 3 angles.
  - **Test photos:** of *different physical objects* from the training photos.
  - **Getting photos onto the Mac:** AirDrop to the Mac; HEIC is fine.
  - **Counts:** 40 training photos for the 12 classes marked "—" in Task 1, at least 10 training photos recommended for every other class, and 10 test photos for every class.
  - **Checking progress:** `python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items` shows what's still short.
- [ ] **Step 6: Commit** `ml/prepare_data.py`, `ml/PHOTO_GUIDE.md`, `ml/tests/test_prepare_data.py` and any `__init__.py`: `feat(ml): dataset preparation with leakage check`.

### Task 4: `train.swift` and `predict.swift`

**Files:**
- Create: `ml/train.swift`, `ml/predict.swift`, `ml/tests/smoke_swift.sh`, `ml/tests/fixtures/smoke.json`

**Interfaces:**
- Consumes: the `<out>/{train,test}` layout from Task 3.
- Produces:
  - `swift ml/train.swift --train <dir> --model-out <path>.mlmodel --iterations <n>` writes the model and `<path>.labels.txt`. The labels file holds the train dir's subdirectory names, sorted, one per line. The script prints `training accuracy` and `validation accuracy` as `1 - classificationError`.
    - Parameters: `ModelParameters(validation: .split(strategy: .automatic), maxIterations: n, augmentation: [.crop, .rotation, .blur, .exposure, .flip], algorithm: .transferLearning(featureExtractor: .scenePrint(revision: 2), classifier: .logisticRegressor))`. This compiles on this Mac (verified 2026-10-08).
    - Metadata: author `Round Rock Waste Wizard`, version the ISO date.
  - `swift ml/predict.swift --model <path>.mlmodel --test <dir> --out <predictions.json>` writes `[{"file", "truth", "top": [{"label", "confidence"}] }]` with the top 5. `truth` is the parent dir name.
    - The script calls `MLModel.compileModel(at:)`, then runs `VNCoreMLRequest` with `imageCropAndScaleOption = .centerCrop` (as in the app) and orientation `.up`.

- [ ] **Step 1: Write `ml/tests/fixtures/smoke.json`.** It has two classes (`glass` and `paper`, folder sources from `Garbage Dataset/standardized_384`, split `hash`), `ownPhotos: false`, `minTrain: 1` and `minTest: 1`.
- [ ] **Step 2: Write `ml/tests/smoke_swift.sh`** (`set -euo pipefail`). It runs these steps in order:
  1. Prepare the fixture into `$TMPDIR/smoke`.
  2. Prune each class to 20 train and 5 test images.
  3. Train with `--iterations 5`.
  4. Predict.
  5. Assert with `python3 -c` that `predictions.json` has 10 entries, each with `len(top) == 2`, and that `labels.txt == "glass\npaper\n"`.
  6. Print `SMOKE OK`.
- [ ] **Step 3: Run `bash ml/tests/smoke_swift.sh`.** Expected: FAIL, because `ml/train.swift` doesn't exist yet.
- [ ] **Step 4: Implement both scripts.**
- [ ] **Step 5: Run `bash ml/tests/smoke_swift.sh`.** Expected: `SMOKE OK` (about 1–3 min).
- [ ] **Step 6: Commit** the four files: `feat(ml): Create ML training and Vision prediction scripts`.

### Task 5: `report.py`

**Files:**
- Create: `ml/report.py`
- Test: `ml/tests/test_report.py`

**Interfaces:**
- Consumes: `predictions.json` (Task 4) and configs (Task 1).
- Produces: the CLI `python3 ml/report.py --predictions <json> --truth-config <json> [--pred-config <json>] [--max-wrong-bin 0.02] [--confirm-below 0.60] [--require item=0.80,bin=0.92,coverage=0.65] --out <report.json>`. It prints a markdown summary and returns these exit codes:
  - `0` if all targets are met
  - `1` if any target is missed
  - `3` if no threshold meets the max wrong-bin rate

  `--pred-config` defaults to `--truth-config`. It is used for baselines whose labels differ.

- Functions:
  - `bins(config) -> dict[label, bin]`
  - `score(preds, truth_bins, pred_bins) -> dict` with keys:
    - `n`
    - `item_top1`, `item_top3`: these are `None` unless the predicted labels are a subset of the truth labels
    - `bin_acc`: `'none' == 'none'` counts as correct
    - `per_class: {label: {n, acc}}`
    - `worst_classes`: the 5 lowest-accuracy labels
    - `confusions`: the top 10 `[truth, pred, count]` triples
    - `hazard_misses`: the count of photos whose truth bin is `hhw` but whose top prediction's bin isn't
  - `THRESHOLDS = [round(0.30 + 0.05*i, 2) for i in range(14)]` (0.30 … 0.95)
  - `sweep(preds, truth_bins, pred_bins) -> list[{t, coverage, wrong_bin_rate, item_acc}]`.
    - A photo counts as accepted when its top confidence is ≥ t and its top label isn't `not_waste`.
    - `coverage` = accepted photos ÷ photos whose truth isn't `not_waste`.
    - `wrong_bin_rate` = accepted photos in the wrong bin ÷ max(1, accepted photos).
  - `recommend_threshold(rows, max_wrong) -> float|None`: the smallest t with at least 1 accepted photo and `wrong_bin_rate <= max_wrong`.
  - `effective_accept = max(recommended, confirm_below)`. Coverage targets are checked at this value.
  - `check_targets(report, require: dict) -> list[str]` returns the names of the missed targets.

- [ ] **Step 1: Write failing tests** with a synthetic truth config (`aluminum_can: recycle`, `aerosol_can: hhw`, `cardboard_box: recycle`, `paper: recycle`, `not_waste: none`):
  - `test_perfect`: 2 correct predictions give `item_top1 1.0` and `bin_acc 1.0`.
  - `test_item_wrong_bin_right`: truth `cardboard_box` with prediction `paper` gives `item_top1 0.0` and `bin_acc 1.0`.
  - `test_sweep_and_recommend`: a correct prediction at 0.9 plus `aerosol_can` predicted as `aluminum_can` at 0.5. At t=0.40 coverage is 1.0 and wrong is 0.5. At t=0.60 coverage is 0.5 and wrong is 0.0. `recommend_threshold(rows, 0.02) == 0.55`.
  - `test_not_waste_never_accepted`: a `not_waste` prediction at 0.99 isn't counted in coverage at any t.
  - `test_hazard_misses`: truth `aerosol_can` with prediction `aluminum_can` gives `hazard_misses == 1`.
  - `test_baseline_material_labels`: with a pred config of `plastic: recycle, metal: recycle`, `item_top1 is None` and `bin_acc` is computed.
  - `test_check_targets`: `{'item': 0.8}` against `item_top1 0.5` gives `['item']`.
- [ ] **Step 2: Run `python3 -m unittest ml.tests.test_report -v`.** Expected: FAIL, because `ModuleNotFoundError`.
- [ ] **Step 3: Implement `ml/report.py`.**
- [ ] **Step 4: Run `npm test`.** Expected: PASS.
- [ ] **Step 5: Commit** `ml/report.py` and `ml/tests/test_report.py`: `feat(ml): evaluation report with bin-safety threshold`.

### Task 6: Retrain the material model (10 classes)

**Files:**
- Create: `ml/baseline/WasteMaterial-6class.mlmodel` (a copy of the current model), `ml/models/WasteMaterial.labels.txt`, `ml/models/WasteMaterial.report.json`
- Modify: `Waste Wizard - Round Rock/Waste Wizard - Round Rock/WasteMaterial.mlmodel` (replace it)

**Interfaces:**
- Consumes: Tasks 3–5 and `ml/materials.json`.
- Produces: a 10-class `WasteMaterial.mlmodel` whose labels match `MATERIAL_TO_ITEM` keys. The `VisionBridge` `material` op is unchanged.

- [ ] **Step 1:** `mkdir -p ml/baseline ml/models && cp "Waste Wizard - Round Rock/Waste Wizard - Round Rock/WasteMaterial.mlmodel" ml/baseline/WasteMaterial-6class.mlmodel`
- [ ] **Step 2:** `python3 ml/prepare_data.py --config ml/materials.json --out ml/build/materials`. Expected: `summary.json` has 10 labels and an empty `underfilled` list.
- [ ] **Step 3:** `swift ml/train.swift --train ml/build/materials/train --model-out ml/build/materials/WasteMaterial.mlmodel --iterations 100`
- [ ] **Step 4:** `swift ml/predict.swift --model ml/build/materials/WasteMaterial.mlmodel --test ml/build/materials/test --out ml/build/materials/predictions.json && python3 ml/report.py --predictions ml/build/materials/predictions.json --truth-config ml/materials.json --require item=0.90 --out ml/build/materials/report.json`. Expected: exit 0 with `item_top1 ≥ 0.90`. **If it exits 1, stop and report the worst classes to the user. Do not install.**
- [ ] **Step 5:** Copy the model into the Swift folder (replacing the old one), and copy `WasteMaterial.labels.txt` and `report.json` (renamed `WasteMaterial.report.json`) into `ml/models/`. Run `npm test`. Expected: PASS.
- [ ] **Step 6: Commit** the baseline copy, the new model and `ml/models/WasteMaterial.*`: `feat(ml): retrain material model on 10 Garbage Dataset classes`.

### Task 7: Collect item data and record the device baseline *(the user does part of this)*

**Files:**
- Create: `docs/superpowers/eval/device-checklist.md`

**Interfaces:**
- Produces: `ml/data/taco/` (gitignored), `ml/data/own/`, `ml/data/own-test/`, and a `summary.json` with no underfilled classes, or a list of dropped classes that the user has agreed to.

- [ ] **Step 1: Ask the user for permission to download TACO** from `github.com/pedropro/TACO`. Its README's `download.py` fetches the images. Tell the user its size before downloading. On a yes, clone it and download into `ml/data/taco/`.
- [ ] **Step 2:** `python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items --check`. Expected: exit 0. If a TACO category name differs, fix `ml/sources.json` to the name the error lists and re-run.
- [ ] **Step 3: Write `docs/superpowers/eval/device-checklist.md`.** It's a 20-row table: `item | expected bin | before: answer / asked / wrong | after: …`. Rows:
  - aluminum can
  - aerosol can
  - AA battery
  - pizza box
  - cardboard box
  - broken glass (photo)
  - glass bottle
  - foam egg carton
  - cardboard egg carton
  - plastic grocery bag
  - water bottle
  - paper coffee cup
  - chip bag
  - yogurt tub
  - foil tin
  - paper towel tube
  - light bulb
  - old T-shirt
  - banana peel
  - a houseplant (`not_waste`)
- [ ] **Step 4: Ask the user** to run the current build on their iPhone, scan the 20 items with no barcode or label visible, fill in the *before* column, and take photos per `ml/PHOTO_GUIDE.md`.
- [ ] **Step 5:** `python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items`. Show the user `summary.json`'s `underfilled` list and repeat Step 4 until they accept the remainder being dropped.
- [ ] **Step 6: Commit** `docs/superpowers/eval/device-checklist.md`: `docs: device checklist with baseline`.

### Task 8: Train, evaluate and install the item model

**Files:**
- Create: `Waste Wizard - Round Rock/Waste Wizard - Round Rock/WasteItems.mlmodel`, `ml/models/WasteItems.labels.txt`, `ml/models/WasteItems.report.json`, `ml/models/WasteMaterial-6class.baseline.json`
- Modify: `www/waste-models.js` (`ITEM_ACCEPT`)

**Interfaces:**
- Consumes: Tasks 3–5 and Task 7's data.
- Produces: the shipped item model, and `ITEM_ACCEPT` = the report's `effective_accept` rounded to 2 decimals.

- [ ] **Step 1:** `python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items --drop-underfilled`. Expected: exit 0.
- [ ] **Step 2:** `swift ml/train.swift --train ml/build/items/train --model-out ml/build/items/WasteItems.mlmodel --iterations 100`
- [ ] **Step 3:** `swift ml/predict.swift --model ml/build/items/WasteItems.mlmodel --test ml/build/items/test --out ml/build/items/predictions.json && python3 ml/report.py --predictions ml/build/items/predictions.json --truth-config ml/sources.json --require item=0.80,bin=0.92,coverage=0.65 --out ml/build/items/report.json`. Expected: exit 0. **On exit 1 or 3, stop and show the user `worst_classes`, `confusions` and `hazard_misses`. Collecting more photos for those classes is the remedy. Do not install.**
- [ ] **Step 4: Baseline.** `swift ml/predict.swift --model ml/baseline/WasteMaterial-6class.mlmodel --test ml/build/items/test --out ml/build/items/baseline.json && python3 ml/report.py --predictions ml/build/items/baseline.json --truth-config ml/sources.json --pred-config ml/materials.json --out ml/models/WasteMaterial-6class.baseline.json`. Record both models' `bin_acc` and `hazard_misses` for the hand-off. The exit code is informational.
- [ ] **Step 5: Install.**
  - Copy the model into the Swift folder.
  - Copy `WasteItems.labels.txt` and `report.json` (renamed `WasteItems.report.json`) into `ml/models/`.
  - Set `ITEM_ACCEPT` in `www/waste-models.js` to `effective_accept`.
  - Run `npm test`. Expected: PASS, including the installed-model labels test, which no longer skips.
- [ ] **Step 6: Commit** the model, `ml/models/WasteItems.*`, the baseline json and `www/waste-models.js`: `feat(ml): ship WasteItems item classifier`.

### Task 9: `VisionBridge` `item` op

**Files:**
- Modify: `Waste Wizard - Round Rock/Waste Wizard - Round Rock/ContentView.swift:141-147` (model loading) and `:229-248` (the `material` op)

**Interfaces:**
- Produces: the bridge accepts op `"item"`. On success, the result includes `items: [{value: String, confidence: Double}]` with the top 5, sorted descending. When the model is missing, `errors["item"] = "model not bundled"` and there is no `items` key. The `material` op output is unchanged (top 3 under `materials`).

- [ ] **Step 1: Replace `materialModel`'s inline loader** with `private static func loadModel(_ name: String) -> VNCoreMLModel?`, which loads `<name>.mlmodelc` from the bundle. Add `private static let itemModel = loadModel("WasteItems")`.
- [ ] **Step 2: Inside the queue closure, add `func classify(_ op: String, model: VNCoreMLModel?, key: String, top: Int)`.** It does what the current `material` block does: the revision loop via `run`, `.centerCrop`, sort, and prefix. Re-express the `material` op as `classify("material", model: Self.materialModel, key: "materials", top: 3)` and add `if ops.contains("item") { classify("item", model: Self.itemModel, key: "items", top: 5) }`.
- [ ] **Step 3: Build:** `xcodebuild -project "Waste Wizard - Round Rock/Waste Wizard - Round Rock.xcodeproj" -scheme "Waste Wizard - Round Rock" -destination 'generic/platform=iOS Simulator' build | tail -3`. Expected: `** BUILD SUCCEEDED **`.
- [ ] **Step 4:** Run `graphify update .`, then `graphify explain "VisionBridge"`. Expected: lists `.loadModel()`.
- [ ] **Step 5: Commit** `ContentView.swift`: `feat(ios): item model op in VisionBridge`.

### Task 10: Wire the item model into `handlePhoto`

**Files:**
- Modify: `www/index.html`. The relevant spots are:
  - script tags at about line 1239
  - `NativeVision.analyze` at 2407-2431
  - `MATERIAL_TO_ITEM` and `matchMaterialPrediction` at 3644-3674
  - `buildCandidates` at 3782
  - `handlePhoto` at 3983-4280
- Modify: `scripts/sync-targets.sh`

**Interfaces:**
- Consumes: `WasteModels.itemModelRan`, `interpretItems` and `interpretMaterials` (Task 2), and the bridge's `items` (Task 9).
- Produces: `_lastScanSignals.items: string[]`, the ITEMS names of the item model's candidates, used by `buildCandidates`.

- [ ] **Step 1: Load and expose the item model.**
  - Add `<script src="waste-models.js"></script>` after `mobilenet.min.js`.
  - In `NativeVision.analyze`, add `'item'` to the default ops and `items: res.items || []` to the returned object.
  - Initialise `_lastScanSignals` with `items: []`.
- [ ] **Step 2: Make the material model delegate.** Delete the inline `MATERIAL_TO_ITEM` and `MATERIAL_MIN_CONFIDENCE`. The body of `matchMaterialPrediction(materials)` becomes `return typeof WasteModels === 'undefined' ? null : WasteModels.interpretMaterials(materials, ITEMS);`.
- [ ] **Step 3: Add the item-model step in `handlePhoto`** immediately before `// Step 3: On-Device Vision Neural Network`.
  - Compute `const itemModelRan = typeof WasteModels !== 'undefined' && WasteModels.itemModelRan(_nativeVisionForCurrentPhoto)`.
  - When it ran:
    - Call `const v = WasteModels.interpretItems(_nativeVisionForCurrentPhoto.items, ITEMS, { confirmBelow: CONFIRM_BELOW })`, then set `_lastScanSignals.items = v ? v.candidates.map(i => i.n) : []`.
    - If `v` is `null`, fall through to MobileNet (still capped by Step 4).
    - If `v.kind === 'not_waste'`, call `showItemChooser` with the copy below and `buildCandidates(null, _lastScanSignals)`, then `return`.
    - If `v.kind` is `'answer'` or `'ask'`, call `await deliverScanResult({ n: v.item.n, cat: v.item.cat, note: v.item.note || '' }, { why: v.why, confidence: v.confidence, badge: v.badge, badgeIcon: v.item.cat === 'recycle' ? 'recycle' : 'bolt', badgeSub: 'On-device item model' })`, then `return`.
  - Exact copy for the `not_waste` chooser:
    - title: `Nothing to sort here?`
    - lead: `I could not spot one item to throw away in that photo. Fill the frame with just that item and try again, or pick it below.`
- [ ] **Step 4: Cap MobileNet when the item model ran.** In the `matched` MobileNet branch, pass `confidence: itemModelRan ? Math.min(matched.confidence, CONFIRM_BELOW - 0.01) : matched.confidence`.
- [ ] **Step 5: Put the item model's candidates first.** In `buildCandidates`, right after `push(bestItem)`, add `(sig && sig.items || []).forEach(n => push(ITEMS.find(i => i.n === n)));`.
- [ ] **Step 6: Update the sync script.** Add `waste-models.js` to `ASSETS` in `scripts/sync-targets.sh`, then run it. Expected: `Synced www/ -> …`.
- [ ] **Step 7: Verify the browser build has no regression.** Open `www/index.html` in the preview browser and scan any JPEG from `Garbage Dataset/standardized_384/glass`. Expected:
  - an answer or the chooser
  - no console errors
  - `typeof WasteModels === 'object'`
  - no `item` step runs, because there is no native bridge
- [ ] **Step 8:** Run `npm test` and the Task 9 `xcodebuild` command. Expected: PASS and `** BUILD SUCCEEDED **`.
- [ ] **Step 9: Device run.** Ask the user to install on their iPhone and fill in the *after* column of `docs/superpowers/eval/device-checklist.md`. Expected: more rows answered correctly, and no row in the wrong bin that wasn't wrong before. If a regression appears, debug it with superpowers:systematic-debugging before closing.
- [ ] **Step 10:** Run `graphify update .`, then commit `www/index.html`, `scripts/sync-targets.sh` and the updated checklist: `feat(scan): on-device item model leads visual recognition on iOS`.
