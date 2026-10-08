# Photo guide: teaching the app your items

The on-device item model learns from public datasets plus **your own photos**. Photos of real
objects in real kitchens matter most, and 12 of the 42 classes have no public data at all.

## Folder layout

```
ml/data/own/<label>/        training photos
ml/data/own-test/<label>/   test photos (held out; never used for training)
```

Create a folder named exactly like the label (the left column below). Folders you do not create
are simply skipped.

## Labels

| Label | What it is | Training photos |
|---|---|---|
| `cardboard_box` | Cardboard box | at least 10 recommended |
| `pizza_box` | Pizza box | at least 10 recommended |
| `egg_carton_cardboard` | Egg carton (cardboard) | at least 10 recommended |
| `egg_carton_foam` | Egg carton (foam) | **40 required** |
| `paper_tube` | Paper towel / toilet paper tube | at least 10 recommended |
| `paper` | Junk mail, letters, envelopes | at least 10 recommended |
| `paper_bag` | Paper bag | at least 10 recommended |
| `paper_plate` | Paper plate | **40 required** |
| `tissue` | Napkin / paper towel / tissue | at least 10 recommended |
| `drink_carton` | Milk or juice carton | at least 10 recommended |
| `paper_cup` | Coffee cup (paper, with lining) | at least 10 recommended |
| `plastic_bottle` | Water or soda bottle | at least 10 recommended |
| `milk_jug` | Milk jug | **40 required** |
| `plastic_tub` | Yogurt / butter / cream cheese tub | at least 10 recommended |
| `detergent_bottle` | Detergent or cleaning bottle | at least 10 recommended |
| `plastic_lid` | Plastic lid | at least 10 recommended |
| `plastic_bag` | Plastic grocery bag | at least 10 recommended |
| `plastic_film` | Plastic wrap / bubble wrap | at least 10 recommended |
| `plastic_cup_cutlery` | Plastic cutlery, straws, cups | at least 10 recommended |
| `chip_bag` | Chip bag / candy wrapper | at least 10 recommended |
| `styrofoam_food` | Styrofoam cup or takeout box | at least 10 recommended |
| `styrofoam_packaging` | Styrofoam packaging (TV, furniture) | at least 10 recommended |
| `glass_bottle` | Glass bottle (wine, beer, liquor) | at least 10 recommended |
| `glass_jar` | Pasta sauce jar | at least 10 recommended |
| `broken_glass` | Broken glass | at least 10 recommended |
| `drinking_glass` | Drinking glass, Pyrex, ceramic dish | at least 10 recommended |
| `aluminum_can` | Aluminum can | at least 10 recommended |
| `food_can` | Soup or vegetable can | at least 10 recommended |
| `aluminum_foil` | Aluminum foil | at least 10 recommended |
| `foil_tin` | Foil baking tin | **40 required** |
| `aerosol_can` | Aerosol can (hairspray, spray paint) | at least 10 recommended |
| `paint_can` | Paint can | **40 required** |
| `reusable_bottle` | Reusable bottle, flask or travel mug | **40 required** |
| `battery` | Alkaline battery (AA, AAA, 9V) | at least 10 recommended |
| `light_bulb` | Regular light bulb / LED | **40 required** |
| `fluorescent_bulb` | Fluorescent light bulb / tube | **40 required** |
| `clothing` | Clothing / shoes / textiles | at least 10 recommended |
| `food_scraps` | Food scraps / leftovers | at least 10 recommended |
| `kcup` | K-Cup / coffee pod | **40 required** |
| `cell_phone` | Cell phone | **40 required** |
| `laptop` | Computer, laptop, tablet | **40 required** |
| `not_waste` | (anything that is not waste: hands, tables, empty scenes, non-waste objects) | **40 required** |

Every class also needs **10 test photos** in `ml/data/own-test/<label>/`.

## Framing

- One item per photo.
- The item fills at least half the frame.
- Keep the item in focus. No other waste in the background.

## Variety

- Different counters, floors and hands.
- Both daylight and lamp light.
- At least 3 angles per item (top, side, tilted).
- Dirty, crushed or half-empty versions where it is realistic.

## Test photos

Test photos must show **different physical objects** from the training photos. A second photo of
the same bottle you trained on tells you nothing; use another bottle. Exact duplicate files in
both folders are rejected by the script (it exits with code 2).

## Getting photos onto the Mac

AirDrop them from the iPhone and drop them into the right folder. HEIC files are fine; the script
converts them. JPEG, PNG and WebP also work.

## How many

- **40 training photos** for the 12 classes marked "40 required" above (no public data).
- **At least 10 training photos** recommended for every other class.
- **10 test photos** for every class.

## Checking progress

```
python3 ml/prepare_data.py --config ml/sources.json --out ml/build/items
```

This command needs the TACO dataset to be downloaded into `ml/data/taco/` first (the workflow
downloads TACO before photo collection). Until `ml/data/taco/data/annotations.json` exists it
stops with exit code 1 and an error like:

```
config error: cardboard_box: COCO annotations not found: .../ml/data/taco/data/annotations.json
```

Once TACO is in place, it prints the train/test counts for each class and flags the ones still short (`<-- short`).
Add `--check` to only validate the config, or `--drop-underfilled` to leave short classes out of
the build. The same counts are written to `ml/build/items/summary.json`.
