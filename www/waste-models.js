/* Constants and pure decision logic for the on-device item and material
   classifiers. Loaded by www/index.html before its inline script (handlePhoto
   and matchMaterialPrediction call it, always guarded by
   typeof WasteModels !== 'undefined') and required by the Node tests in
   ml/tests. Keep it free of DOM and app state. */
(function (root) {
  'use strict';

  /* Item-model label -> exact ITEMS[].n name in index.html. null means the
     class is deliberately not a waste item (nothing to look up). */
  const ITEM_LABELS = {
    cardboard_box: 'Cardboard box',
    pizza_box: 'Pizza box',
    egg_carton_cardboard: 'Egg carton (cardboard)',
    egg_carton_foam: 'Egg carton (foam)',
    paper_tube: 'Paper towel / toilet paper tube',
    paper: 'Junk mail, letters, envelopes',
    paper_bag: 'Paper bag',
    paper_plate: 'Paper plate',
    tissue: 'Napkin / paper towel / tissue',
    drink_carton: 'Milk or juice carton',
    paper_cup: 'Coffee cup (paper, with lining)',
    plastic_bottle: 'Water or soda bottle',
    milk_jug: 'Milk jug',
    plastic_tub: 'Yogurt / butter / cream cheese tub',
    detergent_bottle: 'Detergent or cleaning bottle',
    plastic_lid: 'Plastic lid',
    plastic_bag: 'Plastic grocery bag',
    plastic_film: 'Plastic wrap / bubble wrap',
    plastic_cup_cutlery: 'Plastic cutlery, straws, cups',
    chip_bag: 'Chip bag / candy wrapper',
    styrofoam_food: 'Styrofoam cup or takeout box',
    styrofoam_packaging: 'Styrofoam packaging (TV, furniture)',
    glass_bottle: 'Glass bottle (wine, beer, liquor)',
    glass_jar: 'Pasta sauce jar',
    broken_glass: 'Broken glass',
    drinking_glass: 'Drinking glass, Pyrex, ceramic dish',
    aluminum_can: 'Aluminum can',
    food_can: 'Soup or vegetable can',
    aluminum_foil: 'Aluminum foil',
    foil_tin: 'Foil baking tin',
    aerosol_can: 'Aerosol can (hairspray, spray paint)',
    paint_can: 'Paint can',
    reusable_bottle: 'Reusable bottle, flask or travel mug',
    battery: 'Alkaline battery (AA, AAA, 9V)',
    light_bulb: 'Regular light bulb / LED',
    fluorescent_bulb: 'Fluorescent light bulb / tube',
    clothing: 'Clothing / shoes / textiles',
    food_scraps: 'Food scraps / leftovers',
    kcup: 'K-Cup / coffee pod',
    cell_phone: 'Cell phone',
    laptop: 'Computer, laptop, tablet',
    not_waste: null
  };

  const NOT_WASTE = 'not_waste';

  /* Material-model label -> ITEMS entry plus the display name shown to the user. */
  const MATERIAL_TO_ITEM = {
    battery: { item: 'Alkaline battery (AA, AAA, 9V)', name: 'Battery' },
    biological: { item: 'Food scraps / leftovers', name: 'Food waste' },
    cardboard: { item: 'Cardboard box', name: 'Cardboard' },
    clothes: { item: 'Clothing / shoes / textiles', name: 'Clothing' },
    glass: { item: 'Glass bottle (wine, beer, liquor)', name: 'Glass' },
    metal: { item: 'Aluminum can', name: 'Metal' },
    paper: { item: 'Junk mail, letters, envelopes', name: 'Paper' },
    plastic: { item: 'Water or soda bottle', name: 'Plastic' },
    shoes: { item: 'Clothing / shoes / textiles', name: 'Shoes' },
    trash: { item: 'Plastic cutlery, straws, cups', name: 'Non-recyclable waste' }
  };

  /* Below this a material guess is not worth acting on. */
  const MATERIAL_MIN_CONFIDENCE = 0.60;
  /* Item model: show as a suggestion at or above ITEM_SUGGEST, act on it at or
     above ITEM_ACCEPT. ITEM_ACCEPT is a placeholder; a later task sets the real value. */
  const ITEM_SUGGEST = 0.35;
  const ITEM_ACCEPT = 0.60;
  /* Hazard safety net. Confidences are softmax probabilities that sum to 1,
     so a hazardous runner-up can never reach ITEM_SUGGEST while the top is
     above ITEM_ACCEPT. A non-hhw answer is asked about instead whenever an hhw
     item at rank 2 or 3 scores at least this much. ml/report.py applies the
     same rule (its HAZARD_RUNNER_UP must equal this; a test checks it). */
  const HAZARD_RUNNER_UP = 0.10;

  /* True only when the native item model actually returned predictions. */
  function itemModelRan(native) {
    return !!native && Array.isArray(native.items) && native.items.length > 0;
  }

  /* Usable predictions, best first: finite confidence, a known label, and
     (unless it is not_waste) an ITEMS entry the app has. item is null for
     not_waste. */
  function rankPredictions(predictions, items) {
    const ranked = [];
    for (const p of predictions) {
      if (!p) continue;
      const confidence = Number(p.confidence);
      if (!Number.isFinite(confidence)) continue;
      const label = String(p.value);
      if (!Object.prototype.hasOwnProperty.call(ITEM_LABELS, label)) continue;
      const name = ITEM_LABELS[label];
      let item = null;
      if (name !== null) {
        item = items.find(i => i.n === name);
        if (!item) continue;
      }
      ranked.push({ label: label, item: item, confidence: confidence });
    }
    ranked.sort((a, b) => b.confidence - a.confidence);
    return ranked;
  }

  /* The top 3 distinct mapped items of a ranked list, at any confidence. */
  function candidatesOf(ranked) {
    const out = [];
    for (const r of ranked) {
      if (!r.item || out.indexOf(r.item) !== -1) continue;
      out.push(r.item);
      if (out.length === 3) break;
    }
    return out;
  }

  /* The item model's guesses for the chooser: up to 3 distinct ITEMS entries
     (never not_waste), top first, whatever the confidence and whatever
     interpretItems decides. [] when there is nothing usable. */
  function itemCandidates(predictions, items) {
    if (!Array.isArray(predictions) || !Array.isArray(items)) return [];
    return candidatesOf(rankPredictions(predictions, items));
  }

  /* Turn raw item-model predictions into an answer / ask / not_waste verdict,
     or null when the model has no usable opinion. An answer or ask carries
     `hazard`: the hhw item at rank 2 or 3 that scored at least
     hazardRunnerUp while the top item is not hhw (null otherwise). When it
     is set the verdict is always an ask. */
  function interpretItems(predictions, items, opts) {
    if (!Array.isArray(predictions) || !Array.isArray(items)) return null;
    const o = opts || {};
    const confirmBelow = o.confirmBelow !== undefined ? o.confirmBelow : 0.60;
    const accept = o.accept !== undefined ? o.accept : ITEM_ACCEPT;
    const suggest = o.suggest !== undefined ? o.suggest : ITEM_SUGGEST;
    const hazardRunnerUp = o.hazardRunnerUp !== undefined ? o.hazardRunnerUp : HAZARD_RUNNER_UP;

    const ranked = rankPredictions(predictions, items);
    if (!ranked.length || ranked[0].confidence < suggest) return null;
    const top = ranked[0];
    const candidates = candidatesOf(ranked);

    if (top.label === NOT_WASTE) {
      /* The item guesses still lead the "nothing here?" chooser. */
      return {
        kind: 'not_waste', item: null, confidence: top.confidence, candidates: candidates,
        hazard: null,
        why: 'On-device item model does not see a waste item here.',
        badge: 'Item model: not waste (' + Math.round(top.confidence * 100) + '%)'
      };
    }

    /* Safety net: a closed-set model can be confidently wrong about a
       hazardous item. If a hazardous item is a plausible runner-up, ask
       instead of sending it to the cart. It is always among the candidates. */
    const runnerUp = top.item.cat === 'hhw' ? null : ranked.slice(1, 3).find(r =>
      r.item && r.item.cat === 'hhw' && r.confidence >= hazardRunnerUp);
    const hazard = runnerUp ? runnerUp.item : null;
    const isAnswer = !hazard && top.confidence >= Math.max(accept, confirmBelow);
    const confidence = isAnswer ? top.confidence : Math.min(top.confidence, confirmBelow - 0.01);
    return {
      kind: isAnswer ? 'answer' : 'ask',
      item: top.item,
      confidence: confidence,
      candidates: candidates,
      hazard: hazard,
      why: 'On-device item model recognised this as ' + top.item.n + '.',
      badge: 'Item model: ' + top.item.n + ' (' + Math.round(confidence * 100) + '%)'
    };
  }

  /* Material-model fallback. index.html's matchMaterialPrediction is a thin,
     guarded wrapper around this. */
  function interpretMaterials(materials, items) {
    if (!Array.isArray(materials) || !materials.length || !Array.isArray(items)) return null;
    const top = materials[0];
    if (!top || !(top.confidence >= MATERIAL_MIN_CONFIDENCE)) return null;
    const mapped = MATERIAL_TO_ITEM[String(top.value).toLowerCase()];
    if (!mapped) return null;
    const item = items.find(i => i.n === mapped.item);
    if (!item) return null;
    return {
      item: mapped.name + ' item',
      match: item.n,
      cat: item.cat,
      why: 'Trained material classifier identified this as ' + mapped.name.toLowerCase() + '.',
      tip: null,
      badge: 'Material: ' + mapped.name + ' (' + Math.round(top.confidence * 100) + '%)',
      badgeIcon: item.cat === 'recycle' ? 'recycle' : 'bolt',
      confidence: top.confidence
    };
  }

  const WasteModels = {
    ITEM_LABELS: ITEM_LABELS,
    NOT_WASTE: NOT_WASTE,
    MATERIAL_TO_ITEM: MATERIAL_TO_ITEM,
    MATERIAL_MIN_CONFIDENCE: MATERIAL_MIN_CONFIDENCE,
    ITEM_SUGGEST: ITEM_SUGGEST,
    ITEM_ACCEPT: ITEM_ACCEPT,
    HAZARD_RUNNER_UP: HAZARD_RUNNER_UP,
    itemModelRan: itemModelRan,
    itemCandidates: itemCandidates,
    interpretItems: interpretItems,
    interpretMaterials: interpretMaterials
  };

  if (typeof module === 'object' && module.exports) module.exports = WasteModels;
  else root.WasteModels = WasteModels;
})(typeof self !== 'undefined' ? self : this);
