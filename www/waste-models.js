/* Constants shared by the on-device item/material classifiers, the training
   pipeline's tests, and (later) the web app. Constants only; the runtime
   wiring arrives in a later task. */
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

  const WasteModels = {
    ITEM_LABELS: ITEM_LABELS,
    NOT_WASTE: NOT_WASTE,
    MATERIAL_TO_ITEM: MATERIAL_TO_ITEM,
    MATERIAL_MIN_CONFIDENCE: MATERIAL_MIN_CONFIDENCE,
    ITEM_SUGGEST: ITEM_SUGGEST,
    ITEM_ACCEPT: ITEM_ACCEPT
  };

  if (typeof module === 'object' && module.exports) module.exports = WasteModels;
  else root.WasteModels = WasteModels;
})(typeof self !== 'undefined' ? self : this);
