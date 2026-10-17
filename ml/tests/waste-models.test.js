'use strict';
const test = require('node:test');
const assert = require('node:assert');
const WM = require('../../www/waste-models.js');
const { loadItems } = require('./helpers/items.js');

const ITEMS = loadItems();
const ACCEPT = { accept: 0.60 };

test('answers a confident can', () => {
  const v = WM.interpretItems(
    [{ value: 'aluminum_can', confidence: 0.91 }, { value: 'food_can', confidence: 0.05 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.item.n, 'Aluminum can');
  assert.strictEqual(v.confidence, 0.91);
  assert.deepStrictEqual(v.candidates.map(c => c.n), ['Aluminum can', 'Soup or vegetable can']);
  assert.strictEqual(v.why, 'On-device item model recognised this as Aluminum can.');
  assert.strictEqual(v.badge, 'Item model: Aluminum can (91%)');
  assert.strictEqual(v.hazard, null);
});

test('asks when unsure', () => {
  const v = WM.interpretItems(
    [{ value: 'aerosol_can', confidence: 0.5 }, { value: 'aluminum_can', confidence: 0.4 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.item.n, 'Aerosol can (hairspray, spray paint)');
  assert.strictEqual(v.confidence, 0.5);
});

test('clips ask confidence below confirmBelow', () => {
  const v = WM.interpretItems([{ value: 'aluminum_can', confidence: 0.7 }], ITEMS, { accept: 0.8 });
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.confidence, 0.59);
});

test('accept is never below confirmBelow', () => {
  const v = WM.interpretItems([{ value: 'aluminum_can', confidence: 0.5 }], ITEMS, { accept: 0.4 });
  assert.strictEqual(v.kind, 'ask');
});

test('not_waste', () => {
  const v = WM.interpretItems([{ value: 'not_waste', confidence: 0.8 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'not_waste');
  assert.strictEqual(v.item, null);
  assert.deepStrictEqual(v.candidates, []);
});

test('not_waste keeps the item guesses (any confidence) for the chooser', () => {
  const v = WM.interpretItems([
    { value: 'not_waste', confidence: 0.8 },
    { value: 'aluminum_can', confidence: 0.1 },
    { value: 'food_can', confidence: 0.05 },
    { value: 'paper', confidence: 0.03 },
    { value: 'battery', confidence: 0.01 }
  ], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'not_waste');
  assert.strictEqual(v.item, null);
  assert.deepStrictEqual(v.candidates.map(c => c.n),
    ['Aluminum can', 'Soup or vegetable can', 'Junk mail, letters, envelopes']);
});

/* Item-model confidences are softmax probabilities that sum to 1, so every
   hazard test below keeps its confidences summing to <= 1. */
const AEROSOL = 'Aerosol can (hairspray, spray paint)';

test('HAZARD_RUNNER_UP is 0.10', () => {
  assert.strictEqual(WM.HAZARD_RUNNER_UP, 0.10);
});

test('a hazardous runner-up at >= HAZARD_RUNNER_UP turns an answer into a question', () => {
  const v = WM.interpretItems(
    [{ value: 'aluminum_can', confidence: 0.80 }, { value: 'aerosol_can', confidence: 0.15 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.item.n, 'Aluminum can');
  assert.strictEqual(v.confidence, 0.59);
  assert.strictEqual(v.hazard.n, AEROSOL);
  assert.strictEqual(v.hazard.cat, 'hhw');
  assert.ok(v.candidates.includes(v.hazard));
  assert.strictEqual(v.badge, 'Item model: Aluminum can (59%)');
});

test('a hazardous runner-up below HAZARD_RUNNER_UP does not block an answer', () => {
  const v = WM.interpretItems(
    [{ value: 'aluminum_can', confidence: 0.85 }, { value: 'aerosol_can', confidence: 0.08 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.confidence, 0.85);
  assert.strictEqual(v.hazard, null);
});

test('a hazardous top answer is not downgraded by a non-hazardous runner-up', () => {
  const v = WM.interpretItems(
    [{ value: 'aerosol_can', confidence: 0.80 }, { value: 'aluminum_can', confidence: 0.15 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.item.n, AEROSOL);
  assert.strictEqual(v.hazard, null);
});

test('a hazardous top answer is not downgraded by another hazardous guess', () => {
  const v = WM.interpretItems(
    [{ value: 'aerosol_can', confidence: 0.75 }, { value: 'paint_can', confidence: 0.20 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.hazard, null);
});

test('the hazard check also looks at rank 3', () => {
  const v = WM.interpretItems([
    { value: 'aluminum_can', confidence: 0.70 }, { value: 'food_can', confidence: 0.15 },
    { value: 'aerosol_can', confidence: 0.12 }
  ], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.hazard.n, AEROSOL);
});

test('the hazard check looks only at ranks 2 and 3', () => {
  const v = WM.interpretItems([
    { value: 'aluminum_can', confidence: 0.62 }, { value: 'food_can', confidence: 0.13 },
    { value: 'foil_tin', confidence: 0.12 }, { value: 'aerosol_can', confidence: 0.11 }
  ], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.hazard, null);
});

test('opts.hazardRunnerUp overrides the hazard threshold', () => {
  const preds = [{ value: 'aluminum_can', confidence: 0.85 }, { value: 'aerosol_can', confidence: 0.08 }];
  const v = WM.interpretItems(preds, ITEMS, { accept: 0.60, hazardRunnerUp: 0.05 });
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.hazard.n, AEROSOL);
});

test('an ask below accept still names a plausible hazardous runner-up', () => {
  const v = WM.interpretItems(
    [{ value: 'aluminum_can', confidence: 0.50 }, { value: 'aerosol_can', confidence: 0.40 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(v.hazard.n, AEROSOL);
});

test('a not_waste verdict carries no hazard', () => {
  const v = WM.interpretItems(
    [{ value: 'not_waste', confidence: 0.80 }, { value: 'aerosol_can', confidence: 0.15 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'not_waste');
  assert.strictEqual(v.hazard, null);
});

test('HAZARD_RUNNER_UP matches ml/report.py', () => {
  const src = require('fs').readFileSync(require('path').join(__dirname, '..', 'report.py'), 'utf8');
  const m = src.match(/^HAZARD_RUNNER_UP = ([0-9.]+)\s*$/m);
  assert.ok(m, 'HAZARD_RUNNER_UP not found in ml/report.py');
  assert.strictEqual(Number(m[1]), WM.HAZARD_RUNNER_UP);
});

test('itemCandidates: up to 3 distinct mapped items, top first, skipping not_waste', () => {
  const preds = [
    { value: 'not_waste', confidence: 0.5 }, { value: 'hovercraft', confidence: 0.2 },
    { value: 'food_can', confidence: 0.1 }, { value: 'aluminum_can', confidence: 0.15 },
    { value: 'paper', confidence: 0.03 }, { value: 'battery', confidence: 0.02 }
  ];
  assert.deepStrictEqual(WM.itemCandidates(preds, ITEMS).map(c => c.n),
    ['Aluminum can', 'Soup or vegetable can', 'Junk mail, letters, envelopes']);
});

test('itemCandidates keeps guesses below suggest, and is [] for bad input', () => {
  assert.deepStrictEqual(WM.itemCandidates([{ value: 'aluminum_can', confidence: 0.2 }], ITEMS).map(c => c.n),
    ['Aluminum can']);
  assert.strictEqual(WM.interpretItems([{ value: 'aluminum_can', confidence: 0.2 }], ITEMS, ACCEPT), null);
  assert.deepStrictEqual(WM.itemCandidates(null, ITEMS), []);
  assert.deepStrictEqual(WM.itemCandidates([], ITEMS), []);
  assert.deepStrictEqual(WM.itemCandidates([{ value: 'x', confidence: NaN }], ITEMS), []);
});

test('interpretItems candidates equal itemCandidates', () => {
  const preds = [{ value: 'aluminum_can', confidence: 0.7 }, { value: 'food_can', confidence: 0.2 },
    { value: 'foil_tin', confidence: 0.05 }, { value: 'paper', confidence: 0.04 }];
  assert.deepStrictEqual(WM.interpretItems(preds, ITEMS, ACCEPT).candidates, WM.itemCandidates(preds, ITEMS));
});

test('interpretItems reuses itemCandidates rather than duplicating it', () => {
  const src = require('fs').readFileSync(require('path').join(__dirname, '..', '..', 'www', 'waste-models.js'), 'utf8');
  const body = src.slice(src.indexOf('function interpretItems('), src.indexOf('function interpretMaterials('));
  assert.doesNotMatch(body, /candidates\.length === 3/);
  assert.doesNotMatch(body, /splice\(/);
});

test('no opinion below suggest', () => {
  assert.strictEqual(WM.interpretItems([{ value: 'aluminum_can', confidence: 0.2 }], ITEMS, ACCEPT), null);
});

test('skips unknown labels', () => {
  const v = WM.interpretItems(
    [{ value: 'hovercraft', confidence: 0.9 }, { value: 'aluminum_can', confidence: 0.7 }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
  assert.strictEqual(v.item.n, 'Aluminum can');
});

test('ignores non-finite confidences', () => {
  assert.strictEqual(WM.interpretItems([{ value: 'aluminum_can', confidence: NaN }], ITEMS, ACCEPT), null);
  const v = WM.interpretItems([{ value: 'aluminum_can', confidence: '0.9' }], ITEMS, ACCEPT);
  assert.strictEqual(v.kind, 'answer');
});

test('sorts unsorted input', () => {
  const v = WM.interpretItems(
    [{ value: 'food_can', confidence: 0.1 }, { value: 'aluminum_can', confidence: 0.8 }], ITEMS, ACCEPT);
  assert.strictEqual(v.item.n, 'Aluminum can');
});

test('null / [] / non-array give null', () => {
  assert.strictEqual(WM.interpretItems(null, ITEMS, ACCEPT), null);
  assert.strictEqual(WM.interpretItems([], ITEMS, ACCEPT), null);
  assert.strictEqual(WM.interpretItems('aluminum_can', ITEMS, ACCEPT), null);
});

test('itemModelRan', () => {
  assert.strictEqual(WM.itemModelRan(null), false);
  assert.strictEqual(WM.itemModelRan({ items: [], errors: { item: 'model not bundled' } }), false);
  assert.strictEqual(WM.itemModelRan({ items: [{ value: 'x', confidence: 0.1 }] }), true);
});

test('interpretMaterials', () => {
  const m = WM.interpretMaterials([{ value: 'battery', confidence: 0.9 }], ITEMS);
  assert.strictEqual(m.match, 'Alkaline battery (AA, AAA, 9V)');
  assert.strictEqual(m.cat, 'hhw');
  assert.strictEqual(m.badge, 'Material: Battery (90%)');
  assert.strictEqual(m.item, 'Battery item');
  assert.strictEqual(m.confidence, 0.9);
  // Below the gate-measured 0.85 threshold the material model gives no answer.
  assert.strictEqual(WM.interpretMaterials([{ value: 'battery', confidence: 0.8 }], ITEMS), null);
  assert.strictEqual(WM.interpretMaterials([{ value: 'battery', confidence: 0.5 }], ITEMS), null);
  assert.strictEqual(WM.interpretMaterials([{ value: 'unobtainium', confidence: 0.9 }], ITEMS), null);
});

test('ITEM_ACCEPT is within [0.60, 0.95]', () => {
  assert.ok(WM.ITEM_ACCEPT >= 0.60 && WM.ITEM_ACCEPT <= 0.95);
});
