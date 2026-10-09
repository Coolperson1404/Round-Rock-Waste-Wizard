'use strict';
/* handlePhoto's item-model and material-model steps in www/index.html. The pure
   pieces are pulled out by name and evaluated; the wiring is checked in source. */
const test = require('node:test');
const assert = require('node:assert');
const WM = require('../../www/waste-models.js');
const { extractSource, readIndexHtml, loadItems } = require('./helpers/items.js');

const HTML = readIndexHtml();
const ITEMS = loadItems();

function handlePhotoSource() {
  const start = HTML.indexOf('async function handlePhoto(');
  const end = HTML.indexOf('/* ================= schedule', start);
  assert.ok(start > 0 && end > start);
  return HTML.slice(start, end);
}

function loadItemModelSignals() {
  // eslint-disable-next-line no-new-func
  return new Function('WasteModels', 'ITEMS',
    extractSource('function itemModelSignals(', HTML) + '\nreturn itemModelSignals;')(WM, ITEMS);
}

const interpret = preds => WM.interpretItems(preds, ITEMS, { confirmBelow: 0.60 });

test('itemModelSignals: an answer keys on the top item and keeps the guesses', () => {
  const preds = [{ value: 'aluminum_can', confidence: 0.9 }, { value: 'food_can', confidence: 0.05 }];
  assert.deepStrictEqual(loadItemModelSignals()(preds, interpret(preds)),
    { items: ['Aluminum can', 'Soup or vegetable can'], itemKey: 'Aluminum can' });
});

test('itemModelSignals: an ask keys on the top item', () => {
  const preds = [{ value: 'aluminum_can', confidence: 0.80 }, { value: 'aerosol_can', confidence: 0.15 }];
  const v = interpret(preds);
  assert.strictEqual(v.kind, 'ask');
  assert.strictEqual(loadItemModelSignals()(preds, v).itemKey, 'Aluminum can');
});

test('itemModelSignals: not_waste keeps the guesses but has no key', () => {
  const preds = [{ value: 'not_waste', confidence: 0.8 }, { value: 'paper_bag', confidence: 0.1 }];
  const v = interpret(preds);
  assert.strictEqual(v.kind, 'not_waste');
  assert.deepStrictEqual(loadItemModelSignals()(preds, v), { items: ['Paper bag'], itemKey: null });
});

test('itemModelSignals: a null verdict still keeps the guesses', () => {
  const preds = [{ value: 'aluminum_can', confidence: 0.30 }, { value: 'food_can', confidence: 0.25 }];
  assert.strictEqual(interpret(preds), null);
  assert.deepStrictEqual(loadItemModelSignals()(preds, null),
    { items: ['Aluminum can', 'Soup or vegetable can'], itemKey: null });
});

test('handlePhoto sets items and itemKey from itemModelSignals whatever the verdict', () => {
  const src = handlePhotoSource();
  assert.match(src, /Object\.assign\(_lastScanSignals,\s*itemModelSignals\(_nativeVisionForCurrentPhoto\.items, v\)\)/);
  assert.doesNotMatch(src, /_lastScanSignals\.items = v \?/);
});

test('the material fallback is capped below CONFIRM_BELOW when the item model ran', () => {
  const src = handlePhotoSource();
  assert.match(src, /confidence: itemModelRan \? Math\.min\(mat\.confidence, CONFIRM_BELOW - 0\.01\) : mat\.confidence/);
});

function loadMatchMaterial(WasteModels) {
  const warnings = [];
  const consoleStub = { warn: m => warnings.push(m) };
  // eslint-disable-next-line no-new-func
  const fn = new Function('WasteModels', 'ITEMS', 'console',
    extractSource('function matchMaterialPrediction(', HTML) + '\nreturn matchMaterialPrediction;')(
    WasteModels, ITEMS, consoleStub);
  return { fn, warnings };
}

test('matchMaterialPrediction warns once when waste-models.js is missing', () => {
  const { fn, warnings } = loadMatchMaterial(undefined);
  assert.strictEqual(fn([{ value: 'battery', confidence: 0.9 }]), null);
  assert.strictEqual(fn([{ value: 'battery', confidence: 0.9 }]), null);
  assert.deepStrictEqual(warnings, ['waste-models.js not loaded; material fallback disabled']);
});

test('matchMaterialPrediction delegates to WasteModels without warning', () => {
  const { fn, warnings } = loadMatchMaterial(WM);
  assert.strictEqual(fn([{ value: 'battery', confidence: 0.9 }]).match, 'Alkaline battery (AA, AAA, 9V)');
  assert.deepStrictEqual(warnings, []);
});

test('the material-model comment no longer describes a six-class TrashNet-only model', () => {
  assert.doesNotMatch(HTML, /only has to separate six\s+materials/);
});
