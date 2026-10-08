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
  const m = WM.interpretMaterials([{ value: 'battery', confidence: 0.8 }], ITEMS);
  assert.strictEqual(m.match, 'Alkaline battery (AA, AAA, 9V)');
  assert.strictEqual(m.cat, 'hhw');
  assert.strictEqual(m.badge, 'Material: Battery (80%)');
  assert.strictEqual(m.item, 'Battery item');
  assert.strictEqual(m.confidence, 0.8);
  assert.strictEqual(WM.interpretMaterials([{ value: 'battery', confidence: 0.5 }], ITEMS), null);
  assert.strictEqual(WM.interpretMaterials([{ value: 'unobtainium', confidence: 0.9 }], ITEMS), null);
});

test('ITEM_ACCEPT is within [0.60, 0.95]', () => {
  assert.ok(WM.ITEM_ACCEPT >= 0.60 && WM.ITEM_ACCEPT <= 0.95);
});
