'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');
const WasteModels = require('../../www/waste-models.js');
const { loadItems } = require('./helpers/items.js');

const ROOT = path.join(__dirname, '..', '..');
const readJson = (p) => JSON.parse(fs.readFileSync(path.join(ROOT, p), 'utf8'));
const { ITEM_LABELS, MATERIAL_TO_ITEM } = WasteModels;
const ITEMS = loadItems();
const byName = new Map(ITEMS.map((i) => [i.n, i]));

test('every mapped item exists in ITEMS', () => {
  for (const [label, name] of Object.entries(ITEM_LABELS)) {
    if (name === null) continue;
    assert.ok(byName.has(name), `${label} -> "${name}" is not in ITEMS`);
  }
});

test('only not_waste is unmapped', () => {
  const unmapped = Object.entries(ITEM_LABELS).filter(([, v]) => v === null).map(([k]) => k);
  assert.deepEqual(unmapped, ['not_waste']);
});

test('sources.json classes match ITEM_LABELS', () => {
  const classes = Object.keys(readJson('ml/sources.json').classes).sort();
  assert.deepEqual(classes, Object.keys(ITEM_LABELS).sort());
  assert.equal(classes.length, 42);
});

test("sources.json bin equals the mapped item's cat", () => {
  const { classes } = readJson('ml/sources.json');
  for (const [label, name] of Object.entries(ITEM_LABELS)) {
    const bin = classes[label].bin;
    if (name === null) assert.equal(bin, 'none', label);
    else assert.equal(bin, byName.get(name).cat, label);
  }
});

test('materials.json matches MATERIAL_TO_ITEM', () => {
  const { classes } = readJson('ml/materials.json');
  assert.deepEqual(Object.keys(classes).sort(), Object.keys(MATERIAL_TO_ITEM).sort());
  assert.equal(Object.keys(classes).length, 10);
  for (const [label, { item }] of Object.entries(MATERIAL_TO_ITEM)) {
    assert.ok(byName.has(item), `${label} -> "${item}" is not in ITEMS`);
    assert.equal(classes[label].bin, byName.get(item).cat, label);
  }
});

test('installed item model labels are all mapped', (t) => {
  const p = path.join(ROOT, 'ml', 'models', 'WasteItems.labels.txt');
  if (!fs.existsSync(p)) return t.skip('no model installed yet');
  const lines = fs.readFileSync(p, 'utf8').split('\n').map((l) => l.trim()).filter(Boolean);
  for (const l of lines) assert.ok(l in ITEM_LABELS, `${l} is not a key of ITEM_LABELS`);
});

test('loadItems parses all 116 items', () => {
  assert.equal(loadItems().length, 116);
});
