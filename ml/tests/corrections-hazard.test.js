'use strict';
/* A broad learned key (cls: / itm:) must never move an answer out of hazardous
   waste. The decision is learnedMayAssert in www/index.html's inline script,
   pulled out by name and evaluated here, as in signal-keys.test.js. */
const test = require('node:test');
const assert = require('node:assert');
const { extractSource, readIndexHtml, loadItems } = require('./helpers/items.js');

const HTML = readIndexHtml();
const ITEMS = loadItems();
const SRC = ['function _normToken(', 'function signalKeys(', 'function isBroadKey(',
  'function learnedMayAssert(', 'const Corrections = {'].map(h => extractSource(h, HTML)).join('\n');

function load() {
  const mem = {};
  const store = { get: (k, d) => (k in mem ? mem[k] : d), set: (k, v) => { mem[k] = v; } };
  const VisualMemory = { remember() {} };
  // eslint-disable-next-line no-new-func
  return new Function('store', 'ITEMS', 'VisualMemory', 'CORRECTION_STORE_KEY',
    SRC + '\nreturn { learnedMayAssert, Corrections };')(store, ITEMS, VisualMemory, 'c');
}

const item = n => {
  const it = ITEMS.find(i => i.n === n);
  assert.ok(it, n);
  return it;
};
const AEROSOL = item('Aerosol can (hairspray, spray paint)');
const CAN = item('Aluminum can');

/* A twice-taught itm: correction whose learned item is recyclable. */
function twiceTaught() {
  const { learnedMayAssert, Corrections } = load();
  const sig = { itemKey: 'Aerosol can (hairspray, spray paint)' };
  Corrections.record(sig, 'Aluminum can');
  Corrections.record(sig, 'Aluminum can');
  const learned = Corrections.lookup(sig, 0.9);
  assert.strictEqual(learned.key, 'itm:Aerosol can (hairspray, spray paint)');
  assert.strictEqual(learned.hits, 2);
  assert.strictEqual(learned.item.cat, 'recycle');
  return { learnedMayAssert, learned };
}

test('a twice-taught broad key does not move an hhw result out of hazardous', () => {
  const { learnedMayAssert, learned } = twiceTaught();
  assert.strictEqual(learnedMayAssert(learned, 'hhw', null), false);
});

test('a twice-taught broad key does not override a hazard-downgraded ask', () => {
  const { learnedMayAssert, learned } = twiceTaught();
  assert.strictEqual(learnedMayAssert(learned, 'recycle', AEROSOL), false);
});

test('a twice-taught broad key still applies to a normal recycle result', () => {
  const { learnedMayAssert, learned } = twiceTaught();
  assert.strictEqual(learnedMayAssert(learned, 'recycle', null), true);
  assert.strictEqual(learnedMayAssert(learned, 'recycle', undefined), true);
});

test('a broad key whose learned item is hhw may assert on an hhw result', () => {
  const { learnedMayAssert } = load();
  assert.strictEqual(learnedMayAssert({ key: 'cls:can', item: AEROSOL, hits: 3 }, 'hhw', null), true);
});

test('cls: keys are guarded like itm: keys', () => {
  const { learnedMayAssert } = load();
  assert.strictEqual(learnedMayAssert({ key: 'cls:can', item: CAN, hits: 5 }, 'hhw', null), false);
});

test('a specific key (bc:, txt:) still asserts against an hhw result', () => {
  const { learnedMayAssert } = load();
  assert.strictEqual(learnedMayAssert({ key: 'bc:012345', item: CAN, hits: 1 }, 'hhw', null), true);
  assert.strictEqual(learnedMayAssert({ key: 'txt:COKE', item: CAN, hits: 1 }, 'hhw', AEROSOL), true);
});

test('no learned answer means nothing to assert', () => {
  const { learnedMayAssert } = load();
  assert.strictEqual(learnedMayAssert(null, 'hhw', null), false);
});

test('deliverScanResult gates learned answers on learnedMayAssert and offers all three', () => {
  const src = extractSource('async function deliverScanResult(', HTML);
  assert.match(src, /learnedMayAssert\(learned, it\.cat, extra && extra\.hazardItem\)/);
  // The chooser lists the learned item, the incoming item and the hazardous item.
  assert.match(src, /\[learned\.item, incoming, hazardItem\]/);
});

test('the item step passes the hazardous runner-up to deliverScanResult', () => {
  assert.match(HTML, /hazardItem: v\.hazard/);
});
