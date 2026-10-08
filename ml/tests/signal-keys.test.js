'use strict';
/* Correction learning on the item-model path. The functions live in the inline
   script of www/index.html, so they are pulled out by name and evaluated here. */
const test = require('node:test');
const assert = require('node:assert');
const { extractSource, readIndexHtml, loadItems } = require('./helpers/items.js');

const HTML = readIndexHtml();
const ITEMS = loadItems();
const SRC = ['function _normToken(', 'function signalKeys(', 'function isBroadKey(',
  'const Corrections = {'].map(h => extractSource(h, HTML)).join('\n');

function load() {
  const mem = {};
  const store = { get: (k, d) => (k in mem ? mem[k] : d), set: (k, v) => { mem[k] = v; } };
  const VisualMemory = { remember() {} };
  // eslint-disable-next-line no-new-func
  return new Function('store', 'ITEMS', 'VisualMemory', 'CORRECTION_STORE_KEY',
    SRC + '\nreturn { signalKeys, isBroadKey, Corrections };')(store, ITEMS, VisualMemory, 'c');
}

test('item-model top candidate gives an itm: key', () => {
  const { signalKeys } = load();
  assert.deepStrictEqual(signalKeys({ items: ['Aluminum can'] }), [['itm:Aluminum can', 0.75]]);
});

test('only the top item candidate is keyed', () => {
  const { signalKeys } = load();
  assert.deepStrictEqual(signalKeys({ items: ['Aluminum can', 'Soup or vegetable can'] }),
    [['itm:Aluminum can', 0.75]]);
});

test('no items gives no keys', () => {
  const { signalKeys } = load();
  assert.deepStrictEqual(signalKeys({ items: [] }), []);
  assert.deepStrictEqual(signalKeys(null), []);
});

test('classes still give cls: keys', () => {
  const { signalKeys } = load();
  assert.deepStrictEqual(signalKeys({ classes: ['water bottle, flask', 'can'] }),
    [['cls:water bottle', 0.75], ['cls:can', 0.75]]);
});

test('itm: and cls: are broad keys; bc: and txt: are not', () => {
  const { isBroadKey } = load();
  assert.strictEqual(isBroadKey('itm:Aluminum can'), true);
  assert.strictEqual(isBroadKey('cls:can'), true);
  assert.strictEqual(isBroadKey('bc:012345'), false);
  assert.strictEqual(isBroadKey('txt:COKE'), false);
});

test('a correction recorded on the item-model path is found again', () => {
  const { Corrections } = load();
  const sig = { items: ['Aluminum can'] };
  assert.strictEqual(Corrections.record(sig, 'Aerosol can (hairspray, spray paint)'), 1);
  const hit = Corrections.lookup(sig, 0.5);
  assert.strictEqual(hit.key, 'itm:Aluminum can');
  assert.strictEqual(hit.item.n, 'Aerosol can (hairspray, spray paint)');
});

test('a once-taught itm: key does not override a confident pipeline', () => {
  const { Corrections } = load();
  const sig = { items: ['Aluminum can'] };
  Corrections.record(sig, 'Aerosol can (hairspray, spray paint)');
  assert.strictEqual(Corrections.lookup(sig, 0.8), null);
  assert.ok(Corrections.lookup(sig, null));
  Corrections.record(sig, 'Aerosol can (hairspray, spray paint)');
  assert.strictEqual(Corrections.lookup(sig, 0.8).hits, 2);
});

test('both broad-key gates use isBroadKey', () => {
  assert.match(extractSource('const Corrections = {', HTML), /isBroadKey\(k\)/);
  assert.match(extractSource('async function deliverScanResult(', HTML), /isBroadKey\(learned\.key\)/);
  assert.doesNotMatch(HTML, /indexOf\('cls:'\) === 0/);
});

test('not_waste learned override uses the not_waste confidence for the broad-key gate', () => {
  assert.doesNotMatch(HTML, /Corrections\.lookup\(sig, null\)/);
  assert.match(HTML, /Corrections\.lookup\(sig, v\.confidence\)/);
});
