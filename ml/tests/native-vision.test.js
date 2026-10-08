'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { extractSource } = require('./helpers/items.js');

// eslint-disable-next-line no-new-func
const benign = new Function(extractSource('function _benignVisionErrors(') +
  '\nreturn _benignVisionErrors;')();

test('a missing item model alone is not worth a warning', () => {
  assert.strictEqual(benign({ item: 'model not bundled' }), true);
});

test('real failures still warn', () => {
  assert.strictEqual(benign({ item: 'model not bundled', classify: 'no inference context' }), false);
  assert.strictEqual(benign({ item: 'prediction failed' }), false);
  assert.strictEqual(benign({ material: 'model not bundled' }), false);
});

test('no errors is not a failure either way', () => {
  assert.strictEqual(benign(null), false);
  assert.strictEqual(benign({}), false);
});
