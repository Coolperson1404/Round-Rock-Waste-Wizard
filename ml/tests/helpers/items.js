'use strict';
const fs = require('fs');
const path = require('path');

/* Parses the `const ITEMS = [ ... ];` block of www/index.html and returns
   [{ n, cat }]. A regex is enough: every entry starts `{ n: '...', cat: '...'`. */
function loadItems() {
  const html = fs.readFileSync(path.join(__dirname, '..', '..', '..', 'www', 'index.html'), 'utf8');
  const start = html.indexOf('const ITEMS = [');
  if (start < 0) throw new Error('const ITEMS = [ not found in www/index.html');
  const end = html.indexOf('\n];', start);
  const block = html.slice(start, end < 0 ? undefined : end);
  const re = /\{\s*n:\s*'([^']*)',\s*cat:\s*'([a-z]+)'/g;
  const out = [];
  let m;
  while ((m = re.exec(block))) out.push({ n: m[1], cat: m[2] });
  return out;
}

module.exports = { loadItems };
