'use strict';
const fs = require('fs');
const path = require('path');

const INDEX_HTML = path.join(__dirname, '..', '..', '..', 'www', 'index.html');

function readIndexHtml() {
  return fs.readFileSync(INDEX_HTML, 'utf8');
}

/* Parses the `const ITEMS = [ ... ];` block of www/index.html and returns
   [{ n, cat }]. A regex is enough: every entry starts `{ n: '...', cat: '...'`. */
function loadItems() {
  const html = readIndexHtml();
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

/* Source text of a top-level declaration in www/index.html's inline script,
   from `head` (e.g. 'function signalKeys(' or 'const Corrections = {') to its
   matching closing brace. Brace counting is enough for the small, string-free
   blocks the tests pull out; it throws if `head` is missing or ambiguous. */
function extractSource(head, html) {
  const src = html || readIndexHtml();
  const start = src.indexOf(head);
  if (start < 0) throw new Error(`${head} not found in www/index.html`);
  if (src.indexOf(head, start + 1) >= 0) throw new Error(`${head} occurs twice in www/index.html`);
  const open = src.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) {
      return src.slice(start, src[i + 1] === ';' ? i + 2 : i + 1);
    }
  }
  throw new Error(`unbalanced braces after ${head}`);
}

module.exports = { loadItems, extractSource, readIndexHtml };
