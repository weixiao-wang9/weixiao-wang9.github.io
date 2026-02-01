// remark-cite.mjs
// Lightweight remark plugin that:
//   1. Converts Pandoc-style [@key] citations to CSL-formatted in-text citations
//      e.g. APA: (Rizvi et al., 2017), IEEE: [1], etc.
//   2. Auto-generates a "References" section formatted via CSL
//
// Uses @citation-js/core + plugins for proper CSL formatting.
// Initializes once per build (cached), avoiding the per-file citeproc overhead
// that caused rehype-citation to OOM.
//
// To change citation style: replace apa.csl with another CSL file.

import { visit } from 'unist-util-visit';
import { Cite } from '@citation-js/core';
import '@citation-js/plugin-bibtex';
import '@citation-js/plugin-csl';
import fs from 'fs';
import path from 'path';

const BIB_PATH = path.resolve('reference.bib');
const CSL_PATH = path.resolve('apa.csl');

// Module-level cache: parsed once, reused across all markdown files
let bibData = null;   // Array of CSL-JSON items from reference.bib
let cslStyle = null;  // CSL XML string
let itemMap = null;    // Map of citation-key -> CSL-JSON item

function loadBib() {
  if (bibData) return;
  const bibtex = fs.readFileSync(BIB_PATH, 'utf-8');
  const cite = new Cite(bibtex);
  bibData = cite.data;

  // Build lookup map
  itemMap = {};
  for (const item of bibData) {
    const id = item.id || item['citation-key'];
    itemMap[id] = item;
  }

  if (fs.existsSync(CSL_PATH)) {
    cslStyle = fs.readFileSync(CSL_PATH, 'utf-8');
  }
}

// Get CSL format options
function cslOpts() {
  const opts = { format: 'html', lang: 'en-US' };
  if (cslStyle) opts.template = cslStyle;
  return opts;
}

// Format an in-text citation for one or more keys, e.g. (Rizvi et al., 2017)
function formatInTextCitation(keys) {
  const items = keys.map(k => itemMap[k]).filter(Boolean);
  if (items.length === 0) return `(???)`;
  const cite = new Cite(items);
  return cite.format('citation', cslOpts());
}

// Format bibliography for a set of keys
function formatBibliography(keys) {
  const items = keys.map(k => itemMap[k]).filter(Boolean);
  const cite = new Cite(items);
  return cite.format('bibliography', cslOpts());
}

export default function remarkCite() {
  return (tree, file) => {
    // Collect citation keys in order of first appearance
    const citationOrder = [];
    const citationSet = new Set();

    // Also collect per-occurrence groups for in-text formatting
    // Each occurrence is { keys: [...], position info }
    const citePattern = /\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]/g;
    const keyPattern = /@([\w-]+)/g;

    // First pass: collect all citation keys in order
    visit(tree, 'text', (node) => {
      const text = node.value;
      let m;
      citePattern.lastIndex = 0;
      while ((m = citePattern.exec(text)) !== null) {
        const inner = m[1];
        keyPattern.lastIndex = 0;
        let km;
        while ((km = keyPattern.exec(inner)) !== null) {
          const key = km[1];
          if (!citationSet.has(key)) {
            citationSet.add(key);
            citationOrder.push(key);
          }
        }
      }
    });

    if (citationOrder.length === 0) return;

    // Load bib + CSL data (cached across files)
    loadBib();

    // Second pass: replace citation text nodes with CSL-formatted in-text citations
    visit(tree, 'text', (node, index, parent) => {
      const text = node.value;
      citePattern.lastIndex = 0;
      if (!citePattern.test(text)) return;

      citePattern.lastIndex = 0;
      const children = [];
      let lastIndex = 0;
      let m;

      citePattern.lastIndex = 0;
      while ((m = citePattern.exec(text)) !== null) {
        if (m.index > lastIndex) {
          children.push({ type: 'text', value: text.slice(lastIndex, m.index) });
        }

        // Extract keys for this citation group
        const inner = m[1];
        keyPattern.lastIndex = 0;
        const keys = [];
        let km;
        while ((km = keyPattern.exec(inner)) !== null) {
          keys.push(km[1]);
        }

        // Format via CSL engine
        const inText = formatInTextCitation(keys);

        // Link to the first cited key's reference entry
        const firstKey = keys[0];
        children.push({
          type: 'html',
          value: `<a href="#ref-${firstKey}" class="citation-link">${inText}</a>`,
        });

        lastIndex = m.index + m[0].length;
      }

      if (lastIndex < text.length) {
        children.push({ type: 'text', value: text.slice(lastIndex) });
      }

      parent.children.splice(index, 1, ...children);
    });

    // Generate references section via citeproc
    const bibHtml = formatBibliography(citationOrder);

    // Parse citeproc output and add anchor IDs keyed by citation-key
    // citeproc returns: <div class="csl-bib-body"><div data-csl-entry-id="key" class="csl-entry">...</div>...</div>
    const entryRegex = /data-csl-entry-id="([^"]*)"[^>]*>([\s\S]*?)<\/div>/g;
    let entryMatch;
    const entryMap = {};
    while ((entryMatch = entryRegex.exec(bibHtml)) !== null) {
      entryMap[entryMatch[1]] = entryMatch[2].trim();
    }

    // APA uses hanging-indent unordered list; other styles may differ
    let refsHtml = '<section class="references">\n<h2>References</h2>\n<div class="csl-bib-body">\n';
    for (const key of citationOrder) {
      const formatted = entryMap[key] || `<em>Reference not found: ${key}</em>`;
      refsHtml += `<div id="ref-${key}" class="csl-entry">${formatted}</div>\n`;
    }
    refsHtml += '</div>\n</section>';

    tree.children.push({
      type: 'html',
      value: refsHtml,
    });
  };
}
