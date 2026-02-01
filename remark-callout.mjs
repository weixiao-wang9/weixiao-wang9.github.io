/**
 * Custom remark plugin for Obsidian callouts.
 * Unlike remark-obsidian-callout, this plugin preserves MDAST child nodes
 * so that remarkMath / rehypeKatex can still process $...$ inside callouts.
 *
 * Must run BEFORE remarkMath in the plugin chain.
 */
import { visit } from 'unist-util-visit';

// Callout icons (inline SVG)
const icons = {
  note: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="2" x2="22" y2="6"/><path d="M7.5 20.5 19 9l-4-4L3.5 16.5 2 22z"/></svg>',
  abstract: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><line x1="12" y1="11" x2="12" y2="17"/><line x1="9" y1="14" x2="15" y2="14"/></svg>',
  info: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  tip: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2"/><path d="m9 12 2 2 4-4"/></svg>',
  success: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
  question: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  warning: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
  danger: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
  bug: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="14" x="8" y="6" rx="4"/><path d="m19 7-3 2"/><path d="m5 7 3 2"/><path d="m19 19-3-2"/><path d="m5 19 3-2"/><path d="M20 13h-4"/><path d="M4 13h4"/><path d="m10 4 1 2"/><path d="m14 4-1 2"/></svg>',
  example: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
  quote: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3z"/></svg>',
};

// Map aliases to canonical types
const typeAliases = {
  summary: 'abstract', tldr: 'abstract',
  hint: 'tip', important: 'tip',
  check: 'success', done: 'success',
  help: 'question', faq: 'question',
  attention: 'warning', caution: 'warning',
  error: 'danger',
  missing: 'failure', fail: 'failure',
  cite: 'quote',
  todo: 'info',
  definition: 'note',
};

function getIcon(type) {
  const canonical = typeAliases[type] || type;
  return icons[canonical] || icons.note;
}

const CALLOUT_REGEX = /^\[\!(\w+)\]([+-]?)\s*(.*)?$/;

export default function remarkCallout() {
  return (tree) => {
    visit(tree, 'blockquote', (node) => {
      if (!node.children || node.children.length === 0) return;

      const firstChild = node.children[0];
      if (firstChild.type !== 'paragraph' || !firstChild.children) return;

      // The first inline child should be text starting with [!type]
      const firstInline = firstChild.children[0];
      if (!firstInline || firstInline.type !== 'text') return;

      const firstLineEnd = firstInline.value.indexOf('\n');
      const firstLine = firstLineEnd === -1 ? firstInline.value : firstInline.value.slice(0, firstLineEnd);

      const match = firstLine.match(CALLOUT_REGEX);
      if (!match) return;

      const calloutType = match[1].toLowerCase();
      const expandSign = match[2];
      const titleText = match[3] || calloutType.charAt(0).toUpperCase() + calloutType.slice(1);

      // Build data attributes on the blockquote
      const dataExpandable = Boolean(expandSign);
      const dataExpanded = expandSign === '+';

      node.data = {
        hProperties: {
          ...(node.data?.hProperties || {}),
          className: `callout-${calloutType}`,
          'data-callout': calloutType,
          'data-expandable': String(dataExpandable),
          'data-expanded': String(dataExpanded),
        },
      };

      // --- Reconstruct children preserving MDAST nodes ---

      // 1. Title HTML node (icon + title text)
      const titleNode = {
        type: 'html',
        value: `<div class="callout-title"><div class="callout-title-icon">${getIcon(calloutType)}</div><div class="callout-title-text">${titleText}</div></div>`,
      };

      // 2. Content: preserve remaining inline children from first paragraph
      //    plus all subsequent blockquote children
      const contentChildren = [];

      // Remaining text after the [!type] line in the first text node
      if (firstLineEnd !== -1) {
        const remaining = firstInline.value.slice(firstLineEnd + 1);
        if (remaining.trim()) {
          // Rebuild the first paragraph with remaining text + other inline nodes
          const newFirstChildren = [
            { type: 'text', value: remaining },
            ...firstChild.children.slice(1),
          ];
          contentChildren.push({
            type: 'paragraph',
            children: newFirstChildren,
          });
        } else {
          // The rest of the first text node was empty, but there might be more inline nodes
          if (firstChild.children.length > 1) {
            contentChildren.push({
              type: 'paragraph',
              children: firstChild.children.slice(1),
            });
          }
        }
      } else {
        // No newline in first text node — check if there are more inline children
        // (e.g., [!note] followed by math nodes in the same paragraph)
        if (firstChild.children.length > 1) {
          // The remaining inline nodes after the [!type] text
          const restInlines = firstChild.children.slice(1);
          // Check if first remaining is just whitespace text
          if (restInlines.length > 0) {
            contentChildren.push({
              type: 'paragraph',
              children: restInlines,
            });
          }
        }
      }

      // All subsequent children of the blockquote (paragraphs, lists, etc.)
      for (let i = 1; i < node.children.length; i++) {
        contentChildren.push(node.children[i]);
      }

      // 3. Wrap content in a div
      const contentOpenNode = { type: 'html', value: '<div class="callout-content">' };
      const contentCloseNode = { type: 'html', value: '</div>' };

      // Replace blockquote children
      node.children = [
        titleNode,
        contentOpenNode,
        ...contentChildren,
        contentCloseNode,
      ];
    });
  };
}
