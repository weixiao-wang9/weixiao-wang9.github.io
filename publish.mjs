#!/usr/bin/env node
/**
 * publish.mjs — one-command knowledge-base pipeline
 *
 * Obsidian vault ──sync──▶ src/content ──repair/index──▶ git commit+push ──▶ GitHub Pages
 *
 * Steps:
 *   1. Mirror-sync markdown from the vault (deletes stale files, honors `publish: false`)
 *   2. Auto-repair frontmatter (title from H1/filename, created from mtime)
 *   3. Sync images + fix Obsidian image embeds
 *   4. Neutralize dead wikilinks (render as plain text instead of 404 links)
 *   5. Auto-generate index (MOC) pages for topic folders + knowledge-map.json
 *   6. Commit & push (triggers GitHub Actions deploy)
 *
 * Usage:
 *   node publish.mjs             # full run
 *   node publish.mjs --dry-run   # report only, change nothing
 *   node publish.mjs --no-git    # sync + organize, but don't commit/push
 */

import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// ================= CONFIG =================
const REPO = path.dirname(fileURLToPath(import.meta.url));
const VAULT = process.env.VAULT_PATH || '/Users/weixiao09/Documents/Obsidian Vault';

// collection -> vault source (collections without an existing source dir are left untouched)
const CONTENT_MAP = {
  notes: 'Public/notes',
  courses: 'Public/courses',
  french: 'Public/French',
  blog: 'Blog',
  research: 'Public/research',
};
const IMAGE_SRC = path.join(VAULT, 'images');
const IMAGE_DEST = path.join(REPO, 'public/images');

const SKIP_DIRS = new Set(['.claude', '.obsidian', '.trash', '_templates']);
const DRY = process.argv.includes('--dry-run');
const NO_GIT = process.argv.includes('--no-git');

const log = (msg) => console.log(msg);
const changes = [];

// ================= HELPERS =================
function walkMd(root) {
  const out = [];
  if (!fs.existsSync(root)) return out;
  for (const e of fs.readdirSync(root, { withFileTypes: true })) {
    if (e.name.startsWith('.') || SKIP_DIRS.has(e.name)) continue;
    const p = path.join(root, e.name);
    if (e.isDirectory()) out.push(...walkMd(p));
    else if (e.name.endsWith('.md')) out.push(p);
  }
  return out;
}

function parseFrontmatter(text) {
  const m = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!m) return { fm: {}, fmRaw: null, body: text };
  const fm = {};
  for (const line of m[1].split('\n')) {
    const i = line.indexOf(':');
    if (i > 0) fm[line.slice(0, i).trim()] = line.slice(i + 1).trim().replace(/^"|"$/g, '');
  }
  return { fm, fmRaw: m[1], body: text.slice(m[0].length) };
}

function titleFromContent(body, filename) {
  const h1 = body.match(/^#\s+(.+)$/m);
  if (h1) return h1[1].trim().replace(/#+$/, '').trim();
  let base = path.basename(filename, '.md')
    .replace(/^\d+[-_ ]*/, '')
    .replace(/^(Concept|Source|Atom|Lecture)[-_ ]+/i, '')
    .replace(/[-_]+/g, ' ')
    .trim();
  return base === base.toLowerCase()
    ? base.replace(/\b\w/g, (c) => c.toUpperCase())
    : base;
}

function slugify(s) {
  return s.toLowerCase().trim()
    .replace(/[()]/g, '')
    .replace(/[\s_-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// ================= 1. SYNC =================
function sync() {
  for (const [dest, src] of Object.entries(CONTENT_MAP)) {
    const srcRoot = path.join(VAULT, src);
    const destRoot = path.join(REPO, 'src/content', dest);
    if (!fs.existsSync(srcRoot)) { log(`⚠️  skip ${dest}: no vault source (${src}) — repo copy preserved`); continue; }
    fs.mkdirSync(destRoot, { recursive: true });

    // copy vault -> repo, honoring publish: false
    const published = new Set();
    for (const p of walkMd(srcRoot)) {
      const text = fs.readFileSync(p, 'utf8');
      const { fm } = parseFrontmatter(text);
      const rel = path.relative(srcRoot, p);
      if (String(fm.publish).toLowerCase() === 'false') continue;
      published.add(rel);
      const target = path.join(destRoot, rel);
      if (!fs.existsSync(target) || fs.readFileSync(target, 'utf8') !== text) {
        changes.push(`sync ${dest}/${rel}`);
        if (!DRY) { fs.mkdirSync(path.dirname(target), { recursive: true }); fs.writeFileSync(target, text); }
      }
    }
    // delete repo files no longer in vault (mirror)
    for (const p of walkMd(destRoot)) {
      const rel = path.relative(destRoot, p);
      if (!published.has(rel)) {
        changes.push(`delete stale ${dest}/${rel}`);
        if (!DRY) fs.rmSync(p);
      }
    }
    if (!DRY) pruneEmptyDirs(destRoot);
  }
}

function pruneEmptyDirs(root) {
  for (const e of fs.readdirSync(root, { withFileTypes: true })) {
    if (!e.isDirectory()) continue;
    const p = path.join(root, e.name);
    pruneEmptyDirs(p);
    if (fs.readdirSync(p).length === 0) fs.rmSync(p, { recursive: true });
  }
}

// ================= 2. FRONTMATTER REPAIR =================
function repairFrontmatter() {
  for (const dest of Object.keys(CONTENT_MAP)) {
    for (const p of walkMd(path.join(REPO, 'src/content', dest))) {
      const text = fs.readFileSync(p, 'utf8');
      const { fm, fmRaw, body } = parseFrontmatter(text);
      const add = [];
      if (!fm.title) add.push(`title: "${titleFromContent(body, p).replace(/"/g, "'")}"`);
      if (dest === 'blog' && !fm.description) {
        const para = body.split('\n').find((l) => l.trim() && !l.startsWith('#') && !l.startsWith('!'));
        add.push(`description: "${(para || 'Blog post').slice(0, 140).replace(/"/g, "'")}"`);
      }
      if (dest === 'blog' && !fm.date) add.push(`date: ${new Date(fs.statSync(p).mtime).toISOString().slice(0, 10)}`);
      if (dest !== 'blog' && !fm.created && !fm.date) add.push(`created: ${new Date(fs.statSync(p).mtime).toISOString().slice(0, 10)}`);
      if (!add.length) continue;
      changes.push(`repair ${path.relative(REPO, p)}: ${add.map((a) => a.split(':')[0]).join(', ')}`);
      if (!DRY) {
        const newFm = (fmRaw ? fmRaw + '\n' : '') + add.join('\n');
        fs.writeFileSync(p, `---\n${newFm}\n---\n${body}`);
      }
    }
  }
}

// ================= 3. IMAGES =================
function syncImages() {
  if (!fs.existsSync(IMAGE_SRC)) return log('⚠️  vault images folder not found, skipping');
  fs.mkdirSync(IMAGE_DEST, { recursive: true });
  const srcFiles = new Set();
  for (const f of fs.readdirSync(IMAGE_SRC)) {
    if (f.startsWith('.')) continue;
    srcFiles.add(f);
    const s = path.join(IMAGE_SRC, f), d = path.join(IMAGE_DEST, f);
    if (!fs.existsSync(d) || fs.statSync(s).mtimeMs > fs.statSync(d).mtimeMs) {
      changes.push(`image ${f}`);
      if (!DRY) fs.copyFileSync(s, d);
    }
  }
  for (const f of fs.readdirSync(IMAGE_DEST)) {
    if (!f.startsWith('.') && !srcFiles.has(f)) {
      changes.push(`delete stale image ${f}`);
      if (!DRY) fs.rmSync(path.join(IMAGE_DEST, f));
    }
  }
}

function fixImagePaths() {
  for (const dest of Object.keys(CONTENT_MAP)) {
    for (const p of walkMd(path.join(REPO, 'src/content', dest))) {
      let text = fs.readFileSync(p, 'utf8');
      const orig = text;
      // ![[image.png]] -> ![](</images/image.png>)
      text = text.replace(/!\[\[(.*?)\]\]/g, (_, f) => `![](</images/${f.trim()}>)`);
      // wrap /images/ paths containing spaces in <>
      text = text.replace(/!\[(.*?)\]\((?!\s*<)(\/images\/.*?)\)/g, (m, alt, ip) =>
        ip.includes(' ') ? `![${alt}](<${ip.trim()}>)` : m);
      if (text !== orig) {
        changes.push(`img-paths ${path.relative(REPO, p)}`);
        if (!DRY) fs.writeFileSync(p, text);
      }
    }
  }
}

// ================= 4. DEAD WIKILINKS =================
function fixDeadLinks() {
  const slugs = new Set();
  for (const dest of Object.keys(CONTENT_MAP)) {
    for (const p of walkMd(path.join(REPO, 'src/content', dest))) {
      slugs.add(slugify(path.basename(p, '.md')));
    }
  }
  for (const dest of Object.keys(CONTENT_MAP)) {
    for (const p of walkMd(path.join(REPO, 'src/content', dest))) {
      let text = fs.readFileSync(p, 'utf8');
      const { fm, fmRaw, body } = parseFrontmatter(text);
      // only rewrite the body — frontmatter [[...]] values (e.g. course) are handled by the site
      const newBody = body.replace(/(?<!!)\[\[([^\]|#]+)(#[^\]|]*)?(\|([^\]]*))?\]\]/g,
        (m, target, _anchor, _pipe, alias) =>
          slugs.has(slugify(target.trim())) ? m : (alias || target).trim());
      if (newBody !== body) {
        changes.push(`unlink dead refs in ${path.relative(REPO, p)}`);
        if (!DRY) fs.writeFileSync(p, (fmRaw !== null ? `---\n${fmRaw}\n---\n` : '') + newBody);
      }
    }
  }
}

// ================= 5. AUTO-GENERATED INDEXES =================
const AUTOGEN = '<!-- AUTO-GENERATED INDEX: edits will be overwritten. Create your own 00-README.md in Obsidian to replace it. -->';

function generateIndexes() {
  const notesRoot = path.join(REPO, 'src/content/notes');
  const map = {};
  for (const e of fs.readdirSync(notesRoot, { withFileTypes: true })) {
    if (!e.isDirectory() || e.name.startsWith('.') || SKIP_DIRS.has(e.name)) continue;
    const topicDir = path.join(notesRoot, e.name);
    const files = walkMd(topicDir)
      .filter((p) => !/00-README\.md$/i.test(p))
      .sort();
    const entries = files.map((p) => {
      const { fm, body } = parseFrontmatter(fs.readFileSync(p, 'utf8'));
      return {
        file: path.relative(topicDir, p),
        title: fm.title || titleFromContent(body, p),
        type: fm.type || 'note',
        slug: slugify(path.basename(p, '.md')),
      };
    });
    map[e.name] = entries;

    const readmePath = path.join(topicDir, '00-README.md');
    const existing = fs.existsSync(readmePath) ? fs.readFileSync(readmePath, 'utf8') : null;
    if (existing && !existing.includes('AUTO-GENERATED INDEX')) continue; // hand-written — leave alone
    const lines = [
      '---',
      `title: "${e.name} — Index"`,
      'type: index',
      '---',
      AUTOGEN,
      '',
      `# ${e.name}`,
      '',
      `${entries.length} notes in this topic:`,
      '',
      ...entries.map((n) => `- [[${path.basename(n.file, '.md')}|${n.title}]]`),
      '',
    ];
    const content = lines.join('\n');
    if (existing !== content) {
      changes.push(`index ${e.name}/00-README.md`);
      if (!DRY) fs.writeFileSync(readmePath, content);
    }
  }
  const mapPath = path.join(REPO, 'src/data/knowledge-map.json');
  const mapJson = JSON.stringify(map, null, 2);
  if (!fs.existsSync(mapPath) || fs.readFileSync(mapPath, 'utf8') !== mapJson) {
    changes.push('knowledge-map.json');
    if (!DRY) fs.writeFileSync(mapPath, mapJson);
  }
}

// ================= 6. GIT =================
function gitPublish() {
  const run = (cmd) => execSync(cmd, { cwd: REPO, stdio: 'pipe' }).toString().trim();
  const status = run('git status --porcelain -- src/content public/images src/data');
  if (!status) return log('✅ Nothing new to publish.');
  run('git add src/content public/images src/data');
  const stamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
  run(`git commit -m "KB sync ${stamp}"`);
  log('📦 Committed. Pushing…');
  try {
    run('git push');
    log('🚀 Pushed — GitHub Actions is deploying.');
  } catch {
    log('⚠️  Push failed (offline or auth). Commit is saved locally; run `git push` later.');
  }
}

// ================= RUN =================
log(`🔄 KB pipeline ${DRY ? '(dry run)' : ''}\n   vault: ${VAULT}\n   repo : ${REPO}\n`);
sync();
repairFrontmatter();
syncImages();
fixImagePaths();
fixDeadLinks();
generateIndexes();
log(`\n${changes.length} change(s)` + (changes.length ? ':\n  ' + changes.slice(0, 30).join('\n  ') + (changes.length > 30 ? `\n  … +${changes.length - 30} more` : '') : ''));
if (!DRY && !NO_GIT) gitPublish();
log('\n🏁 Done.');
