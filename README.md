# Wei's Knowledge Base & Portfolio

Astro site published to GitHub Pages, fed automatically from my Obsidian vault.

## The workflow

```
Obsidian vault                 this repo                        live site
─────────────                  ─────────                        ─────────
Public/notes    ──┐
Public/courses    │  publish.mjs   src/content/*   git push    GitHub Actions
Public/French   ──┼─────────────▶  public/images  ──────────▶  builds & deploys
Blog              │  (daily via     src/data/                  to GitHub Pages
images          ──┘   launchd)
```

**Input** — write notes in Obsidian. Anything under `Public/` (and `Blog/`) gets published. Add `publish: false` to a note's frontmatter to keep it private.

**Organize** — `publish.mjs` runs automatically every day at 22:00 (launchd) and:

1. Mirror-syncs markdown from the vault (stale files are deleted, `publish: false` respected)
2. Repairs frontmatter — missing `title` (from H1/filename), `created`, blog `description`/`date`
3. Syncs images and converts `![[image.png]]` embeds
4. Converts wikilinks pointing at unpublished notes into plain text (no dead links)
5. Auto-generates a `00-README.md` index (MOC) for topic folders that lack one, plus `src/data/knowledge-map.json`

**Output** — commits and pushes; GitHub Actions (`.github/workflows/deploy.yml`) builds and deploys.

## Commands

| Command | Action |
| :-- | :-- |
| `npm run publish` | Sync + organize + commit + push (full pipeline) |
| `npm run sync` | Sync + organize only, no git |
| `node publish.mjs --dry-run` | Show what would change |
| `npm run dev` | Preview at `localhost:4321` |
| `npm run build` | Build to `./dist/` |
| `zsh automation/install.sh` | Install the daily auto-publish job (one time) |

## Conventions

- **Note frontmatter**: `title`, `type` (`source` / `atom` / `concept` / `index`), `course: "[[Course Name]]"`, `created`. The pipeline fills in missing `title`/`created`, but the `_templates/Source.md` template in the vault sets them up correctly from the start.
- **Topic indexes**: a hand-written `00-README.md` in a topic folder is never touched; if a folder has none, the pipeline generates one (marked `AUTO-GENERATED`).
- **Repo-only collections**: `research` and `projects` live only in this repo and are never deleted by the sync.
- **Courses**: one file per course in `Public/courses/` (`title`, `code`, `instructor`) — the Library page groups notes by matching `course` frontmatter.
