import fs from 'fs';
import { globby } from 'globby';

// node fix-images.mjs

const CONTENT_DIRS = [
  './src/content/notes',
  './src/content/blog',
  './src/content/french',
  './src/content/research',
];

async function fixImagePaths() {
  const paths = await globby(
    CONTENT_DIRS.map(dir => `${dir}/**/*.md`)
  );

  console.log(`📄 Found ${paths.length} markdown files`);

  paths.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    let hasChanged = false;

    // A. Obsidian wiki image: ![[xxx.png]]
    const wikiLinkRegex = /!\[\[(.*?)\]\]/g;
    const contentAfterWiki = content.replace(wikiLinkRegex, (_, fileName) => {
      hasChanged = true;
      return `![](</images/${fileName.trim()}>)`;
    });

    // B. Fix standard markdown with spaces
    const standardRegex = /!\[(.*?)\]\((?!\s*<)(\/images\/.*?)\)/g;
    const finalContent = contentAfterWiki.replace(
      standardRegex,
      (match, alt, imgPath) => {
        if (imgPath.includes(' ') || imgPath.trim() !== imgPath) {
          hasChanged = true;
          return `![${alt}](<${imgPath.trim()}>)`;
        }
        return match;
      }
    );

    if (hasChanged) {
      fs.writeFileSync(filePath, finalContent, 'utf8');
      console.log(`✅ Refactored: ${filePath}`);
    }
  });
}

fixImagePaths().catch(console.error);