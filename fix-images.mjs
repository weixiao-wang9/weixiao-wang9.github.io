import fs from 'fs';
import path from 'path';
import { globby } from 'globby'; 

// node fix-images.mjs

// 1. 设置笔记所在的文件夹路径
const NOTES_DIR = [
  './src/content/notes',
  './src/content/blog',
];



async function fixImagePaths() {
  // 查找所有 markdown 文件
  const paths = await globby(`${NOTES_DIR}/**/*.md`);

  paths.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    let hasChanged = false;

    // 逻辑 A: 处理 Obsidian 风格 ![[Screenshot xxx.png]] 
    // 转换为 Astro 兼容的 ![](</images/Screenshot xxx.png>)
    const wikiLinkRegex = /!\[\[(.*?)\]\]/g;
    const contentAfterWiki = content.replace(wikiLinkRegex, (match, fileName) => {
      hasChanged = true;
      // 这里的 /images/ 路径应根据你 public 下的实际存放位置调整
      return `![](</images/${fileName.trim()}>)`;
    });

    // 逻辑 B: 修复标准 Markdown ![]( /images/xxx.png ) 
    // 自动添加 < > 并清理路径首尾多余空格
    const standardRegex = /!\[(.*?)\]\((?!\s*<)(\/images\/.*?)\)/g;
    const finalContent = contentAfterWiki.replace(standardRegex, (match, alt, imgPath) => {
      if (imgPath.includes(' ') || imgPath.trim() !== imgPath) {
        hasChanged = true;
        return `![${alt}](<${imgPath.trim()}>)`;
      }
      return match;
    });

    if (hasChanged) {
      fs.writeFileSync(filePath, finalContent, 'utf8');
      console.log(`✅ Refactored: ${filePath}`);
    }
  });
}

fixImagePaths().catch(console.error);