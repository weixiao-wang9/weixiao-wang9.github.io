import fs from 'fs';
import path from 'path';
import { globby } from 'globby'; // 如果没有请执行 npm install globby

// 1. 设置笔记所在的文件夹路径
const NOTES_DIR = './src/content/notes';

async function fixImagePaths() {
  // 查找所有 markdown 文件
  const paths = await globby(`${NOTES_DIR}/**/*.md`);

  paths.forEach(filePath => {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // 正则表达式逻辑：
    // 匹配 ![](/images/...) 这种格式，但排除掉已经加了 < > 的
    // 并自动将其转换为 ![](</images/...>)
    const fixedContent = content.replace(/!\[(.*?)\]\((?!\s*<)(\/images\/.*?)\)/g, (match, alt, imgPath) => {
      // 只有当路径包含空格时才处理，或者全部加括号以保安全
      if (imgPath.includes(' ')) {
        return `![${alt}](<${imgPath.trim()}>)`;
      }
      return match;
    });

    if (content !== fixedContent) {
      fs.writeFileSync(filePath, fixedContent, 'utf8');
      console.log(`✅ Fixed: ${filePath}`);
    }
  });
}

fixImagePaths().catch(console.error);