// astro.config.mjs
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import wikiLinkPlugin from 'remark-wiki-link';
import remarkCallout from './remark-callout.mjs';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkCite from './remark-cite.mjs';

export default defineConfig({
  site: 'https://weixiao-wang9.github.io',
  
  vite: {
    plugins: [
      tailwindcss(), // 确保你的陶土橙主题和极简排版生效
    ],
  },

  markdown: {

    image: {
      service: "astro/assets/services/noop",
    },
    
    remarkPlugins: [
      remarkCallout,
      remarkMath,
      remarkCite,
      [wikiLinkPlugin, {
        aliasDivider: '|',
        hrefTemplate: (permalink) => {
          // 1. 标准化处理：转小写，将所有空格、下划线、连续横杠统一为单横杠
          // 这能完美匹配你之前批量重命名后的文件格式
          const slug = permalink.toLowerCase()
            .trim()
            .replace(/[\s_-]+/g, '-'); 

          // 2. 动态路由映射
          
          // 情况 A: 这是一个 Concept (原子笔记)
          // 假设它们统一存放在 src/content/notes/concepts/ 下
          if (slug.startsWith('concept')) {
             return `/notes/concepts/${slug}`;
          }
          
          // 情况 B: 这是一个 Source (课程大课笔记)
          // 为了解决不可扩展问题，我们不再硬编码文件夹名
          // 而是生成一个虚拟路径 "/notes/find/source-xxx"
          // 我们将在 src/pages/notes/[...slug].astro 中捕获并处理这个路径
          if (slug.startsWith('source')) {
             return `/notes/find/${slug}`;
          }

          // 情况 C: 兜底路径（针对普通笔记或其他分类）
          return `/notes/${slug}`;
        }
      }],
    ],
    rehypePlugins: [
    rehypeKatex,
    ],
  },
});