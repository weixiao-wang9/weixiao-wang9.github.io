// astro.config.mjs
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import wikiLinkPlugin from 'remark-wiki-link';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  // ... 其他配置 (site, vite 等)
  site: 'https://weixiao-wang9.github.io',
  vite: {
    plugins: [tailwindcss()], // 确保样式依然生效
  },

  markdown: {
    remarkPlugins: [
      remarkMath,
      [wikiLinkPlugin, {
        aliasDivider: '|',
        hrefTemplate: (permalink) => {
          // 1. 标准化处理：转小写，空格变横杠
          const slug = permalink.toLowerCase()
            .trim()
            .replace(/[\s_-]+/g, '-'); 

          // 2. 智能文件夹匹配
          
          // 情况 A: 这是一个 Concept (Atom)
          if (slug.startsWith('concept')) {
             return `/notes/concepts/${slug}`;
          }
          
          // 情况 B: 这是一个 Source (Lecture Note)
          // ⚠️ 这里是修复重点：确保它指向正确的课程子文件夹
          if (slug.startsWith('source')) {
             // 只有加上文件夹前缀，链接才能生效
             return `/notes/computer-networks-cs-6250/${slug}`;
          }

          // 情况 C: 兜底路径
          return `/notes/${slug}`;
        }
      }],
    ],
    rehypePlugins: [rehypeKatex],
  },
});