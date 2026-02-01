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
            .replace(/[()]/g, '')       // remove parentheses to match Astro slugs
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');   // trim leading/trailing hyphens

          // Concepts live under notes/concepts/
          if (slug.startsWith('concept')) {
             return `/notes/concepts/${slug}`;
          }

          // Everything else uses the short alias route
          // (resolved by [...slug].astro's alias paths)
          return `/notes/${slug}`;
        }
      }],
    ],
    rehypePlugins: [
    rehypeKatex,
    ],
  },
});