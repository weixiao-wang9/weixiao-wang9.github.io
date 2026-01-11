// astro.config.mjs
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// 1. 确保引入了这两个插件
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
  site: 'https://weixiao-wang9.github.io',
  vite: {
    plugins: [tailwindcss()],
  },
  // 2. 确保配置了 markdown 选项
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});