import { defineCollection, z } from 'astro:content';

// 1. 你原来的 Notes 集合 (保持不变)
const notes = defineCollection({
    // ... 你原来的代码 ...
});

// 2. 新增：Blog 集合
const blog = defineCollection({
	type: 'content',
	schema: z.object({
		title: z.string(),
		description: z.string(),
		// 强制要求日期，方便排序
		date: z.coerce.date(), 
	}),
});
const french = defineCollection({
	type: 'content',
	schema: z.object({
		title: z.string(),
        // 对应 Obsidian 的 "created" (有些是 date 类型，有些是 string，这里用 coerce 强制转)
		created: z.coerce.date().optional(), 
        
        // 核心属性
        level: z.string().optional(),   // e.g. "A1"
        type: z.string().optional(),    // e.g. "atom"
        subtype: z.string().optional(), // e.g. "grammar"
        
        // 标签 (Obsidian 的 tags 是数组)
        tags: z.array(z.string()).optional(), 
	}),
});

// 3. 导出所有集合
export const collections = { notes, blog };