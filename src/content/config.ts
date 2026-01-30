import { defineCollection, z } from 'astro:content';

// 1. Notes 集合
const notes = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string().optional(),
        course: z.string().optional(),
        type: z.string().optional(),
        created: z.coerce.date().optional(),
    }),
});

// 2. Blog 集合
const blog = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        description: z.string(),
        date: z.coerce.date(), 
    }),
});

// 3. French 集合
const french = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        created: z.coerce.date().optional(), 
        level: z.string().optional(),
        type: z.string().optional(),
        subtype: z.string().optional(),
        module: z.string().optional(),
        status: z.string().optional(),
        tags: z.array(z.string()).optional(), 
    }),
});

// 4. 新增并修复：Projects 集合 (解决 404 的关键)
const projects = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        description: z.string().optional(),
        github: z.string().url(),
        demo: z.string().url().optional(),
        type: z.string().optional(),
        course: z.string().optional(),
        created: z.coerce.date().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

// 5. Research 集合
const research = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        description: z.string().optional(),
        date: z.coerce.date().optional(),
        status: z.string().optional(),
        github: z.string().url().optional(),
        demo: z.string().url().optional(),
        tags: z.array(z.string()).optional(),
    }),
});

// 6. Courses 集合
const courses = defineCollection({
    type: 'content',
    schema: z.object({
        title: z.string(),
        code: z.string().optional(),
        instructor: z.string().optional(),
        description: z.string().optional(),
    }),
});

// 7. 导出所有集合
export const collections = {
    'notes': notes,
    'blog': blog,
    'french': french,
    'projects': projects,
    'research': research,
    'courses': courses,
};
