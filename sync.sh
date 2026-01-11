#!/bin/bash

# ================= 1. 你的配置 (Configuration) =================

# 你的 Obsidian 仓库根目录
VAULT="/Users/weixiao09/Documents/Obsidian Vault"

# --- 源路径设置 (Source Paths) ---

# 1. Public 文件夹 (我们约定的发布区)
PUBLIC_DIR="$VAULT/Public"

# [关键修改]: 现在去 Public 里找 notes 和 courses
SOURCE_NOTES="$PUBLIC_DIR/notes"
SOURCE_COURSES="$PUBLIC_DIR/courses"  # 👈 新增：课程字典

# 2. 其他内容 (根据你现有的位置)
SOURCE_FRENCH="$PUBLIC_DIR/French"    # 假设 French 也在 Public 下
SOURCE_BLOG="$VAULT/Blog"             # Blog 依然保持在你原来的位置

# ================= 2. 脚本逻辑 (Do not edit) =================

echo "🔄 Starting Sync..."

# 强制创建 src/content 目录 (防止第一次运行报错)
mkdir -p src/content

# 定义同步函数
sync_folder() {
    src="$1"
    dest_name="$2"
    dest_path="src/content/$dest_name"

    # 检查源文件夹是否存在
    if [ -d "$src" ]; then
        echo "👉 Syncing $dest_name..."
        # 1. 清理旧数据 (防止删掉的文件还留在网站上)
        rm -rf "$dest_path"
        # 2. 复制新数据
        cp -R "$src" "$dest_path"
        echo "   ✅ Success! ($src -> $dest_path)"
    else
        echo "⚠️  Skipped $dest_name: Source folder not found at $src"
    fi
}

# --- 执行同步 ---

# 1. 核心笔记系统 (Notes + Courses)
sync_folder "$SOURCE_NOTES"   "notes"
sync_folder "$SOURCE_COURSES" "courses" # 👈 这一步至关重要

# 2. 其他板块
sync_folder "$SOURCE_FRENCH"  "french"
sync_folder "$SOURCE_BLOG"    "blog"

echo "🏁 All Done! Now run: npm run publish"