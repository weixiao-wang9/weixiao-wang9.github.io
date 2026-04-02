#!/bin/bash

# ================= 1. 你的配置 (Configuration) =================

# 你的 Obsidian 仓库根目录
VAULT="/Users/weixiao09/Documents/Obsidian Vault"

# --- 源路径设置 (Source Paths) ---
PUBLIC_DIR="$VAULT/Public"

# 笔记与课程逻辑
SOURCE_NOTES="$PUBLIC_DIR/notes"
SOURCE_COURSES="$PUBLIC_DIR/courses"

# 其他内容
SOURCE_FRENCH="$PUBLIC_DIR/French"
SOURCE_BLOG="$VAULT/Blog"
SOURCE_RESEARCH="$PUBLIC_DIR/research"

# 图片存放路径
SOURCE_PIC="$VAULT/images"

# ================= 2. 脚本逻辑 (Do not edit) =================

echo "🔄 Starting Sync..."

# 确保目标目录存在
mkdir -p src/content
mkdir -p public/images

# --- 函数 A: 同步 Markdown 内容到 src/content ---
# 用 rsync 替代 rm + cp：只覆盖 Obsidian 中有的文件，保留本地独有的文件
sync_content() {
    src="$1"
    dest_name="$2"
    dest_path="src/content/$dest_name"

    if [ -d "$src" ]; then
        echo "👉 Syncing Content: $dest_name..."
        mkdir -p "$dest_path"
        rsync -a --update "$src/" "$dest_path/"
        # 清理 .DS_Store 和空目录
        find "$dest_path" -name '.DS_Store' -delete 2>/dev/null
        find "$dest_path" -type d -empty -delete 2>/dev/null
        echo "   ✅ Success!"
    else
        echo "⚠️  Skipped $dest_name: Source not found at $src"
    fi
}

# --- 函数 B: 同步图片到根目录的 public/images ---
sync_assets() {
    src="$1"
    dest_path="./public/images" # 👈 修复：确保是在根目录的 public 下

    if [ -d "$src" ]; then
        echo "🖼️  Syncing Images to $dest_path..."
        # 清理旧图片，确保文件名变更后不会留下残留
        rm -rf "$dest_path"/*
        # 复制所有图片
        cp -R "$src/"* "$dest_path/"
        echo "   ✅ Images Updated!"
    else
        echo "⚠️  Images Skipped: Source folder $src not found"
    fi
}

# ================= 3. 执行同步 =================

# 1. 执行文本内容同步
sync_content "$SOURCE_NOTES"   "notes"
sync_content "$SOURCE_COURSES" "courses"
sync_content "$SOURCE_FRENCH"  "french"
sync_content "$SOURCE_BLOG"    "blog"
sync_content "$SOURCE_RESEARCH" "research"

# 2. 执行资源文件同步
sync_assets "$SOURCE_PIC"

echo "---------------------------------------"
echo "🏁 Sync Complete!"
echo "🛠️  Running Image Path Fixer..."

# 3. 自动运行你的路径修复脚本
if [ -f "fix-images.mjs" ]; then
    node fix-images.mjs
else
    echo "❌ Error: fix-images.mjs not found. Paths not fixed."
fi

echo "🚀 System Ready. Run 'npm run dev' to preview."