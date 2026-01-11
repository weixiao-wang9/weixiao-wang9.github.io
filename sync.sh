#!/bin/bash

# ================= 1. 你的配置 (Configuration) =================

# 你的 Obsidian 仓库根目录
VAULT="/Users/weixiao09/Documents/Obsidian Vault"

# --- 源路径设置 (Source Paths) ---
# 1. French: 你之前的日志显示它在 public 下，所以保持不变
SOURCE_FRENCH="$VAULT/public/French"

# 2. Blog & Notes: 之前的日志提示找不到 "/public/Blog"，说明它们可能在根目录
# 我去掉了中间的 "/public"，如果还在报错，请手动确认它们在 Obsidian 里的位置
SOURCE_BLOG="$VAULT/Blog"
SOURCE_NOTES="$VAULT/Notes"

# ================= 2. 脚本逻辑 (Do not edit) =================

echo "🔄 Starting Sync..."

# 【关键修复】: 强制创建 src/content 目录
# 只要这一行在，就不会报 "src/content/french: No such file" 的错
mkdir -p src/content

# 定义一个安全的同步函数
sync_folder() {
    src="$1"
    dest_name="$2"
    dest_path="src/content/$dest_name"

    # 检查源文件夹是否存在
    if [ -d "$src" ]; then
        echo "👉 Syncing $dest_name..."
        # 先删除旧的 (确保彻底同步)
        rm -rf "$dest_path"
        # 复制新的
        cp -R "$src" "$dest_path"
        echo "   ✅ Success!"
    else
        echo "⚠️  Skipped $dest_name: Source folder not found at $src"
    fi
}

# 执行同步
sync_folder "$SOURCE_FRENCH" "french"
sync_folder "$SOURCE_BLOG"   "blog"
sync_folder "$SOURCE_NOTES"  "notes"

echo "🏁 All Done!"