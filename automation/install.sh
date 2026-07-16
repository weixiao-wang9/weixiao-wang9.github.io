#!/bin/zsh
# One-time setup: installs the daily auto-publish job (launchd).
# Run:  zsh automation/install.sh
set -e
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.wei.kb-publish.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.wei.kb-publish.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
cp "$PLIST_SRC" "$PLIST_DEST"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo "✅ Installed. The knowledge base now publishes automatically every day at 22:00 (and at login)."
echo "   Logs:      ~/Library/Logs/kb-publish.log"
echo "   Run now:   launchctl start com.wei.kb-publish"
echo "   Uninstall: launchctl unload $PLIST_DEST && rm $PLIST_DEST"
