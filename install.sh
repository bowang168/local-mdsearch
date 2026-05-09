#!/usr/bin/env bash
# install.sh — install local-mdsearch as a Claude Code skill.
#
# Resolves {{MDSEARCH_HOME}} in SKILL.md to the absolute path of this repo
# and writes the rendered skill to ${CLAUDE_HOME:-$HOME/.claude}/skills/mdsearch/.
#
# Idempotent: re-run after pulling updates.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/SKILL.md"
CLAUDE_HOME="${CLAUDE_HOME:-$HOME/.claude}"
DEST_DIR="$CLAUDE_HOME/skills/mdsearch"
DEST="$DEST_DIR/SKILL.md"

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found." >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

# Substitute {{MDSEARCH_HOME}} with the absolute repo path.
# Use awk over sed to sidestep delimiter conflicts when the path has slashes.
awk -v home="$HERE" '{ gsub(/\{\{MDSEARCH_HOME\}\}/, home); print }' "$SRC" > "$DEST"

echo "✓ Installed skill: $DEST"
echo "  pointing at: $HERE"
echo
echo "Next steps:"
echo "  - Ensure Ollama is running:   ollama serve &"
echo "  - Ensure model is pulled:     ollama pull qwen3-embedding:0.6b"
echo "  - In Claude Code, invoke:     /mdsearch  (or use a trigger phrase)"
