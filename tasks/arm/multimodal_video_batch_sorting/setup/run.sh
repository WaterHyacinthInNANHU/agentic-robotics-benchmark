#!/usr/bin/env bash
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
python3 "$(dirname "$0")/generate_scenes.py" --output "$HOME_DIR/scenes"
for entry in payload/* payload/.[!.]*; do [ -e "$entry" ] || continue; chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"; done
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/scenes"
