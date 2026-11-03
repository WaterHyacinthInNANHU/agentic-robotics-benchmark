#!/usr/bin/env bash
set -euo pipefail
task_home="${ALE_HOME:-/home/user}"
task_user="$(basename "$task_home")"
cp -r payload/. "$task_home/"
for entry in payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$task_user:$task_user" "$task_home/$(basename "$entry")"
done
