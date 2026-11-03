#!/usr/bin/env bash
set -euo pipefail
task_home="${ALE_HOME:-/home/user}"
mkdir -p "$task_home/submission"
cp "$(dirname "$0")/controller.py" "$task_home/submission/controller.py"
cp "$(dirname "$0")/submission_run.sh" "$task_home/submission/run.sh"
chown -R "$(basename "$task_home"):$(basename "$task_home")" "$task_home/submission"
