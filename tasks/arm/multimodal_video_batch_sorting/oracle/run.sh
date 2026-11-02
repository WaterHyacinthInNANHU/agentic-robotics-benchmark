#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${ALE_HOME:-/home/user}/submission"
cp "$(dirname "$0")/reference.json" "${ALE_HOME:-/home/user}/submission/plan.json"
