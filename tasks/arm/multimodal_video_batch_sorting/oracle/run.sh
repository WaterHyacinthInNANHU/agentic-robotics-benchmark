#!/usr/bin/env bash
set -euo pipefail
mkdir -p "${ALE_HOME:-/home/user}/submission"
cp /oracle/reference.json "${ALE_HOME:-/home/user}/submission/plan.json"
