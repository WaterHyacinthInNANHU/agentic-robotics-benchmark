#!/usr/bin/env bash
set -euo pipefail
cd /home/user/submission 2>/dev/null || cd /tmp/nut_handle_carry_blind_01/submission
exec python3 -u controller.py
