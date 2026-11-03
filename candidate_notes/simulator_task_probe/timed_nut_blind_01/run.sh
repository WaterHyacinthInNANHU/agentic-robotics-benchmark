#!/bin/sh
cd /home/user/submission 2>/dev/null || cd "$(dirname "$0")"
exec python3 controller.py
