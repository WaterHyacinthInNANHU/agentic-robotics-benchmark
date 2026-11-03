#!/usr/bin/env bash
cd /home/user/submission 2>/dev/null || cd /tmp/square_nut_assembly_blind_01/submission
exec python3 controller.py
