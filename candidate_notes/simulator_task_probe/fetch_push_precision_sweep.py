"""Measure stricter final-position tolerances on untouched seed ranges."""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/setup/payload"))
from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402
from fetch_push_recovery_sweep import Policy  # noqa: E402


def trial(stop_gap, first, count=80):
    errors = []
    for seed in range(first, first + count):
        world = PushEpisode(seed, render=False)
        policy = Policy(0.10, stop_gap)
        for _ in range(MAX_STEPS):
            world.step(policy.act(world.observe()))
        errors.append(world.result()["error_m"])
        world.close()
    ordered = sorted(errors)
    return {"stop_gap": stop_gap, "first": first, "count": count,
            "under_20mm": sum(e < 0.020 for e in errors),
            "under_10mm": sum(e < 0.010 for e in errors),
            "under_5mm": sum(e < 0.005 for e in errors),
            "under_2mm": sum(e < 0.002 for e in errors),
            "p90_mm": round(ordered[math.ceil(0.9 * count) - 1] * 1000, 2),
            "max_mm": round(ordered[-1] * 1000, 2)}


if __name__ == "__main__":
    started = time.perf_counter()
    for stop_gap in (0.0, 0.005, 0.010, 0.015):
        for first in (3000, 4000):
            print(json.dumps(trial(stop_gap, first)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
