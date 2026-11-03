"""Probe reference-controller contact offsets on independent FetchPush seeds."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/setup/payload"))
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/oracle"))

from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402
import controller  # noqa: E402


def run(offset, seed_start, count):
    controller.OFFSET = offset
    wins = 0
    misses = []
    for seed in range(seed_start, seed_start + count):
        world = PushEpisode(seed, render=False)
        policy = controller.Policy()
        for _ in range(MAX_STEPS):
            world.step(policy.act(world.observe()))
        result = world.result()
        wins += int(result["success"])
        if not result["success"]:
            misses.append([seed, round(result["error_m"], 4)])
        world.close()
    return {"offset": offset, "start": seed_start, "wins": wins, "count": count, "misses": misses}


if __name__ == "__main__":
    started = time.perf_counter()
    for offset in (0.045, 0.055, 0.065, 0.075, 0.085):
        for first in (1000, 2000):
            print(json.dumps(run(offset, first, 40)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
