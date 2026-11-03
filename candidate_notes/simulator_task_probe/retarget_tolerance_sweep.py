"""Measure the frozen public-observation reference at stricter tolerances."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_retarget/oracle"))
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_retarget/verify"))
from controller import Policy  # noqa: E402
from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402


def run(first, count):
    rows = []
    for seed in range(first, first + count):
        world = PushEpisode(seed, render=False)
        policy = Policy()
        try:
            obs = world.observe()
            for _ in range(MAX_STEPS):
                obs = world.step(policy.act(obs))
            result = world.result()
            rows.append((result["stage1_error_m"], result["stage2_error_m"]))
        finally:
            world.close()
    thresholds = [0.003, 0.002, 0.00175, 0.0015, 0.00125, 0.001]
    return {
        "first": first, "count": count,
        "successes": {str(t): sum(a < t and b < t for a, b in rows)
                      for t in thresholds},
        "mean_stage1_error_m": sum(a for a, _ in rows) / count,
        "mean_stage2_error_m": sum(b for _, b in rows) / count,
    }


if __name__ == "__main__":
    for first in (7000, 8000):
        print("TOLERANCE_SWEEP", json.dumps(run(first, 50), sort_keys=True), flush=True)
