"""Probe whether longer control horizons improve millimetre precision."""
import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_push_recovery_sweep import Policy  # noqa: E402


def trial(horizon, first, count=80, lost_distance=0.10):
    errors = []
    for seed in range(first, first + count):
        env = gym.make("FetchPush-v4", max_episode_steps=horizon)
        raw, _ = env.reset(seed=seed)
        policy = Policy(lost_distance, 0.0)
        for _ in range(horizon):
            obs = {"state": raw["observation"].tolist(),
                   "goal": raw["desired_goal"].tolist()}
            raw, _, _, _, _ = env.step(policy.act(obs))
        errors.append(float(np.linalg.norm(raw["achieved_goal"] - raw["desired_goal"])))
        env.close()
    ordered = sorted(errors)
    return {"horizon": horizon, "first": first, "count": count,
            "recovery": math.isfinite(lost_distance),
            "under_20mm": sum(e < 0.020 for e in errors),
            "under_5mm": sum(e < 0.005 for e in errors),
            "under_3mm": sum(e < 0.003 for e in errors),
            "under_2mm": sum(e < 0.002 for e in errors),
            "p90_mm": round(ordered[math.ceil(0.9 * count) - 1] * 1000, 2),
            "max_mm": round(ordered[-1] * 1000, 2)}


if __name__ == "__main__":
    started = time.perf_counter()
    for lost_distance in (0.10, float("inf")):
        for horizon in (80, 160):
            for first in (3000, 4000):
                print(json.dumps(trial(horizon, first, lost_distance=lost_distance)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
