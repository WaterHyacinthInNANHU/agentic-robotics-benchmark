"""Probe two consecutive precision pushes in one native MuJoCo scene."""
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


def run(first, count=40, steps_per_goal=160, lost_distance=0.10):
    rows = []
    for seed in range(first, first + count):
        env = gym.make("FetchPush-v4", max_episode_steps=2 * steps_per_goal)
        raw, _ = env.reset(seed=seed)
        first_goal = raw["desired_goal"].copy()
        # Keep the second waypoint inside the simulator's own goal distribution.
        for _ in range(100):
            second_goal = env.unwrapped._sample_goal().copy()
            if np.linalg.norm(second_goal[:2] - first_goal[:2]) > 0.10:
                break
        errors = []
        for goal in (first_goal, second_goal):
            policy = Policy(lost_distance, 0.0)
            for _ in range(steps_per_goal):
                obs = {"state": raw["observation"].tolist(), "goal": goal.tolist()}
                raw, _, _, _, _ = env.step(policy.act(obs))
            errors.append(float(np.linalg.norm(raw["achieved_goal"] - goal)))
        rows.append({"seed": seed, "e1": errors[0], "e2": errors[1]})
        env.close()
    return {"first": first, "count": count, "recovery": math.isfinite(lost_distance),
            "both_5mm": sum(r["e1"] < .005 and r["e2"] < .005 for r in rows),
            "both_3mm": sum(r["e1"] < .003 and r["e2"] < .003 for r in rows),
            "both_2mm": sum(r["e1"] < .002 and r["e2"] < .002 for r in rows),
            "stage1_3mm": sum(r["e1"] < .003 for r in rows),
            "stage2_3mm": sum(r["e2"] < .003 for r in rows),
            "fails_3mm": [[r["seed"], round(r["e1"], 4), round(r["e2"], 4)]
                          for r in rows if r["e1"] >= .003 or r["e2"] >= .003]}


if __name__ == "__main__":
    started = time.perf_counter()
    for lost_distance in (0.10, float("inf")):
        for first in (3000, 4000):
            print(json.dumps(run(first, lost_distance=lost_distance)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
