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
from fetch_push_recovery_sweep import Policy, scaled  # noqa: E402


class WaypointPolicy(Policy):
    def act(self, obs):
        if self.phase == -1:
            gripper = obs["state"][:3]
            cube = obs["state"][3:6]
            target_height = cube[2] + 0.15
            if abs(gripper[2] - target_height) < 0.015:
                self.phase = 0
            else:
                return [0.0, 0.0, scaled(target_height - gripper[2]), 0.0]
        return super().act(obs)


def run(first, count=40, steps_per_goal=160, lost_distance=0.10):
    rows = []
    for seed in range(first, first + count):
        env = gym.make("FetchPush-v4", max_episode_steps=2 * steps_per_goal)
        raw, _ = env.reset(seed=seed)
        first_goal = raw["desired_goal"].copy()
        for _ in range(100):
            second_goal = env.unwrapped._sample_goal().copy()
            if np.linalg.norm(second_goal[:2] - first_goal[:2]) > 0.10:
                break
        errors = []
        for stage, goal in enumerate((first_goal, second_goal)):
            policy = WaypointPolicy(lost_distance, 0.0)
            if stage == 1:
                policy.phase = -1
            for _ in range(steps_per_goal):
                obs = {"state": raw["observation"].tolist(), "goal": goal.tolist()}
                raw, _, _, _, _ = env.step(policy.act(obs))
            errors.append(float(np.linalg.norm(raw["achieved_goal"] - goal)))
        rows.append({"seed": seed, "e1": errors[0], "e2": errors[1],
                     "separation": float(np.linalg.norm(first_goal - second_goal))})
        env.close()
    return {"first": first, "count": count, "steps_per_goal": steps_per_goal,
            "recovery": math.isfinite(lost_distance),
            "both_5mm": sum(r["e1"] < .005 and r["e2"] < .005 for r in rows),
            "both_3mm": sum(r["e1"] < .003 and r["e2"] < .003 for r in rows),
            "both_2mm": sum(r["e1"] < .002 and r["e2"] < .002 for r in rows),
            "stage1_3mm": sum(r["e1"] < .003 for r in rows),
            "stage2_3mm": sum(r["e2"] < .003 for r in rows),
            "mean_separation_m": sum(r["separation"] for r in rows) / len(rows),
            "fails_3mm": [[r["seed"], round(r["e1"], 4), round(r["e2"], 4)]
                          for r in rows if r["e1"] >= .003 or r["e2"] >= .003]}


if __name__ == "__main__":
    started = time.perf_counter()
    for steps_per_goal in (160, 200, 240):
        for first in (5000, 6000):
            print(json.dumps(run(first, count=50, steps_per_goal=steps_per_goal)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
