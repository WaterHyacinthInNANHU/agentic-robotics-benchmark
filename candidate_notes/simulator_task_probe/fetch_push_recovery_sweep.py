"""Evaluate contact recovery and final target stop conditions."""
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/setup/payload"))
from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402


def scaled(delta):
    return max(-1.0, min(1.0, delta / 0.05))


class Policy:
    def __init__(self, lost_distance, stop_gap):
        self.phase = 0
        self.lost_distance = lost_distance
        self.stop_gap = stop_gap

    def act(self, obs):
        gripper = obs["state"][:3]
        cube = obs["state"][3:6]
        goal = obs["goal"]
        dx, dy = goal[0] - cube[0], goal[1] - cube[1]
        gap = math.hypot(dx, dy)
        ux, uy = dx / max(gap, 1e-9), dy / max(gap, 1e-9)
        if self.phase == 2 and math.dist(gripper[:2], cube[:2]) > self.lost_distance:
            self.phase = 0
        behind = [cube[0] - 0.065 * ux, cube[1] - 0.065 * uy, cube[2]]
        if self.phase == 0:
            target = [behind[0], behind[1], behind[2] + 0.10]
            if math.dist(gripper, target) < 0.020:
                self.phase = 1
        elif self.phase == 1:
            target = [behind[0], behind[1], behind[2] + 0.005]
            if math.dist(gripper, target) < 0.015:
                self.phase = 2
        else:
            distance = min(0.05, gap * 0.45) if gap > self.stop_gap else 0.0
            target = [gripper[0] + ux * distance,
                      gripper[1] + uy * distance,
                      cube[2] + 0.005]
        return [scaled(target[i] - gripper[i]) for i in range(3)] + [0.0]


def run(lost_distance, stop_gap, first, count=40):
    wins = 0
    misses = []
    for seed in range(first, first + count):
        world = PushEpisode(seed, render=False)
        policy = Policy(lost_distance, stop_gap)
        for _ in range(MAX_STEPS):
            world.step(policy.act(world.observe()))
        result = world.result()
        wins += int(result["success"])
        if not result["success"]:
            misses.append([seed, round(result["error_m"], 4)])
        world.close()
    return {"lost_distance": lost_distance, "stop_gap": stop_gap,
            "first": first, "wins": wins, "count": count, "misses": misses}


if __name__ == "__main__":
    started = time.perf_counter()
    for lost_distance in (0.08, 0.10):
        for stop_gap in (0.0, 0.015):
            for first in (1000, 2000):
                print(json.dumps(run(lost_distance, stop_gap, first)), flush=True)
    print(f"elapsed_s={time.perf_counter() - started:.1f}", flush=True)
