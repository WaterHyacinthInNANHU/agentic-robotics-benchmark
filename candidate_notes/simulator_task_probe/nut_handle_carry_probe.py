"""Observation-only recovery controller for a native square-nut handle carry."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_square_nut_assembly/verify"))
from square_nut_env import SquareNutEpisode  # noqa: E402

HORIZON = 400


def clip(value):
    return max(-1.0, min(1.0, value))


def move(eef, target, gripper, limit=1.0):
    return [max(-limit, min(limit, 12 * (target[i] - eef[i])))
            for i in range(3)] + [0, 0, 0, gripper]


class Policy:
    def __init__(self, carry_limit=0.15):
        self.carry = False
        self.grasp_step = None
        self.carry_limit = carry_limit

    def act(self, t, obs):
        eef = obs["eef_pos"]
        nut = obs["nut_pos"]
        handle = obs["handle_pos"]
        peg = obs["peg_pos"]
        if self.carry and nut[2] < 0.92:
            self.carry = False
        if nut[2] > 0.95 and t % 100 >= 95:
            if not self.carry:
                self.grasp_step = t
            self.carry = True
        if self.carry:
            target = [eef[0] + peg[0] - nut[0],
                      eef[1] + peg[1] - nut[1], 1.15]
            return move(eef, target, 1, limit=self.carry_limit)
        attempt = min(t // 100, 2)
        local = t - attempt * 100
        rotate_sign = -1 if attempt == 0 else 1
        if local < 6:
            return [0, 0, 0, 0, 0, rotate_sign, -1]
        if local < 36:
            return move(eef, [handle[0], handle[1], handle[2] + 0.075], -1)
        if local < 65:
            return move(eef, [handle[0], handle[1], handle[2] - 0.005], -1)
        if local < 80:
            return [0, 0, 0, 0, 0, 0, 1]
        return move(eef, [eef[0], eef[1], 1.08], 1)


def run(seed, carry_limit):
    world = SquareNutEpisode(seed, render=False)
    world.env.horizon = HORIZON
    policy = Policy(carry_limit)
    try:
        for t in range(HORIZON):
            world.step(policy.act(t, world.observe()))
        final = world.observe()
        nut = final["nut_pos"]
        peg = final["peg_pos"]
        xy_error = math.hypot(nut[0] - peg[0], nut[1] - peg[1])
        return {"seed": seed, "success": xy_error < 0.03 and nut[2] > 1.0,
                "xy_error_m": xy_error, "nut_z": nut[2],
                "grasp_step": policy.grasp_step}
    finally:
        world.close()


if __name__ == "__main__":
    for carry_limit in (0.25, 0.35):
        rows = []
        for seed in range(180, 220):
            row = run(seed, carry_limit)
            rows.append(row)
        print("NUT_CARRY_RESULT", json.dumps({
            "carry_limit": carry_limit, "count": len(rows),
            "successes": sum(row["success"] for row in rows),
            "grasped": sum(row["grasp_step"] is not None for row in rows),
            "fail_seeds": [row["seed"] for row in rows if not row["success"]],
        }, sort_keys=True), flush=True)
