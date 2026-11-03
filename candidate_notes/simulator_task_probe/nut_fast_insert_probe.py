"""Matched-seed probe for a 170-action native square-nut insertion task."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_square_nut_assembly/verify"))
sys.path.insert(0, str(ROOT / "candidate_notes/simulator_task_probe/nut_insert_blind_02"))
from square_nut_env import SquareNutEpisode  # noqa: E402
from controller import Policy as Baseline  # noqa: E402
from nut_fast_insert_controller import Policy as Fast  # noqa: E402


def run(seed, policy_type):
    world = SquareNutEpisode(seed, render=False)
    policy = policy_type()
    obs = world.observe()
    try:
        for t in range(170):
            obs, done = world.step(policy.step(t, obs))
            if done:
                break
        return {"seed": seed, "success": bool(world.success), "steps": world.t,
                "stage": policy.stage, "nut_z": obs["nut_pos"][2]}
    finally:
        world.close()


if __name__ == "__main__":
    for name, policy in (("baseline", Baseline), ("fast", Fast)):
        rows = [run(seed, policy) for seed in range(200, 240)]
        print("NUT_FAST_INSERT_RESULT", json.dumps({"policy": name,
              "successes": sum(r["success"] for r in rows), "episodes": len(rows),
              "failures": [r for r in rows if not r["success"]]}, sort_keys=True), flush=True)
