"""Diagnose native robosuite NutAssemblySquare handle grasps."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_square_nut_assembly/verify"))
from square_nut_env import SquareNutEpisode  # noqa: E402


def clip(value):
    return max(-1.0, min(1.0, value))


def move(eef, target, gripper):
    return [clip(12 * (target[i] - eef[i])) for i in range(3)] + [0, 0, 0, gripper]


def run(seed, yaw_steps, height_offset):
    world = SquareNutEpisode(seed, render=False)
    try:
        first = world.observe()
        nut_start_z = first["nut_pos"][2]
        close_pose = None
        for t in range(125):
            obs = world.observe()
            eef = obs["eef_pos"]
            handle = obs["handle_pos"]
            if t < abs(yaw_steps):
                action = [0, 0, 0, 0, 0, 1 if yaw_steps > 0 else -1, -1]
            elif t < abs(yaw_steps) + 30:
                action = move(eef, [handle[0], handle[1], handle[2] + 0.075], -1)
            elif t < abs(yaw_steps) + 60:
                action = move(eef, [handle[0], handle[1], handle[2] + height_offset], -1)
            elif t < abs(yaw_steps) + 78:
                if close_pose is None:
                    close_pose = {"eef": eef, "quat": obs["eef_quat"],
                                  "handle": handle, "nut": obs["nut_pos"]}
                action = [0, 0, 0, 0, 0, 0, 1]
            else:
                action = move(eef, [eef[0], eef[1], handle[2] + 0.18], 1)
            world.step(action)
        final = world.observe()
        return {"seed": seed, "yaw_steps": yaw_steps,
                "height_offset": height_offset,
                "lift_m": final["nut_pos"][2] - nut_start_z,
                "eef_z": final["eef_pos"][2],
                "nut_z": final["nut_pos"][2],
                "close_pose": close_pose}
    finally:
        world.close()


if __name__ == "__main__":
    for yaw_steps in (-6, -3, 0, 3, 6):
        for height_offset in (-0.005, 0.015, 0.035):
            rows = [run(seed, yaw_steps, height_offset) for seed in (11, 22, 33)]
            print("NUT_GRASP_PROBE", json.dumps({
                "yaw_steps": yaw_steps, "height_offset": height_offset,
                "grasped": sum(row["lift_m"] > 0.04 for row in rows),
                "lift_m": [round(row["lift_m"], 4) for row in rows],
                "close_quat": rows[0]["close_pose"]["quat"],
                "close_eef": rows[0]["close_pose"]["eef"],
                "close_handle": rows[0]["close_pose"]["handle"],
            }, sort_keys=True), flush=True)
