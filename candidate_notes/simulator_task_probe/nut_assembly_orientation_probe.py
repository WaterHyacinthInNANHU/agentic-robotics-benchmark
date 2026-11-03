"""Test native square-nut assembly after several handle-grasp orientations."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_square_nut_assembly/verify"))
from square_nut_env import MAX_STEPS, SquareNutEpisode  # noqa: E402


def clip(value):
    return max(-1.0, min(1.0, value))


def move(eef, target, gripper):
    return [clip(12 * (target[i] - eef[i])) for i in range(3)] + [0, 0, 0, gripper]


def run(seed, yaw_steps):
    world = SquareNutEpisode(seed, render=False)
    first = world.observe()
    start_z = first["nut_pos"][2]
    lifted_z = None
    min_peg_distance = 1.0
    try:
        for t in range(MAX_STEPS):
            obs = world.observe()
            eef, nut, handle, peg = (obs[key] for key in
                                      ("eef_pos", "nut_pos", "handle_pos", "peg_pos"))
            if t < abs(yaw_steps):
                action = [0, 0, 0, 0, 0, 1 if yaw_steps > 0 else -1, -1]
            elif t < abs(yaw_steps) + 29:
                action = move(eef, [handle[0], handle[1], handle[2] + 0.075], -1)
            elif t < abs(yaw_steps) + 58:
                action = move(eef, [handle[0], handle[1], handle[2] - 0.005], -1)
            elif t < 82:
                action = [0, 0, 0, 0, 0, 0, 1]
            elif t < 108:
                action = move(eef, [eef[0], eef[1], max(eef[2], nut[2] + 0.16)], 1)
            elif t < 145:
                action = move(eef, [eef[0] + peg[0] - nut[0],
                                    eef[1] + peg[1] - nut[1],
                                    max(eef[2], nut[2] + 0.14)], 1)
            elif t < 166:
                action = move(eef, [eef[0] + peg[0] - nut[0],
                                    eef[1] + peg[1] - nut[1],
                                    eef[2] + 0.86 - nut[2]], 1)
            elif t < 181:
                action = [0, 0, 0, 0, 0, 0, -1]
            else:
                action = move(eef, [eef[0], eef[1], max(eef[2], 1.08)], -1)
            world.step(action)
            if t == 107:
                lifted_z = world.observe()["nut_pos"][2]
            min_peg_distance = min(min_peg_distance,
                                   ((nut[0] - peg[0]) ** 2 +
                                    (nut[1] - peg[1]) ** 2) ** 0.5)
        end = world.observe()
        return {"seed": seed, "yaw_steps": yaw_steps,
                "success": world.success,
                "lifted_m": lifted_z - start_z,
                "min_peg_distance_m": min_peg_distance,
                "end_nut": end["nut_pos"], "peg": end["peg_pos"]}
    finally:
        world.close()


if __name__ == "__main__":
    totals = {str(yaw): 0 for yaw in (-6, 0, 6)}
    for seed in range(11, 31):
        rows = [run(seed, yaw) for yaw in (-6, 0, 6)]
        for row in rows:
            totals[str(row["yaw_steps"])] += bool(row["success"])
        print("NUT_ASSEMBLY_SEED", json.dumps({"seed": seed,
              "choices": [{"yaw": row["yaw_steps"], "success": row["success"],
                           "lifted_m": round(row["lifted_m"], 3),
                           "min_peg_distance_m": round(row["min_peg_distance_m"], 3),
                           "end_nut": [round(v, 3) for v in row["end_nut"]]}
                          for row in rows]}, sort_keys=True), flush=True)
    print("NUT_ASSEMBLY_COUNTS", json.dumps(totals, sort_keys=True), flush=True)
