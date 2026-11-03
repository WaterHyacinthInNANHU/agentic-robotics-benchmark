"""Probe physical square-nut insertion after an observation-only grasp policy."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_nut_handle_carry/verify"))
sys.path.insert(0, str(ROOT / "candidate_notes/simulator_task_probe/nut_carry_blind_02"))
from nut_carry_env import NutCarryEpisode  # noqa: E402
from controller import Controller, distance  # noqa: E402


def run(seed, target_nut_z):
    world = NutCarryEpisode(seed, render=False)
    ctl = Controller()
    phase = "grasp"
    phase_start = 0
    aligned = 0
    native_stable = 0
    ever_native = False
    obs = world.observe()
    try:
        for t in range(400):
            nut, peg, eef, handle = (obs[k] for k in
                                     ("nut_pos", "peg_pos", "eef_pos", "handle_pos"))
            xy = distance(nut[:2], peg[:2])
            handle_angle = math.atan2(handle[1] - nut[1], handle[0] - nut[0])
            yaw_error = round(handle_angle / (math.pi / 2)) * (math.pi / 2) - handle_angle
            if phase == "grasp":
                action = ctl.act(t, obs)
                if ctl.phase == "carry" and nut[2] > 1.03:
                    phase, phase_start = "align", t
            elif phase == "align":
                target = [eef[0] + peg[0] - nut[0], eef[1] + peg[1] - nut[1],
                          eef[2] + 1.08 - nut[2]]
                action = ctl.command(obs, target, 1., ctl.desired_ori, speed=.55)
                action[5] = max(-.35, min(.35, 1.2 * yaw_error))
                aligned = aligned + 1 if xy < .005 and abs(yaw_error) < .08 else 0
                if aligned >= 7:
                    phase, phase_start = "lower", t
            elif phase == "lower":
                target = [eef[0] + peg[0] - nut[0], eef[1] + peg[1] - nut[1],
                          eef[2] + target_nut_z - nut[2]]
                action = ctl.command(obs, target, 1., ctl.desired_ori, speed=.35)
                action[2] = max(-.2, min(.2, action[2]))
                action[5] = max(-.2, min(.2, .8 * yaw_error))
                if nut[2] < target_nut_z + .015 or t - phase_start > 100:
                    phase, phase_start = "release", t
            elif phase == "release":
                action = ctl.command(obs, list(eef), -1., ctl.desired_ori, speed=.35)
                if t - phase_start >= 18:
                    phase, phase_start = "retreat", t
            else:
                action = ctl.command(obs, [eef[0], eef[1], 1.12], -1., ctl.desired_ori, speed=.5)
            obs, _ = world.step(action)
            native = bool(world.env._check_success())
            ever_native |= native
            native_stable = native_stable + 1 if native else 0
            if native_stable >= 10:
                break
        nut, peg = obs["nut_pos"], obs["peg_pos"]
        return {"seed": seed, "target_nut_z": target_nut_z, "success": native_stable >= 10,
                "ever_native": ever_native, "phase": phase, "steps": world.t,
                "xy_error": distance(nut[:2], peg[:2]), "nut_z": nut[2],
                "yaw_error": yaw_error}
    finally:
        world.close()


if __name__ == "__main__":
    for z in (0.83,):
        rows = [run(seed, z) for seed in range(11, 31)]
        print("NUT_INSERT_RESULT", json.dumps({
            "target_nut_z": z, "successes": sum(r["success"] for r in rows),
            "ever_native": sum(r["ever_native"] for r in rows),
            "failures": [r for r in rows if not r["success"]],
        }, sort_keys=True), flush=True)
