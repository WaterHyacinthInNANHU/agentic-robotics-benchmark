"""Probe physical square-nut insertion after an observation-only grasp policy."""
import json
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
    phase = "carry"
    phase_start = 0
    aligned = 0
    native_stable = 0
    ever_native = False
    obs = world.observe()
    try:
        for t in range(400):
            nut, peg, eef = (obs[k] for k in ("nut_pos", "peg_pos", "eef_pos"))
            xy = distance(nut[:2], peg[:2])
            if phase == "carry":
                action = ctl.act(t, obs)
                aligned = aligned + 1 if ctl.phase == "carry" and xy < 0.012 and nut[2] > 1.03 else 0
                if aligned >= 8:
                    phase, phase_start = "lower", t
            elif phase == "lower":
                target = [eef[0] + peg[0] - nut[0], eef[1] + peg[1] - nut[1],
                          eef[2] + target_nut_z - nut[2]]
                action = ctl.command(obs, target, 1., ctl.desired_ori, speed=.42)
                action[2] = max(-.32, min(.32, action[2]))
                if nut[2] < target_nut_z + .012 or t - phase_start > 72:
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
                "xy_error": distance(nut[:2], peg[:2]), "nut_z": nut[2]}
    finally:
        world.close()


if __name__ == "__main__":
    for z in (0.84, 0.86, 0.88):
        rows = [run(seed, z) for seed in range(11, 31)]
        print("NUT_INSERT_RESULT", json.dumps({
            "target_nut_z": z, "successes": sum(r["success"] for r in rows),
            "ever_native": sum(r["ever_native"] for r in rows),
            "failures": [r for r in rows if not r["success"]],
        }, sort_keys=True), flush=True)
