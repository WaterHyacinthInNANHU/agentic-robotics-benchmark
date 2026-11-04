"""Measure native robosuite PickPlaceCan expert timing and sample cameras."""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import robosuite
from PIL import Image

OUT = Path(os.environ.get("PROBE_OUT", "/tmp/rapid-grasp-probe"))
OUT.mkdir(parents=True, exist_ok=True)


def clip_action(raw, target, grip, gain=0.035, speed=1.0):
    e = np.asarray(raw["robot0_eef_pos"], dtype=float)
    xyz = np.clip((np.asarray(target) - e) / gain, -speed, speed)
    return np.r_[xyz, [0., 0., 0., float(grip)]]


def run(seed):
    env = robosuite.make(
        env_name="PickPlaceCan",
        robots="Panda",
        seed=seed,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=[128, 128],
        camera_widths=[128, 128],
        camera_depths=False,
        reward_shaping=False,
        control_freq=20,
        horizon=160,
        hard_reset=False,
    )
    raw = env.reset()
    if seed == 11:
        print("OBS_KEYS", list(raw.keys()), flush=True)
        print("ACTION_SPEC", [x.tolist() for x in env.action_spec], flush=True)
        print("CAMERAS", [env.sim.model.camera_id2name(i) for i in range(env.sim.model.ncam)], flush=True)
        print("GEOMS", [env.sim.model.geom_id2name(i) for i in range(env.sim.model.ngeom) if "finger" in (env.sim.model.geom_id2name(i) or "").lower() or "pad" in (env.sim.model.geom_id2name(i) or "").lower()], flush=True)
        for camera in ("agentview", "robot0_eye_in_hand"):
            Image.fromarray(raw[camera + "_image"][::-1]).save(OUT / f"{camera}_reset.png")
    rest_z = float(env.sim.data.body_xpos[env.obj_body_id["Can"]][2])
    target_bin = np.array([
        env.bin2_pos[0] + env.bin_size[0] / 4,
        env.bin2_pos[1] + env.bin_size[1] / 4,
        env.bin2_pos[2] + 0.06,
    ])
    stage, stage_start = "hover", 0
    transition = []
    success_step = None
    history = []
    for t in range(160):
        obj = np.asarray(env.sim.data.body_xpos[env.obj_body_id["Can"]], dtype=float).copy()
        e = np.asarray(raw["robot0_eef_pos"], dtype=float)
        age = t - stage_start
        if stage == "hover":
            target = obj + [0, 0, .16]
            if np.linalg.norm(target - e) < .025 or age > 30:
                stage, stage_start = "descend", t
            action = clip_action(raw, target, -1)
        elif stage == "descend":
            target = obj + [0, 0, .025]
            if np.linalg.norm(target - e) < .015 or age > 30:
                stage, stage_start = "close", t
            action = clip_action(raw, target, -1, speed=.8)
        elif stage == "close":
            if age >= 8:
                stage, stage_start = "lift", t
            action = clip_action(raw, e, 1)
        elif stage == "lift":
            target = e + [0, 0, .12]
            if obj[2] > rest_z + .075 or age > 25:
                stage, stage_start = "carry", t
            action = clip_action(raw, target, 1)
        elif stage == "carry":
            target_obj = target_bin + [0, 0, .09]
            target = e + target_obj - obj
            if np.linalg.norm(obj[:2] - target_obj[:2]) < .035 or age > 50:
                stage, stage_start = "lower", t
            action = clip_action(raw, target, 1)
        elif stage == "lower":
            target = e + target_bin - obj
            if np.linalg.norm(obj[:2] - target_bin[:2]) < .025 and obj[2] < target_bin[2] + .035 or age > 25:
                stage, stage_start = "release", t
            action = clip_action(raw, target, 1, speed=.7)
        elif stage == "release":
            if age >= 8:
                stage, stage_start = "retreat", t
            action = clip_action(raw, e, -1)
        else:
            action = clip_action(raw, e + [0, 0, .12], -1)
        if not transition or transition[-1][1] != stage:
            transition.append((t, stage))
        raw, reward, _, _ = env.step(action)
        if t % 10 == 0 or env._check_success():
            history.append({
                "t": t + 1, "stage": stage, "obj": obj.round(3).tolist(),
                "eef": e.round(3).tolist(), "gripper": np.asarray(raw["robot0_gripper_qpos"]).round(3).tolist(),
                "success": bool(env._check_success()), "contacts": int(env.sim.data.ncon),
            })
        if seed == 11 and t in (20, 40, 60, 80, 100):
            for camera in ("agentview", "robot0_eye_in_hand"):
                Image.fromarray(raw[camera + "_image"][::-1]).save(OUT / f"{camera}_{t+1:03d}.png")
        if env._check_success():
            success_step = t + 1
            break
    result = {"seed": seed, "success_step": success_step, "success_seconds": None if success_step is None else success_step / 20, "transitions": transition, "history": history}
    print("RESULT", json.dumps(result), flush=True)
    env.close()
    return result


if __name__ == "__main__":
    results = [run(seed) for seed in (11, 22, 33)]
    (OUT / "results.json").write_text(json.dumps(results, indent=2))

