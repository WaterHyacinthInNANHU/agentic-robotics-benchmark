"""Public physics wrapper for a handle carry in stock NutAssemblySquare."""
import base64
import io
import random

import numpy as np
import robosuite
from PIL import Image

MAX_STEPS = 400


class NutCarryEpisode:
    def __init__(self, seed, render=True):
        self.seed = int(seed)
        self.render = bool(render)
        self.env = robosuite.make(
            env_name="NutAssemblySquare",
            robots="Panda",
            seed=self.seed,
            has_renderer=False,
            has_offscreen_renderer=self.render,
            use_camera_obs=self.render,
            use_object_obs=True,
            camera_names="agentview",
            camera_heights=128,
            camera_widths=128,
            camera_depths=False,
            reward_shaping=False,
            control_freq=20,
            horizon=MAX_STEPS,
            hard_reset=False,
        )
        self.bias = np.array([random.Random(self.seed ^ (i + 1) * 127).uniform(-0.003, 0.003)
                              for i in range(3)])
        self.t = 0
        self.success = False
        self.stable_steps = 0
        self.raw = self.env.reset()

    def observe(self):
        raw = self.raw
        nut = self.env.nuts[0]
        handle_id = self.env.sim.model.site_name2id(nut.important_sites["handle"])
        handle = np.asarray(self.env.sim.data.site_xpos[handle_id])
        peg = np.asarray(self.env.sim.data.body_xpos[self.env.peg1_body_id])
        rgb = ""
        if self.render:
            buffer = io.BytesIO()
            Image.fromarray(raw["agentview_image"][::-1]).save(buffer, format="PNG")
            rgb = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "eef_pos": raw["robot0_eef_pos"].tolist(),
            "eef_quat": raw["robot0_eef_quat"].tolist(),
            "joint_pos": raw["robot0_joint_pos"].tolist(),
            "gripper_qpos": raw["robot0_gripper_qpos"].tolist(),
            "nut_pos": (raw["SquareNut_pos"] + self.bias).tolist(),
            "nut_quat": raw["SquareNut_quat"].tolist(),
            "handle_pos": (handle + self.bias).tolist(),
            "peg_pos": peg.tolist(),
            "rgb_png_base64": rgb,
        }

    def step(self, action):
        array = np.asarray(action, dtype=float)
        if array.shape != (7,) or not np.isfinite(array).all():
            raise ValueError("action must be seven finite numbers")
        low, high = self.env.action_spec
        self.raw, _, _, _ = self.env.step(np.clip(array, low, high))
        self.t += 1
        true_nut = np.asarray(self.raw["SquareNut_pos"])
        peg = np.asarray(self.env.sim.data.body_xpos[self.env.peg1_body_id])
        nut = self.env.nuts[0]
        handle_id = self.env.sim.model.site_name2id(nut.important_sites["handle"])
        handle = np.asarray(self.env.sim.data.site_xpos[handle_id])
        eef = np.asarray(self.raw["robot0_eef_pos"])
        good = (np.linalg.norm(true_nut[:2] - peg[:2]) < 0.03
                and true_nut[2] > 1.0
                and np.linalg.norm(eef - handle) < 0.08)
        self.stable_steps = self.stable_steps + 1 if good else 0
        self.success = self.stable_steps >= 10
        return self.observe(), self.success or self.t >= MAX_STEPS

    def result(self):
        nut = np.asarray(self.raw["SquareNut_pos"])
        peg = np.asarray(self.env.sim.data.body_xpos[self.env.peg1_body_id])
        return {
            "success": bool(self.success),
            "xy_error_m": float(np.linalg.norm(nut[:2] - peg[:2])),
            "nut_height_m": float(nut[2]),
            "stable_steps": self.stable_steps,
            "steps": self.t,
        }

    def close(self):
        self.env.close()
