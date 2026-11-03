"""Two late-revealed targets in one native FetchPush-v4 MuJoCo scene."""
import base64
import io

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from PIL import Image

gym.register_envs(gymnasium_robotics)

STEPS_PER_GOAL = 200
MAX_STEPS = 2 * STEPS_PER_GOAL
PRECISION_METRES = 0.0015


class PushEpisode:
    def __init__(self, seed, render=True):
        self.render_images = bool(render)
        self.env = gym.make(
            "FetchPush-v4",
            render_mode="rgb_array" if render else None,
            max_episode_steps=MAX_STEPS,
        )
        self.raw, _ = self.env.reset(seed=int(seed))
        self.steps = 0
        self.stage = 0
        self.goal = self.raw["desired_goal"].copy()
        self.stage1_error = None
        self._cached = None

    def observe(self):
        if self._cached is None:
            image = ""
            if self.render_images:
                frame = Image.fromarray(self.env.render())
                frame = frame.resize((160, 160), Image.Resampling.BILINEAR)
                buffer = io.BytesIO()
                frame.save(buffer, format="PNG")
                image = base64.b64encode(buffer.getvalue()).decode("ascii")
            self._cached = {
                "state": self.raw["observation"].tolist(),
                "goal": self.goal.tolist(),
                "stage": self.stage,
                "stage_step": self.steps % STEPS_PER_GOAL,
                "rgb_png_base64": image,
            }
        return self._cached

    def step(self, action):
        vector = np.asarray(action, dtype=float)
        if vector.shape != (4,) or not np.isfinite(vector).all():
            raise ValueError("action must contain four finite numbers")
        self.raw, _, _, _, _ = self.env.step(np.clip(vector, -1, 1))
        self.steps += 1
        if self.steps == STEPS_PER_GOAL:
            self.stage1_error = float(np.linalg.norm(self.raw["achieved_goal"] - self.goal))
            first_goal = self.goal.copy()
            for _ in range(100):
                second_goal = self.env.unwrapped._sample_goal().copy()
                if np.linalg.norm(second_goal[:2] - first_goal[:2]) > 0.10:
                    break
            self.goal = second_goal
            self.env.unwrapped.goal = second_goal.copy()
            self.stage = 1
        self._cached = None
        return self.observe()

    def result(self):
        error = float(np.linalg.norm(self.raw["achieved_goal"] - self.goal))
        first_ok = self.stage1_error is not None and self.stage1_error < PRECISION_METRES
        second_ok = error < PRECISION_METRES
        return {
            "success": bool(first_ok and second_ok),
            "stage1_success": bool(first_ok),
            "stage2_success": bool(second_ok),
            "stage1_error_m": self.stage1_error,
            "stage2_error_m": error,
            "steps": self.steps,
        }

    def close(self):
        self.env.close()
