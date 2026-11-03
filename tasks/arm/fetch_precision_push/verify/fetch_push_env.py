"""Public wrapper for Gymnasium Robotics FetchPush-v4."""
import base64
import io

import gymnasium as gym
import gymnasium_robotics
import numpy as np
from PIL import Image

gym.register_envs(gymnasium_robotics)

MAX_STEPS = 80
PRECISION_METRES = 0.02


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
        self.last_native_success = False
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
                "goal": self.raw["desired_goal"].tolist(),
                "rgb_png_base64": image,
            }
        return self._cached

    def step(self, action):
        vector = np.asarray(action, dtype=float)
        if vector.shape != (4,) or not np.isfinite(vector).all():
            raise ValueError("action must contain four finite numbers")
        self.raw, _, _, _, info = self.env.step(np.clip(vector, -1, 1))
        self.steps += 1
        self.last_native_success = bool(info.get("is_success", False))
        self._cached = None
        return self.observe()

    def result(self):
        error = float(np.linalg.norm(self.raw["achieved_goal"] - self.raw["desired_goal"]))
        return {
            "success": error < PRECISION_METRES,
            "error_m": error,
            "native_success": self.last_native_success,
            "steps": self.steps,
        }

    def close(self):
        self.env.close()
