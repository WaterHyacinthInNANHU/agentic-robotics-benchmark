"""Rasterize MuJoCo finger-pad contact forces in the wrist camera image frame.

This is a camera-aligned display of physical contact points and normal forces,
not an optical-flow or learned pseudo-tactile signal.
"""
import numpy as np
import mujoco
from PIL import Image
from robosuite.utils.camera_utils import get_camera_transform_matrix


def _pad_contact_names(env, contact):
    model = env.sim.model
    a = model.geom_id2name(int(contact.geom1)) or ""
    b = model.geom_id2name(int(contact.geom2)) or ""
    return a, b


def contact_heatmap(env, height=128, width=128):
    """Return uint8 force heatmap and count of finger-pad contacts."""
    sim = env.sim
    transform = get_camera_transform_matrix(sim, "robot0_eye_in_hand", height, width)
    heat = np.zeros((height, width), dtype=np.float32)
    yy, xx = np.mgrid[:height, :width]
    count = 0
    for idx in range(sim.data.ncon):
        contact = sim.data.contact[idx]
        a, b = _pad_contact_names(env, contact)
        if not ("pad_collision" in a or "pad_collision" in b):
            continue
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, idx, force)
        normal_newtons = max(0.0, float(force[0]))
        if normal_newtons <= 0:
            continue
        point = transform @ np.r_[np.asarray(contact.pos), 1.0]
        if point[2] <= 1e-6:
            continue
        x, y = point[0] / point[2], point[1] / point[2]
        if not (-6 <= x < width + 6 and -6 <= y < height + 6):
            continue
        amplitude = min(255.0, 75.0 * np.log1p(normal_newtons))
        heat += amplitude * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3.0 ** 2))
        count += 1
    return np.clip(heat, 0, 255).astype(np.uint8), count


def render_contact_overlay(wrist_rgb, heatmap):
    """Return RGB image with red contact highlights aligned to wrist RGB."""
    frame = np.asarray(wrist_rgb, dtype=np.uint8).copy()
    if frame.shape[:2] != heatmap.shape:
        raise ValueError("wrist frame and tactile heatmap dimensions differ")
    alpha = (heatmap.astype(np.float32) / 255.0 * .8)[..., None]
    red = np.zeros_like(frame)
    red[..., 0] = 255
    red[..., 1] = 30
    frame = np.clip(frame * (1 - alpha) + red * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)

