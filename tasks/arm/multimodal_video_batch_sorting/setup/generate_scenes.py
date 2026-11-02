#!/usr/bin/env python3
"""Deterministically render agent-visible RGB-D commissioning scenes.

This file runs only in the trusted setup stage.  The agent receives the rendered frames,
calibration, work orders and inspection logs, never this generator or its scene parameters.
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 384, 288
SCENE_COUNT = 16
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
KINDS = ("gear", "plate", "coupler")
LABELS = ("AX", "BK", "CP", "DR", "EV", "FQ", "GT", "HU", "JM", "LN", "PS", "WY", "ZR", "QT", "NX", "SV", "KD")
TOOLS = {"gear": "parallel_jaw", "plate": "vacuum", "coupler": "three_finger"}
GRASPS = {"gear": "rim", "plate": "top", "coupler": "body"}
COLORS = {"gear": (190, 139, 42), "plate": (45, 112, 162), "coupler": (146, 58, 67)}
PART_SIZE = {"gear": (0.046, 0.030), "plate": (0.052, 0.026), "coupler": (0.050, 0.034)}
GRID = ((-0.172, 0.105), (-0.080, 0.105), (0.010, 0.105),
        (-0.172, -0.090), (-0.080, -0.090), (0.010, -0.090))
CAMERAS = (
    {"theta_deg": -2.8, "scale": 620.0, "cx": 180.0, "cy": 145.0, "zx": -105.0, "zy": 34.0},
    {"theta_deg": 2.2, "scale": 628.0, "cx": 184.0, "cy": 143.0, "zx": 85.0, "zy": 24.0},
    {"theta_deg": 0.4, "scale": 616.0, "cx": 181.0, "cy": 147.0, "zx": -20.0, "zy": -38.0},
)


def scene_spec(index):
    rng = random.Random(240519 + index * 7919)
    labels = rng.sample(LABELS, 6)
    kinds = [KINDS[(index + j) % 3] for j in range(6)]
    rng.shuffle(kinds)
    parts = []
    for j, ((gx, gy), label, kind) in enumerate(zip(GRID, labels, kinds)):
        width, height = PART_SIZE[kind]
        parts.append({
            "label": label,
            "kind": kind,
            "position_m": [round(gx + rng.uniform(-0.010, 0.010), 5),
                           round(gy + rng.uniform(-0.010, 0.010), 5), height],
            "yaw_deg": float(rng.randrange(0, 360, 15)),
            "radius_m": width / 2,
            "height_m": height,
        })
    relation = "above" if index % 2 == 0 else "below"
    force_threshold = round(rng.uniform(3.6, 4.6), 2)
    vibration_threshold = round(rng.uniform(0.43, 0.64), 2)
    distractor = rng.randrange(6)
    rows, targets = [], []
    for j, part in enumerate(parts):
        selected = j != distractor
        if relation == "above":
            force = rng.uniform(force_threshold + 0.25, 6.1) if selected else rng.uniform(2.1, force_threshold - 0.25)
        else:
            force = rng.uniform(2.1, force_threshold - 0.25) if selected else rng.uniform(force_threshold + 0.25, 6.1)
        vibration = round(rng.uniform(0.18, 0.88), 3)
        rows.append({"stamp": part["label"], "peak_force_n": round(force, 2),
                     "mass_g": round(rng.uniform(44, 98), 1), "vibration_rms": vibration})
        if selected:
            targets.append({
                "label": part["label"],
                "destination": "reject" if vibration >= vibration_threshold else "ready",
            })

    # A tall central machine fence forces an upper/lower route.  A second block closes one
    # lane, so a robust path must use the other lane before crossing to the destination bin.
    blocked_lane = "upper" if index % 2 == 0 else "lower"
    block_y = 0.118 if blocked_lane == "upper" else -0.118
    obstacles = [
        {"id": "fence", "min_m": [0.070, -0.054, 0.0], "max_m": [0.128, 0.054, 0.235]},
        {"id": "lane_block", "min_m": [0.142, block_y - 0.034, 0.0],
         "max_m": [0.184, block_y + 0.034, 0.205]},
    ]

    # One or two selected parts sit below a labelled sliding service guard.  Moving any
    # other guard is an unsafe/unneeded action.
    guarded = rng.sample([p for p in parts if p["label"] != labels[distractor]], 2)
    guards = []
    for gi, part in enumerate(guarded):
        x, y, _ = part["position_m"]
        direction = 1.0 if x < -0.04 else -1.0
        guards.append({
            "id": "G{}".format(gi + 1),
            "min_m": [round(x - 0.036, 5), round(y - 0.020, 5), 0.074],
            "max_m": [round(x + 0.036, 5), round(y + 0.020, 5), 0.132],
            "axis": [direction, 0.0, 0.0],
            "distance_m": 0.115,
            "handle_position_m": [round(x - direction * 0.025, 5), round(y, 5), 0.115],
            "covers": part["label"],
        })

    return {
        "scene_id": "scene_{:02d}".format(index),
        "parts": parts,
        "rows": rows,
        "targets": targets,
        "relation": relation,
        "force_threshold": force_threshold,
        "vibration_threshold": vibration_threshold,
        "obstacles": obstacles,
        "guards": guards,
        "blocked_lane": blocked_lane,
        "bins": {"ready": [0.255, 0.112, 0.040], "reject": [0.255, -0.112, 0.040]},
    }


def project(point, camera):
    x, y, z = point
    angle = math.radians(camera["theta_deg"])
    ca, sa = math.cos(angle), math.sin(angle)
    u = camera["cx"] + camera["scale"] * (ca * x - sa * y) + camera["zx"] * z
    v = camera["cy"] - camera["scale"] * (sa * x + ca * y) + camera["zy"] * z
    return u, v


def projection_matrix(camera):
    angle = math.radians(camera["theta_deg"])
    ca, sa = math.cos(angle), math.sin(angle)
    return [
        [camera["scale"] * ca, -camera["scale"] * sa, camera["zx"], camera["cx"]],
        [-camera["scale"] * sa, -camera["scale"] * ca, camera["zy"], camera["cy"]],
    ]


def polygon_for_part(part, camera):
    x, y, z = part["position_m"]
    radius = part["radius_m"]
    if part["kind"] == "gear":
        n = 20
        local = [(radius * (1.0 if i % 2 == 0 else 0.84) * math.cos(2 * math.pi * i / n),
                  radius * (1.0 if i % 2 == 0 else 0.84) * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elif part["kind"] == "plate":
        local = [(-radius, -radius * .65), (radius, -radius * .65),
                 (radius, radius * .65), (-radius, radius * .65)]
    else:
        local = [(-radius, -radius * .62), (radius * .62, -radius * .62),
                 (radius, 0), (radius * .62, radius * .62), (-radius, radius * .62)]
    yaw = math.radians(part["yaw_deg"])
    ca, sa = math.cos(yaw), math.sin(yaw)
    return [project((x + ca * a - sa * b, y + sa * a + ca * b, z), camera) for a, b in local]


def draw_arrow(draw, start, direction, colour, width=3):
    x, y = start
    dx, dy = direction
    end = (x + dx, y + dy)
    draw.line((x, y, end[0], end[1]), fill=colour, width=width)
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    draw.polygon([end, (end[0] - 9 * ux + 5 * px, end[1] - 9 * uy + 5 * py),
                  (end[0] - 9 * ux - 5 * px, end[1] - 9 * uy - 5 * py)], fill=colour)


def render(scene, output):
    font_small = ImageFont.truetype(FONT, 13)
    font_tiny = ImageFont.truetype(FONT, 10)
    rng = random.Random(718281 + int(scene["scene_id"].split("_")[1]) * 104729)
    calibration = {"image_size_px": [W, H], "depth_encoding": "height_mm_uint16",
                   "invalid_depth_value": 0, "views": []}
    for view_index, camera in enumerate(CAMERAS):
        # Brushed-metal tray background with illumination gradient and repeatable sensor grain.
        image = Image.new("RGB", (W, H))
        pixels = image.load()
        for yy in range(H):
            base = 92 + int(18 * yy / H)
            for xx in range(W):
                grain = rng.randrange(-7, 8)
                stripe = 4 if (xx // 17) % 2 == 0 else 0
                pixels[xx, yy] = (max(0, base + grain + stripe), max(0, base + grain + stripe + 4), max(0, base + grain + stripe + 7))
        image = image.filter(ImageFilter.GaussianBlur(radius=.35))
        draw = ImageDraw.Draw(image)
        depth = Image.new("I;16", (W, H), 1)
        dd = ImageDraw.Draw(depth)

        # Tray boundary and two destination bins.
        tray = [project((-0.225, -0.165, 0), camera), project((0.310, -0.165, 0), camera),
                project((0.310, 0.165, 0), camera), project((-0.225, 0.165, 0), camera)]
        draw.line(tray + [tray[0]], fill=(210, 215, 217), width=3)
        for name, centre in scene["bins"].items():
            u, v = project((centre[0], centre[1], 0), camera)
            colour = (54, 116, 73) if name == "ready" else (135, 48, 52)
            draw.rounded_rectangle((u - 31, v - 24, u + 31, v + 24), radius=7, fill=colour, outline=(235, 235, 226), width=2)
            draw.text((u - 25, v - 7), name.upper(), font=font_tiny, fill=(245, 245, 238))
            dd.rounded_rectangle((u - 31, v - 24, u + 31, v + 24), radius=7, fill=12)

        # Tall static machine geometry.  Top surfaces are projected with their real height,
        # producing view-dependent parallax relative to the tray calibration.
        for obstacle in scene["obstacles"]:
            lo, hi = obstacle["min_m"], obstacle["max_m"]
            top = [project((lo[0], lo[1], hi[2]), camera), project((hi[0], lo[1], hi[2]), camera),
                   project((hi[0], hi[1], hi[2]), camera), project((lo[0], hi[1], hi[2]), camera)]
            shadow = [(u + 7, v + 9) for u, v in top]
            draw.polygon(shadow, fill=(35, 35, 38))
            draw.polygon(top, fill=(62, 66, 70), outline=(225, 180, 38))
            # Hazard stripes make obstacle boundaries recoverable under texture and shadow.
            for k in range(0, 36, 8):
                draw.line((top[0][0] + k, top[0][1], top[0][0] + k + 13, top[0][1] + 13), fill=(225, 176, 35), width=3)
            dd.polygon(top, fill=int(round(hi[2] * 1000)))

        # Parts first, guards second: elevated service guards naturally occlude them.
        for part in scene["parts"]:
            poly = polygon_for_part(part, camera)
            shadow = [(u + 4, v + 5) for u, v in poly]
            draw.polygon(shadow, fill=(28, 29, 30))
            base = COLORS[part["kind"]]
            draw.polygon(poly, fill=base, outline=(232, 232, 221))
            u, v = project(part["position_m"], camera)
            r = max(4, int(part["radius_m"] * camera["scale"] * .23))
            draw.ellipse((u - r, v - r, u + r, v + r), fill=(30, 33, 35), outline=(235, 235, 228))
            bbox = draw.textbbox((0, 0), part["label"], font=font_small)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.rectangle((u - tw / 2 - 3, v - th / 2 - 2, u + tw / 2 + 3, v + th / 2 + 2), fill=(220, 218, 201))
            draw.text((u - tw / 2, v - th / 2 - 2), part["label"], font=font_small, fill=(25, 25, 25))
            yaw = math.radians(part["yaw_deg"])
            draw_arrow(draw, (u, v), (18 * math.cos(yaw), -18 * math.sin(yaw)), (248, 244, 226), 2)
            dd.polygon(poly, fill=int(round(part["height_m"] * 1000)))

        for guard in scene["guards"]:
            lo, hi = guard["min_m"], guard["max_m"]
            top = [project((lo[0], lo[1], hi[2]), camera), project((hi[0], lo[1], hi[2]), camera),
                   project((hi[0], hi[1], hi[2]), camera), project((lo[0], hi[1], hi[2]), camera)]
            draw.polygon(top, fill=(44, 48, 54), outline=(230, 173, 38))
            hu, hv = project(guard["handle_position_m"], camera)
            draw.ellipse((hu - 6, hv - 6, hu + 6, hv + 6), fill=(224, 174, 37), outline=(245, 240, 220))
            draw.text((hu + 7, hv - 9), guard["id"], font=font_tiny, fill=(255, 244, 196))
            ax = guard["axis"]
            origin = project((guard["handle_position_m"][0], guard["handle_position_m"][1], guard["handle_position_m"][2]), camera)
            end = project((guard["handle_position_m"][0] + .055 * ax[0], guard["handle_position_m"][1] + .055 * ax[1], guard["handle_position_m"][2]), camera)
            draw_arrow(draw, origin, (end[0] - origin[0], end[1] - origin[1]), (255, 205, 55), 3)
            dd.polygon(top, fill=int(round(hi[2] * 1000)))

        # Sparse invalid depth samples and millimetre noise emulate an inexpensive RGB-D unit.
        dp = depth.load()
        for _ in range(1450):
            xx, yy = rng.randrange(W), rng.randrange(H)
            if rng.random() < .38:
                dp[xx, yy] = 0
            elif dp[xx, yy] > 1:
                dp[xx, yy] = max(1, int(dp[xx, yy]) + rng.randrange(-2, 3))
        image.save(output / "rgb_{}.png".format(view_index), optimize=True)
        depth.save(output / "depth_{}.png".format(view_index), optimize=True)
        calibration["views"].append({"rgb": "rgb_{}.png".format(view_index),
                                     "depth": "depth_{}.png".format(view_index),
                                     "projection_uv_from_xyz_m": projection_matrix(camera)})
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")


def write_public(scene, root):
    folder = root / scene["scene_id"]
    folder.mkdir(parents=True, exist_ok=True)
    text = ("Process every stamped unit whose peak_force_n is {relation} {force:.2f} N. "
            "For each selected unit, vibration_rms at least {vibration:.2f} goes to REJECT; "
            "otherwise READY. Safely retract only guards that block selected picks, then "
            "execute collision-free Cartesian picks and placements for the complete batch.")
    (folder / "work_order.txt").write_text(text.format(relation=scene["relation"], force=scene["force_threshold"], vibration=scene["vibration_threshold"]) + "\n")
    with (folder / "inspection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stamp", "peak_force_n", "mass_g", "vibration_rms"))
        writer.writeheader()
        writer.writerows(scene["rows"])
    public = {
        "scene_id": scene["scene_id"],
        "frames": ["rgb_0.png", "rgb_1.png", "rgb_2.png"],
        "depth_frames": ["depth_0.png", "depth_1.png", "depth_2.png"],
        "bins": ["ready", "reject"],
    }
    (folder / "manifest.json").write_text(json.dumps(public, indent=2) + "\n")
    render(scene, folder)

    task_spec = {
        "workspace_bounds_m": [[-0.225, -0.165, 0.0], [0.310, 0.165, 0.260]],
        "tool_radius_m": {"parallel_jaw": 0.022, "vacuum": 0.018, "three_finger": 0.024},
        "part_tool": TOOLS,
        "part_grasp": GRASPS,
        "position_tolerance_m": 0.007,
        "yaw_tolerance_deg": 8.0,
        "guard_handle_tolerance_m": 0.012,
        "maximum_waypoints_per_pick": 8,
        "maximum_segment_length_m": 0.26,
        "required_approach_clearance_m": [0.065, 0.125],
        "robustness_note": "Plans are replayed under hidden millimetre-scale pose and clearance perturbations; use margin.",
    }
    (folder / "task_spec.json").write_text(json.dumps(task_spec, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(SCENE_COUNT):
        write_public(scene_spec(index), args.output)


if __name__ == "__main__":
    main()
