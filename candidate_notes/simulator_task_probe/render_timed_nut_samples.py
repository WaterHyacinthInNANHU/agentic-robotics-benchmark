"""Render two genuine native simulator trajectories as annotated contact sheets."""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/robosuite_square_nut_assembly/verify"))
from square_nut_env import SquareNutEpisode  # noqa: E402
from nut_fast_insert_controller import Policy  # noqa: E402


def make_frame(obs, t, success):
    raw = base64.b64decode(obs["rgb_png_base64"])
    image = Image.open(io.BytesIO(raw)).convert("RGB").resize((256, 256))
    canvas = Image.new("RGB", (256, 304), "#20242b")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    nut, peg = obs["nut_pos"], obs["peg_pos"]
    xy = ((nut[0] - peg[0]) ** 2 + (nut[1] - peg[1]) ** 2) ** .5
    draw.text((8, 264), f"t={t}/170  XY={xy * 100:.1f} cm", fill="white")
    draw.text((8, 282), f"nut z={nut[2]:.3f} m  success={int(success)}", fill="white")
    return canvas


def run(seed, label, outdir):
    world = SquareNutEpisode(seed, render=True)
    policy = Policy()
    frames = []
    obs = world.observe()
    frames.append(make_frame(obs, 0, False))
    try:
        for t in range(170):
            obs, done = world.step(policy.step(t, obs))
            if world.t in (40, 80, 120):
                frames.append(make_frame(obs, world.t, world.success))
            if done:
                break
        frames.append(make_frame(obs, world.t, world.success))
        while len(frames) < 5:
            frames.insert(-1, frames[-2].copy())
        sheet = Image.new("RGB", (256 * len(frames), 332), "#20242b")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 7),
                  f"Native robosuite NutAssemblySquare: {label.upper()} seed {seed}",
                  fill="#ffffff")
        for i, frame in enumerate(frames):
            sheet.paste(frame, (256 * i, 28))
        path = outdir / f"{label}_seed_{seed}.jpg"
        sheet.save(path, "JPEG", quality=82, optimize=True)
        print("NUT_VIS_RESULT", json.dumps({"label": label, "seed": seed,
              "success": world.success, "steps": world.t, "path": str(path)}), flush=True)
        print("NUT_VIS_JPEG", label, base64.b64encode(path.read_bytes()).decode("ascii"), flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    outdir = Path("/tmp/timed-nut-visuals")
    outdir.mkdir(exist_ok=True)
    run(201, "success", outdir)
    run(200, "failure", outdir)
