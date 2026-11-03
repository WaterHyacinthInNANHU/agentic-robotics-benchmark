"""Trace failure geometry for the FetchPush reference controller."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/setup/payload"))
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_push/oracle"))

from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402
from controller import Policy  # noqa: E402

for seed in (1019, 1024, 1027, 1034, 1036, 2015, 2018, 2034):
    world = PushEpisode(seed, render=False)
    policy = Policy()
    rows = []
    for t in range(MAX_STEPS):
        obs = world.observe()
        state = obs["state"]
        action = policy.act(obs)
        if t % 5 == 0 or t == MAX_STEPS - 1:
            rows.append({
                "t": t, "phase": policy.phase,
                "grip": [round(v, 3) for v in state[:3]],
                "cube": [round(v, 3) for v in state[3:6]],
                "goal": [round(v, 3) for v in obs["goal"]],
                "action": [round(v, 2) for v in action[:3]],
            })
        world.step(action)
    print(json.dumps({"seed": seed, "result": world.result(), "trace": rows}), flush=True)
    world.close()
