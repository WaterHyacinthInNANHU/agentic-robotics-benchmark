"""Probe a late physical cube impulse against frozen observation-only policies."""
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_retarget/oracle"))
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_retarget/verify"))
from controller import Policy  # noqa: E402
from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402


class AgentPolicy:
    def __init__(self, directory):
        self.proc = subprocess.Popen(
            ["bash", "run.sh"], cwd=directory,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            start_new_session=True,
        )
        if self.proc.stdout.readline().strip() != "READY":
            raise RuntimeError("agent did not become ready")
        self.request({"type": "reset"})

    def request(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def act(self, t, obs):
        return self.request({"type": "act", "t": t, "obs": obs})["action"]

    def close(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait(timeout=5)


def trial(seed, force, kind):
    world = PushEpisode(seed, render=False)
    policy = Policy() if kind == "reference" else AgentPolicy(
        ROOT / "agent_trials/fetch_precision_retarget_blind_02/submission"
    )
    body_id = mujoco.mj_name2id(world.env.unwrapped.model,
                               mujoco.mjtObj.mjOBJ_BODY, "object0")
    if body_id < 0:
        raise RuntimeError("Fetch cube body object0 missing")
    theta = seed * 2.399963229728653
    impulse = [force * math.cos(theta), force * math.sin(theta), 0.0]
    post_impulse_error = None
    try:
        obs = world.observe()
        for t in range(MAX_STEPS):
            if t == 360:
                world.env.unwrapped.data.xfrc_applied[body_id, :3] = impulse
            action = policy.act(obs) if kind == "reference" else policy.act(t, obs)
            obs = world.step(action)
            if t == 360:
                world.env.unwrapped.data.xfrc_applied[body_id, :] = 0
                post_impulse_error = math.dist(
                    world.raw["achieved_goal"], world.goal)
        row = world.result()
        row["post_impulse_error_m"] = post_impulse_error
        return row
    finally:
        world.close()
        if kind != "reference":
            policy.close()


def main():
    for force in (1.0, 3.0, 6.0, 10.0, 20.0):
        for kind in ("reference", "agent02"):
            rows = [trial(seed, force, kind) for seed in range(9000, 9020)]
            print("IMPULSE_PROBE", json.dumps({
                "force_n": force, "kind": kind, "count": len(rows),
                "wins": sum(row["success"] for row in rows),
                "stage1": sum(row["stage1_success"] for row in rows),
                "stage2": sum(row["stage2_success"] for row in rows),
                "mean_post_impulse_error_m": sum(row["post_impulse_error_m"]
                                                 for row in rows) / len(rows),
                "mean_final_stage2_error_m": sum(row["stage2_error_m"]
                                                 for row in rows) / len(rows),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
