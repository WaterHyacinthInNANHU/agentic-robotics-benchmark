"""Independent solver trial using the task's exact public observation and physics."""
import argparse
import json
import os
import queue
import secrets
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tasks/arm/fetch_precision_retarget/verify"))
from fetch_push_env import MAX_STEPS, PushEpisode  # noqa: E402


def episode(submission, seed):
    proc = subprocess.Popen(
        ["bash", "run.sh"], cwd=submission, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        bufsize=1, start_new_session=True,
    )
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def read(timeout):
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("solver reply timeout")
        if line is None:
            raise RuntimeError("solver closed stdout")
        return line.strip()

    def request(payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        reply = json.loads(read(10))
        if not isinstance(reply, dict):
            raise ValueError("solver reply is not an object")
        return reply

    try:
        if read(40) != "READY":
            raise ValueError("missing READY")
        world = PushEpisode(seed)
        try:
            request({"type": "reset"})
            for t in range(MAX_STEPS):
                obs = world.observe()
                action = request({"type": "act", "t": t, "obs": obs})["action"]
                world.step(action)
            return world.result()
        finally:
            world.close()
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()
    if not (args.submission / "run.sh").is_file():
        parser.error("missing run.sh")
    rows = []
    for i in range(args.episodes):
        seed = secrets.randbits(64)
        try:
            result = episode(args.submission, seed)
        except Exception as exc:
            result = {"success": False, "stage1_success": False,
                      "stage2_success": False, "error": repr(exc)}
        rows.append(result)
        print("episode", i, json.dumps(result, sort_keys=True), flush=True)
    summary = {
        "episodes": len(rows),
        "complete_successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "stage1_success_rate": sum(bool(row["stage1_success"]) for row in rows) / len(rows),
        "stage2_success_rate": sum(bool(row["stage2_success"]) for row in rows) / len(rows),
        "aborted": sum("error" in row for row in rows),
    }
    print("BLIND_AGENT_RESULT", args.submission, json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
