"""Visible-seed practice runner; prints only the raw physical outcomes."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fetch_push_env import MAX_STEPS, PushEpisode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=os.environ.get("SUBMISSION_DIR", "/home/user/submission"))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    directory = Path(args.submission)
    if not (directory / "run.sh").is_file():
        parser.error("missing submission/run.sh")
    proc = subprocess.Popen(["bash", "run.sh"], cwd=directory, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=1)
    lines = queue.Queue()

    def pump():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    def read(timeout=10):
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("controller reply timeout")
        if line is None:
            raise RuntimeError("controller exited")
        return line.strip()

    def request(payload):
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        reply = json.loads(read())
        if not isinstance(reply, dict):
            raise ValueError("reply must be an object")
        return reply

    rows = []
    try:
        if read(40) != "READY":
            raise RuntimeError("controller did not print READY")
        for seed in (11, 22, 33, 44, 55)[:max(1, min(5, args.episodes))]:
            world = PushEpisode(seed)
            try:
                request({"type": "reset"})
                for t in range(MAX_STEPS):
                    reply = request({"type": "act", "t": t, "obs": world.observe()})
                    world.step(reply["action"])
                rows.append(dict(seed=seed, **world.result()))
            finally:
                world.close()
        request({"type": "close"})
    finally:
        if proc.poll() is None:
            proc.kill()
    print(json.dumps({"success_rate": sum(row["success"] for row in rows) / len(rows),
                      "episodes": rows, "note": "visible practice seeds only"}))


if __name__ == "__main__":
    main()
