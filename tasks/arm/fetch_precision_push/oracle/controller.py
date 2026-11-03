"""Reference contact controller using only the public observation."""
import json
import math
import sys

OFFSET = 0.065


def scaled(delta):
    return max(-1.0, min(1.0, delta / 0.05))


class Policy:
    def __init__(self):
        self.phase = 0

    def act(self, obs):
        state = obs["state"]
        gripper = state[:3]
        cube = state[3:6]
        goal = obs["goal"]
        dx, dy = goal[0] - cube[0], goal[1] - cube[1]
        gap = math.hypot(dx, dy)
        ux, uy = dx / max(gap, 1e-9), dy / max(gap, 1e-9)
        if self.phase == 2 and math.dist(gripper[:2], cube[:2]) > 0.10:
            self.phase = 0
        behind = [cube[0] - OFFSET * ux, cube[1] - OFFSET * uy, cube[2]]
        if self.phase == 0:
            target = [behind[0], behind[1], behind[2] + 0.10]
            if math.dist(gripper, target) < 0.020:
                self.phase = 1
        elif self.phase == 1:
            target = [behind[0], behind[1], behind[2] + 0.005]
            if math.dist(gripper, target) < 0.015:
                self.phase = 2
        else:
            distance = min(0.05, gap * 0.45) if gap > 0.015 else 0.0
            target = [gripper[0] + ux * distance,
                      gripper[1] + uy * distance,
                      cube[2] + 0.005]
        return [scaled(target[i] - gripper[i]) for i in range(3)] + [0.0]


def main():
    policy = Policy()
    print("READY", flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        kind = message.get("type")
        if kind == "reset":
            policy = Policy()
            reply = {"ok": True}
        elif kind == "act":
            reply = {"action": policy.act(message["obs"])}
        else:
            reply = {"ok": True}
        print(json.dumps(reply), flush=True)
        if kind == "close":
            break


if __name__ == "__main__":
    main()
