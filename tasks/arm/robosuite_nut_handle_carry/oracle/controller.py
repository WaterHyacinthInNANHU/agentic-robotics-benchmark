"""Public-observation reference for native robosuite nut handle carry."""
import json
import sys

def clip(value):
    return max(-1.0, min(1.0, value))


def move(eef, target, gripper, limit=1.0):
    return [max(-limit, min(limit, 12 * (target[i] - eef[i])))
            for i in range(3)] + [0, 0, 0, gripper]


class Policy:
    def __init__(self):
        self.carry = False
        self.grasp_step = None

    def act(self, t, obs):
        eef = obs["eef_pos"]
        nut = obs["nut_pos"]
        handle = obs["handle_pos"]
        peg = obs["peg_pos"]
        if self.carry and nut[2] < 0.92:
            self.carry = False
        if nut[2] > 0.95 and t % 100 >= 95:
            if not self.carry:
                self.grasp_step = t
            self.carry = True
        if self.carry:
            target = [eef[0] + peg[0] - nut[0],
                      eef[1] + peg[1] - nut[1], 1.15]
            return move(eef, target, 1, limit=0.15)
        attempt = min(t // 100, 2)
        local = t - attempt * 100
        rotate_sign = -1 if attempt == 0 else 1
        if local < 6:
            return [0, 0, 0, 0, 0, rotate_sign, -1]
        if local < 36:
            return move(eef, [handle[0], handle[1], handle[2] + 0.075], -1)
        if local < 65:
            return move(eef, [handle[0], handle[1], handle[2] - 0.005], -1)
        if local < 80:
            return [0, 0, 0, 0, 0, 0, 1]
        return move(eef, [eef[0], eef[1], 1.08], 1)

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
            reply = {"action": policy.act(int(message["t"]), message["obs"])}
        else:
            reply = {"ok": True}
        print(json.dumps(reply), flush=True)
        if kind == "close":
            break


if __name__ == "__main__":
    main()
