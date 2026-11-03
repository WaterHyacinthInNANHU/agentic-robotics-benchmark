"""Initial physically acting reference candidate; success must be measured."""
import json
import sys


def clip(value):
    return max(-1.0, min(1.0, value))


def action(t, obs):
    eef = obs["eef_pos"]
    nut = obs["nut_pos"]
    handle = obs["handle_pos"]
    peg = obs["peg_pos"]
    grip = -1.0 if t < 65 or 165 <= t < 180 else 1.0
    if t < 38:
        target = [handle[0], handle[1], handle[2] + 0.07]
    elif t < 65:
        target = [handle[0], handle[1], handle[2] + 0.008]
    elif t < 82:
        target = [eef[0], eef[1], eef[2]]
    elif t < 106:
        target = [eef[0], eef[1], max(eef[2], nut[2] + 0.16)]
    elif t < 145:
        target = [eef[0] + peg[0] - nut[0],
                  eef[1] + peg[1] - nut[1],
                  max(eef[2], nut[2] + 0.13)]
    elif t < 165:
        target = [eef[0] + peg[0] - nut[0],
                  eef[1] + peg[1] - nut[1],
                  eef[2] + 0.85 - nut[2]]
    elif t < 180:
        target = list(eef)
    else:
        target = [eef[0], eef[1], max(eef[2], 1.04)]
    return [clip(9 * (target[i] - eef[i])) for i in range(3)] + [0.0, 0.0, 0.0, grip]


def main():
    print("READY", flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        if message["type"] == "act":
            reply = {"action": action(message["t"], message["obs"])}
        else:
            reply = {"ok": True}
        print(json.dumps(reply), flush=True)
        if message["type"] == "close":
            break


if __name__ == "__main__":
    main()
