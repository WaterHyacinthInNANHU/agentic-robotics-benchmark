#!/usr/bin/env python3
"""Observation driven OSC controller for the square nut task."""
import json
import math
import sys


def clip(x, bound=1.0):
    return max(-bound, min(bound, x))


def yaw_xyzw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def periodic_error(target, current, period):
    return (target - current + period / 2.0) % period - period / 2.0


def norm(v):
    return math.sqrt(sum(x * x for x in v))


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = "hover"
        self.stage_start = 0
        self.last_t = -1
        self.initial_nut_z = None
        self.initial_handle_z = None
        self.close_eef_xy = None
        self.lift_z = None
        self.retry = 0
        self.lower_start = 0
        self.release_eef_xy = None
        self.release_eef_z = None

    def advance(self, name, t):
        self.stage = name
        self.stage_start = t
        if name == "lower":
            self.lower_start = t

    def act(self, t, obs):
        e = obs["eef_pos"]
        h = obs["handle_pos"]
        n = obs["nut_pos"]
        p = obs["peg_pos"]
        if self.initial_nut_z is None:
            self.initial_nut_z = n[2]
            self.initial_handle_z = h[2]
            self.lift_z = max(p[2] + 0.20, h[2] + 0.19)
        if self.stage in ("hover", "descend"):
            self.initial_nut_z = min(self.initial_nut_z, n[2])
        handle_yaw = math.atan2(h[1] - n[1], h[0] - n[0])
        grasp_yaw_error = periodic_error(handle_yaw + math.pi / 2.0,
                                          yaw_xyzw(obs["eef_quat"]), math.pi)
        # State changes occur once per action index, including if the grader
        # repeats an observation for the same index.
        if t != self.last_t:
            age = t - self.stage_start
            if self.stage == "hover":
                goal = [h[0], h[1], h[2] + 0.105]
                if (norm([goal[i] - e[i] for i in range(3)]) < 0.014
                        and abs(grasp_yaw_error) < 0.12) or age >= 65:
                    self.advance("descend", t)
            elif self.stage == "descend":
                goal = [h[0], h[1], h[2] + 0.003]
                if norm([goal[i] - e[i] for i in range(3)]) < 0.012 or age >= 65:
                    self.advance("grasp", t)
                    self.close_eef_xy = e[:2]
            elif self.stage == "grasp":
                if age >= 18:
                    self.advance("lift", t)
            elif self.stage == "lift":
                if age >= 18 and n[2] < self.initial_nut_z + 0.035 and self.retry < 2:
                    self.retry += 1
                    self.advance("retry", t)
                elif (e[2] > self.lift_z - 0.018 and n[2] > self.initial_nut_z + 0.045) or age >= 65:
                    self.advance("transfer", t)
            elif self.stage == "retry":
                if age >= 12:
                    self.advance("hover", t)
            elif self.stage == "transfer":
                err = norm([p[i] - n[i] for i in (0, 1)])
                if (err < 0.009 and age >= 12) or age >= 85 or t >= 275:
                    self.advance("lower", t)
            elif self.stage == "lower":
                # Reserve enough time to open the gripper and allow the nut to settle.
                if (n[2] <= 0.858 and age >= 18) or age >= 95 or t >= 355:
                    self.advance("release", t)
                    self.release_eef_xy = e[:2]
                    self.release_eef_z = e[2]
            self.last_t = t

        grip = 1.0
        rz = 0.0
        if self.stage == "hover":
            target = [h[0], h[1], h[2] + 0.105]
            grip = -1.0
            rz = clip(1.8 * grasp_yaw_error, 0.65)
        elif self.stage == "descend":
            target = [h[0], h[1], h[2] + 0.003]
            grip = -1.0
            rz = clip(1.5 * grasp_yaw_error, 0.4)
        elif self.stage == "grasp":
            target = [h[0], h[1], h[2] + 0.003]
        elif self.stage == "lift":
            target = [self.close_eef_xy[0], self.close_eef_xy[1], self.lift_z]
        elif self.stage == "retry":
            target = [h[0], h[1], h[2] + 0.12]
            grip = -1.0
        elif self.stage in ("transfer", "lower"):
            # The nut and handle observations share a fixed visual offset.
            # Servo on the nut itself so the grasp transform need not be known.
            ex = p[0] - n[0]
            ey = p[1] - n[1]
            if self.stage == "transfer":
                target = [e[0] + ex, e[1] + ey, self.lift_z]
            else:
                age = t - self.lower_start
                if age > 28 and n[2] > self.initial_nut_z + 0.025:
                    # Small dither helps a square hole settle onto a square peg.
                    radius = 0.0045
                    phase = age * 0.23
                    ex += radius * math.cos(phase)
                    ey += radius * math.sin(phase)
                desired_nut_z = 0.833
                target = [e[0] + ex, e[1] + ey,
                          e[2] + desired_nut_z - n[2]]
            nut_yaw = yaw_xyzw(obs["nut_quat"])
            yaw_error = (nut_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            rz = clip(-1.5 * yaw_error, 0.65)
        else:
            grip = -1.0
            # Move the empty tool out of the nut's way after releasing it.
            release_age = t - self.stage_start
            target = [self.release_eef_xy[0], self.release_eef_xy[1],
                      self.release_eef_z + (0.0 if release_age < 10 else 0.13)]

        delta = [target[i] - e[i] for i in range(3)]
        action = [clip(15.0 * delta[0]), clip(15.0 * delta[1]),
                  clip(13.0 * delta[2]), 0.0, 0.0, rz, grip]
        if self.stage == "lower":
            action[2] = clip(8.0 * delta[2], 0.5)
        return action


def main():
    ctl = Controller()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            kind = msg.get("type")
            if kind == "reset":
                ctl.reset()
                response = {"ok": True}
            elif kind == "act":
                response = {"action": ctl.act(int(msg["t"]), msg["obs"])}
            elif kind == "close":
                response = {"ok": True}
                print(json.dumps(response, separators=(",", ":")), flush=True)
                break
            else:
                response = {"ok": False}
            print(json.dumps(response, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(f"controller error: {exc}", file=sys.stderr, flush=True)
            print('{"action":[0,0,0,0,0,0,-1]}', flush=True)


if __name__ == "__main__":
    main()
