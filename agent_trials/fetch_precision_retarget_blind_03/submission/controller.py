#!/usr/bin/env python3
"""Closed-loop Cartesian controller for the two-goal Fetch push task.

Uses only the public kinematic observation.  All distances are in metres.
"""
import json
import math
import sys


def norm2(x, y):
    return math.hypot(x, y)


def limit(x, bound):
    return max(-bound, min(bound, x))


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = None
        self.phase = "approach"
        self.last_t = None
        self.last_action = None
        self.last_object = None
        self.contact_gap = 0.060
        self.push_stall = 0
        self.z_index = 0
        self.phase_start = 0
        self.settled = 0

    def move(self, grip, target, gain=1.0, xy_cap=1.0, z_cap=1.0):
        ax = limit(gain * (target[0] - grip[0]) / 0.05, xy_cap)
        ay = limit(gain * (target[1] - grip[1]) / 0.05, xy_cap)
        az = limit(gain * (target[2] - grip[2]) / 0.05, z_cap)
        return [ax, ay, az, 0.0]

    def act(self, t, obs):
        if t == self.last_t:
            return self.last_action

        stage = int(obs["stage"])
        stage_step = int(obs["stage_step"])
        s = obs["state"]
        grip = [float(s[i]) for i in range(3)]
        cube = [float(s[i]) for i in range(3, 6)]
        goal = [float(x) for x in obs["goal"]]
        if self.stage != stage:
            self.stage = stage
            self.phase = "approach"
            self.phase_start = stage_step
            self.last_object = None
            self.contact_gap = 0.060
            self.push_stall = 0
            self.z_index = 0
            self.settled = 0

        ex, ey = goal[0] - cube[0], goal[1] - cube[1]
        distance = norm2(ex, ey)
        if distance < 1e-8:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = ex / distance, ey / distance
        vx = vy = 0.0
        if self.last_object is not None:
            vx = cube[0] - self.last_object[0]
            vy = cube[1] - self.last_object[1]
        speed = norm2(vx, vy)
        toward = vx * ux + vy * uy
        rx, ry = cube[0] - grip[0], cube[1] - grip[1]
        along = rx * ux + ry * uy
        across = abs(rx * uy - ry * ux)

        # A small success margin leaves room for final contact relaxation.
        if distance < 0.0023 and speed < 0.00065:
            self.settled += 1
        else:
            self.settled = 0
        if self.settled >= 2 and self.phase not in ("park", "brake"):
            self.phase = "park"
            self.phase_start = stage_step

        if self.phase == "park":
            # Lift vertically, then hold position until a new target appears.
            target = [grip[0], grip[1], max(grip[2], cube[2] + 0.095)]
            action = self.move(grip, target, gain=1.0, xy_cap=0.0)
            if distance > 0.0031 and stage_step - self.phase_start > 3:
                self.phase = "approach"
                self.phase_start = stage_step

        elif self.phase == "brake":
            # Breaking contact is more reliable than waiting under force.
            target = [grip[0] - 0.025 * ux, grip[1] - 0.025 * uy,
                      max(grip[2], cube[2] + 0.065)]
            action = self.move(grip, target, gain=0.9, xy_cap=0.35)
            if stage_step - self.phase_start >= 5:
                self.phase = "approach"
                self.phase_start = stage_step

        elif self.phase == "approach":
            target = [cube[0] - 0.085 * ux, cube[1] - 0.085 * uy,
                      cube[2] + 0.095]
            action = self.move(grip, target, gain=1.3, xy_cap=0.85)
            if norm2(target[0] - grip[0], target[1] - grip[1]) < 0.012 and abs(target[2] - grip[2]) < 0.025:
                self.phase = "descend"
                self.phase_start = stage_step

        elif self.phase == "descend":
            target = [cube[0] - 0.080 * ux, cube[1] - 0.080 * uy,
                      cube[2] + (0.014, 0.0, -0.009, 0.026)[self.z_index]]
            action = self.move(grip, target, gain=1.1, xy_cap=0.35, z_cap=0.65)
            if norm2(target[0] - grip[0], target[1] - grip[1]) < 0.013 and abs(target[2] - grip[2]) < 0.012:
                self.phase = "push"
                self.phase_start = stage_step
                self.push_stall = 0

        else:  # push
            if along < -0.007 or across > 0.055:
                self.phase = "approach"
                self.phase_start = stage_step
                target = [cube[0] - 0.085 * ux, cube[1] - 0.085 * uy,
                          cube[2] + 0.095]
                action = self.move(grip, target, gain=1.2, xy_cap=0.8)
            elif distance < 0.015 and toward > max(0.0012, 0.48 * distance):
                self.phase = "brake"
                self.phase_start = stage_step
                target = [grip[0] - 0.025 * ux, grip[1] - 0.025 * uy,
                          cube[2] + 0.065]
                action = self.move(grip, target, gain=0.9, xy_cap=0.35)
            else:
                if speed > 0.0002 and along > 0.018 and along < 0.095 and toward > 0:
                    self.contact_gap = 0.80 * self.contact_gap + 0.20 * along
                    self.push_stall = 0
                else:
                    self.push_stall += 1
                if self.push_stall > 10 and distance > 0.006 and along < 0.06:
                    self.z_index = (self.z_index + 1) % 4
                    self.push_stall = 0
                if distance > 0.060:
                    gap = 0.019
                    cap = 0.70
                elif distance > 0.025:
                    gap = max(0.018, self.contact_gap - 0.026)
                    cap = 0.43
                elif distance > 0.010:
                    gap = max(0.014, self.contact_gap - 0.013)
                    cap = 0.23
                else:
                    gap = max(0.010, self.contact_gap - 0.006)
                    cap = 0.11
                if self.push_stall > 5:
                    gap -= min(0.014, 0.002 * (self.push_stall - 5))
                z = cube[2] + (0.014, 0.0, -0.009, 0.026)[self.z_index]
                target = [cube[0] - gap * ux, cube[1] - gap * uy, z]
                action = self.move(grip, target, gain=1.2, xy_cap=cap, z_cap=0.30)

        self.last_object = cube
        self.last_t = t
        self.last_action = [float(limit(x, 1.0)) for x in action]
        return self.last_action


def main():
    controller = Controller()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            typ = msg.get("type")
            if typ == "reset":
                controller.reset()
                answer = {"ok": True}
            elif typ == "close":
                answer = {"ok": True}
                print(json.dumps(answer), flush=True)
                break
            elif typ == "act":
                answer = {"action": controller.act(int(msg["t"]), msg["obs"])}
            else:
                answer = {"error": "unknown request"}
            print(json.dumps(answer, separators=(",", ":")), flush=True)
        except Exception as exc:
            print("controller error: %r" % (exc,), file=sys.stderr, flush=True)
            print('{"action":[0,0,0,0]}', flush=True)


if __name__ == "__main__":
    main()
