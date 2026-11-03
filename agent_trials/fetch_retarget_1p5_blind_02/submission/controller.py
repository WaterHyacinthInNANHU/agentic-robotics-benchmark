#!/usr/bin/env python3
"""Closed-loop FetchPush controller. Uses only the public kinematic observation."""

import json
import math
import sys


def norm(a):
    return math.hypot(a[0], a[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def unit(a):
    d = norm(a)
    if d < 1e-12:
        return (1.0, 0.0)
    return (a[0] / d, a[1] / d)


class Controller:
    def __init__(self):
        self.stage = None
        self.phase = "lift"
        self.direction = (1.0, 0.0)
        self.last_cube = None
        self.last_t = -1
        self.last_action = [0.0, 0.0, 0.0, 0.0]
        self.replans = 0
        self.wait_steps = 0

    @staticmethod
    def go(grip, target, max_xy=0.8, max_z=0.8):
        # Each action moves the mocap target by 5 cm. The gripper tracks it
        # through the MuJoCo actuator, so position feedback remains necessary.
        return [clamp((target[0] - grip[0]) / 0.05, -max_xy, max_xy),
                clamp((target[1] - grip[1]) / 0.05, -max_xy, max_xy),
                clamp((target[2] - grip[2]) / 0.05, -max_z, max_z), 0.0]

    def replan(self, cube, goal):
        self.direction = unit((goal[0] - cube[0], goal[1] - cube[1]))
        self.phase = "lift"
        self.replans += 1

    def act(self, t, obs):
        if t == self.last_t:
            return self.last_action

        state = obs["state"]
        grip = tuple(float(x) for x in state[0:3])
        cube = tuple(float(x) for x in state[3:6])
        goal = tuple(float(x) for x in obs["goal"])
        stage = int(obs["stage"])
        stage_step = int(obs["stage_step"])
        err = (goal[0] - cube[0], goal[1] - cube[1])
        dist = norm(err)

        if self.stage != stage:
            self.stage = stage
            self.replan(cube, goal)
            self.last_cube = cube
            self.wait_steps = 0

        if self.last_cube is None or t != self.last_t + 1:
            cube_move = (0.0, 0.0)
        else:
            cube_move = (cube[0] - self.last_cube[0],
                         cube[1] - self.last_cube[1])
        cube_speed = norm(cube_move)

        u = self.direction
        longitudinal = dot(err, u)
        lateral = err[0] * (-u[1]) + err[1] * u[0]
        gap = dot((cube[0] - grip[0], cube[1] - grip[1]), u)

        # At goal, lift straight up to break contact and let the cube settle.
        # Recheck every step, including after stage 1 has been scored.
        if dist < 0.0010 and self.phase != "settle":
            self.phase = "settle"
            self.wait_steps = 0

        if self.phase == "settle":
            self.wait_steps += 1
            safe_z = cube[2] + 0.125
            if grip[2] < safe_z - 0.006:
                action = [0.0, 0.0, clamp((safe_z - grip[2]) / 0.05, 0.0, 0.8), 0.0]
            else:
                action = [0.0, 0.0, 0.0, 0.0]
            # The vertical retreat can release a small amount of stored
            # contact energy. Once it has settled, correct any residual.
            if self.wait_steps >= 4 and cube_speed < 0.00015 and dist > 0.0010:
                self.replan(cube, goal)

        elif self.phase == "lift":
            safe_z = cube[2] + 0.125
            if grip[2] < safe_z - 0.006:
                action = self.go(grip, (grip[0], grip[1], safe_z), max_xy=0.0)
            else:
                self.phase = "travel"
                action = [0.0, 0.0, 0.0, 0.0]

        elif self.phase == "travel":
            # A clearance of seven centimeters avoids sweeping the fingers
            # across the cube before the gripper has reached push height.
            behind = (cube[0] - 0.070 * u[0], cube[1] - 0.070 * u[1])
            safe_z = cube[2] + 0.125
            if norm((behind[0] - grip[0], behind[1] - grip[1])) > 0.005:
                action = self.go(grip, (behind[0], behind[1], safe_z), max_xy=0.8)
            else:
                self.phase = "lower"
                action = [0.0, 0.0, 0.0, 0.0]

        elif self.phase == "lower":
            # The cube's observed center is the correct pushing height for
            # the stock closed Fetch gripper.
            push_z = cube[2] + 0.002
            if abs(grip[2] - push_z) > 0.003:
                action = self.go(grip, (grip[0], grip[1], push_z), max_xy=0.0,
                                 max_z=0.5)
            else:
                self.phase = "push"
                action = [0.0, 0.0, 0.0, 0.0]

        else:  # push
            # A sharply changed residual is best reached from a new side.
            # This also permits corrections when the cube has passed a goal.
            if dist > 0.0012 and (longitudinal < -0.0005 or
                                   (abs(lateral) > 0.004 and
                                    abs(lateral) > 0.45 * max(longitudinal, 0.0))):
                self.replan(cube, goal)
                action = [0.0, 0.0, 0.5, 0.0]
            else:
                # Position of the gripper relative to the cube supplies a
                # contact-gap estimate. Advance until contact, then use
                # progressively smaller increments as the cube approaches.
                advance = 0.16 * max(0.0, longitudinal)
                if gap > 0.045:
                    advance += 0.40 * (gap - 0.045)
                # Damping reduces overshoot during rapid motion.
                advance -= 0.35 * dot(cube_move, u)
                if dist < 0.015:
                    advance = clamp(advance, 0.0, 0.0025)
                else:
                    advance = clamp(advance, 0.0, 0.012)
                # When the cube has small transverse drift, steer the
                # gripper slightly while maintaining contact.
                sideways = clamp(0.22 * lateral, -0.0010, 0.0010)
                action = [(advance * u[0] - sideways * u[1]) / 0.05,
                          (advance * u[1] + sideways * u[0]) / 0.05,
                          clamp((cube[2] + 0.002 - grip[2]) / 0.05, -0.12, 0.12),
                          0.0]
                action[0] = clamp(action[0], -1.0, 1.0)
                action[1] = clamp(action[1], -1.0, 1.0)

        self.last_cube = cube
        self.last_t = t
        self.last_action = action
        return action


def main():
    controller = Controller()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            message = json.loads(line)
            kind = message.get("type")
            if kind == "reset":
                controller = Controller()
                reply = {"ok": True}
            elif kind == "act":
                reply = {"action": controller.act(int(message["t"]), message["obs"])}
            elif kind == "close":
                print(json.dumps({"ok": True}), flush=True)
                break
            else:
                reply = {"ok": False}
        except Exception as exc:
            print("controller error: %r" % (exc,), file=sys.stderr, flush=True)
            reply = {"action": [0.0, 0.0, 0.0, 0.0]}
        print(json.dumps(reply, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
