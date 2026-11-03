#!/usr/bin/env python3
"""Observation-only geometric controller for a two-goal Fetch push scene."""
import json
import math
import sys


def clip(x, lo=-1.0, hi=1.0):
    return min(hi, max(lo, x))


def norm2(x, y):
    return math.hypot(x, y)


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.stage = None
        self.mode = "lift"
        self.last_cube = None
        self.last_t = None
        self.last_dir = (1.0, 0.0)
        self.park_steps = 0
        self.no_motion_steps = 0
        self.cached_t = None
        self.cached_action = None

    def move_to(self, grip, target, limit=1.0):
        # The Fetch action requests a 5 cm Cartesian mocap displacement.
        # A moderate position gain prevents repeated large target changes.
        return [clip((target[i] - grip[i]) / 0.045, -limit, limit)
                for i in range(3)] + [0.0]

    def act(self, t, obs):
        s = obs["state"]
        grip = [float(v) for v in s[:3]]
        cube = [float(v) for v in s[3:6]]
        goal = [float(v) for v in obs["goal"]]
        stage = int(obs["stage"])
        step = int(obs["stage_step"])

        if stage != self.stage:
            self.stage = stage
            self.mode = "lift"
            self.park_steps = 0
            self.no_motion_steps = 0
            self.last_cube = None

        if self.last_cube is not None and self.last_t != t:
            vx = cube[0] - self.last_cube[0]
            vy = cube[1] - self.last_cube[1]
        else:
            vx = vy = 0.0
        self.last_cube = cube[:]
        self.last_t = t
        speed = norm2(vx, vy)

        ex, ey = goal[0] - cube[0], goal[1] - cube[1]
        distance = norm2(ex, ey)
        if distance > 1e-7:
            ux, uy = ex / distance, ey / distance
        else:
            ux, uy = self.last_dir

        # At a reached goal, withdraw along the approach line.  This removes
        # contact pressure while the cube settles, and the next observation
        # can still trigger a corrective push if it drifts away.
        predicted = distance - 1.2 * (vx * ux + vy * uy)
        if self.mode == "park":
            if distance > 0.00235 and speed < 0.00035 and self.park_steps >= 3:
                self.mode = "lift"
                self.no_motion_steps = 0
            else:
                self.park_steps += 1
                px = cube[0] - self.last_dir[0] * 0.10
                py = cube[1] - self.last_dir[1] * 0.10
                pz = cube[2] + 0.09
                limit = 0.5 if self.park_steps < 5 else 0.8
                return self.move_to(grip, (px, py, pz), limit)

        if distance < 0.0022 or (distance < 0.0045 and predicted < 0.0015 and speed > 0.0003):
            self.mode = "park"
            self.park_steps = 0
            return self.move_to(grip,
                                (cube[0] - self.last_dir[0] * 0.10,
                                 cube[1] - self.last_dir[1] * 0.10,
                                 cube[2] + 0.09), 0.55)

        # Move above the cube before crossing to the opposite side.  An
        # elevated route avoids accidentally pushing it away from the goal.
        high_z = cube[2] + 0.115
        push_z = cube[2] + 0.013
        behind = 0.073
        bx, by = cube[0] - ux * behind, cube[1] - uy * behind
        align_error = norm2(grip[0] - bx, grip[1] - by)

        if self.mode == "push":
            along = (cube[0] - grip[0]) * ux + (cube[1] - grip[1]) * uy
            lateral = abs((cube[0] - grip[0]) * uy - (cube[1] - grip[1]) * ux)
            turn = ux * self.last_dir[0] + uy * self.last_dir[1]
            if along < 0.015 or lateral > 0.052 or turn < 0.1:
                self.mode = "lift"
                self.no_motion_steps = 0

        if self.mode == "lift":
            if grip[2] < high_z - 0.012:
                # Rise first, especially after a goal switch or overshoot.
                return self.move_to(grip, (grip[0], grip[1], high_z), 0.9)
            self.mode = "align"

        if self.mode == "align":
            if grip[2] < high_z - 0.012:
                self.mode = "lift"
                return self.move_to(grip, (grip[0], grip[1], high_z), 0.9)
            if align_error > 0.009:
                return self.move_to(grip, (bx, by, high_z), 0.9)
            self.mode = "lower"

        if self.mode == "lower":
            if align_error > 0.022:
                self.mode = "lift"
                return self.move_to(grip, (grip[0], grip[1], high_z), 0.9)
            if abs(grip[2] - push_z) > 0.008:
                return self.move_to(grip, (bx, by, push_z), 0.7)
            self.mode = "push"

        # Receding target tracking holds the gripper on the pushing side as
        # the cube travels.  The offset and speed decrease near the goal.
        self.last_dir = (ux, uy)
        if distance > 0.06:
            desired_offset, max_action = 0.020, 0.85
        elif distance > 0.025:
            desired_offset, max_action = 0.026, 0.43
        elif distance > 0.010:
            desired_offset, max_action = 0.032, 0.23
        else:
            desired_offset, max_action = 0.038, 0.10

        # If the gripper stopped just shy of the contact surface, advance its
        # requested point in small increments until the cube starts moving.
        if speed < 0.00015 and distance > 0.0035:
            self.no_motion_steps += 1
        else:
            self.no_motion_steps = 0
        desired_offset -= min(0.020, max(0, self.no_motion_steps - 5) * 0.0015)

        if step > 135 and distance > 0.012:
            max_action = max(max_action, 0.35)
            desired_offset = min(desired_offset, 0.024)

        target = (cube[0] - ux * desired_offset,
                  cube[1] - uy * desired_offset,
                  push_z)
        return self.move_to(grip, target, max_action)


def main():
    controller = Controller()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            kind = req.get("type")
            if kind == "reset":
                controller.reset()
                reply = {"ok": True}
            elif kind == "close":
                reply = {"ok": True}
                sys.stdout.write(json.dumps(reply) + "\n")
                sys.stdout.flush()
                break
            elif kind == "act":
                t = req.get("t")
                if t == controller.cached_t:
                    action = controller.cached_action
                else:
                    action = controller.act(t, req["obs"])
                    controller.cached_t = t
                    controller.cached_action = action
                reply = {"action": action}
            else:
                reply = {"ok": False}
        except Exception as exc:
            print("controller error: {}".format(exc), file=sys.stderr)
            reply = {"action": [0.0, 0.0, 0.0, 0.0]}
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
