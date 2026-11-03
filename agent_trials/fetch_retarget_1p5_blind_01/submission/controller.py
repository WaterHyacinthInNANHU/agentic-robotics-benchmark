#!/usr/bin/env python3
"""State feedback controller for two continuous FetchPush targets.

The program deliberately uses only the public observation.  All distances are
metres.  No simulator package is needed in the controller process.
"""
import json
import math
import sys


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def norm(a):
    return math.hypot(a[0], a[1])


def unit(a, fallback=(1.0, 0.0)):
    n = norm(a)
    return (a[0] / n, a[1] / n) if n > 1e-9 else fallback


def clip(x, lo, hi):
    return max(lo, min(hi, x))


class Controller:
    def __init__(self):
        self.stage = None
        self.phase = "lift"
        self.direction = (1.0, 0.0)
        self.contact_radius = 0.055
        self.last_obj = None
        self.last_grip = None
        self.still = 0
        self.replans = 0
        self.push_age = 0
        self.stall = 0

    def pose_action(self, gripper, xyz, xy_limit=0.8, z_limit=0.5):
        # A Fetch action moves the mocap target by at most 5 cm in each axis.
        # Limit the XY norm to keep contact speed predictable at every heading.
        dx = (xyz[0] - gripper[0]) / 0.05
        dy = (xyz[1] - gripper[1]) / 0.05
        n = math.hypot(dx, dy)
        if n > xy_limit:
            dx *= xy_limit / n
            dy *= xy_limit / n
        dz = clip((xyz[2] - gripper[2]) / 0.05, -z_limit, z_limit)
        return [clip(dx, -1, 1), clip(dy, -1, 1), dz, 0.0]

    def act(self, observation):
        s = observation["state"]
        g = tuple(float(v) for v in s[:3])
        o = tuple(float(v) for v in s[3:6])
        target = tuple(float(v) for v in observation["goal"])
        stage = int(observation["stage"])
        step = int(observation["stage_step"])
        delta = (target[0] - o[0], target[1] - o[1])
        error = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + (target[2] - o[2]) ** 2)
        od = (0.0, 0.0) if self.last_obj is None else (o[0] - self.last_obj[0], o[1] - self.last_obj[1])
        gd = (0.0, 0.0) if self.last_grip is None else (g[0] - self.last_grip[0], g[1] - self.last_grip[1])

        if stage != self.stage:
            self.stage = stage
            self.phase = "lift"
            self.direction = unit(delta)
            self.contact_radius = 0.055
            self.still = 0
            self.replans = 0
            self.push_age = 0
            self.stall = 0
            od = (0.0, 0.0)

        high_z = max(o[2] + 0.145, 0.555)
        push_z = o[2] + 0.025
        # A new approach line is needed if the cube is beyond the target or
        # the gripper has ended up beside/in front of it.
        cur_dir = unit(delta, self.direction)

        if self.phase == "lift":
            action = self.pose_action(g, (g[0], g[1], high_z), xy_limit=0.0, z_limit=0.75)
            if g[2] > o[2] + 0.115:
                self.phase = "above"

        elif self.phase == "above":
            self.direction = cur_dir
            bx = o[0] - self.direction[0] * 0.105
            by = o[1] - self.direction[1] * 0.105
            action = self.pose_action(g, (bx, by, high_z), xy_limit=0.85, z_limit=0.55)
            if math.hypot(g[0] - bx, g[1] - by) < 0.008 and abs(g[2] - high_z) < 0.018:
                self.phase = "lower"

        elif self.phase == "lower":
            bx = o[0] - self.direction[0] * 0.105
            by = o[1] - self.direction[1] * 0.105
            action = self.pose_action(g, (bx, by, push_z), xy_limit=0.12, z_limit=0.38)
            if abs(g[2] - push_z) < 0.008:
                self.phase = "push"
                self.push_age = 0
                self.stall = 0

        elif self.phase == "settle":
            # A vertical withdrawal prevents the arm's position servo from
            # injecting another small impulse after the cube has stopped.
            action = self.pose_action(g, (g[0], g[1], high_z), xy_limit=0.0, z_limit=0.35)
            if error > 0.0011 and step < 147 and g[2] > o[2] + 0.10:
                self.phase = "above"
                self.replans += 1
            elif error < 0.0011:
                self.still += 1

        else:  # push
            self.push_age += 1
            along = dot((o[0] - g[0], o[1] - g[1]), cur_dir)
            lateral = abs((o[0] - g[0]) * cur_dir[1] - (o[1] - g[1]) * cur_dir[0])
            if (along < 0.020 or lateral > 0.047) and error > 0.0010 and step < 145:
                self.phase = "lift"
                self.replans += 1
                action = self.pose_action(g, (g[0], g[1], high_z), xy_limit=0.0, z_limit=0.65)
            else:
                # Estimate the physical contact gap while the object is moving
                # in the direction of the pusher.  Its value varies with push
                # angle because the gripper is not rotationally symmetric.
                progress = dot(od, self.direction)
                grip_progress = dot(gd, self.direction)
                gap = dot((o[0] - g[0], o[1] - g[1]), self.direction)
                if progress > 0.00025 and grip_progress > 0.00015 and 0.030 < gap < 0.085:
                    self.contact_radius = clip(0.85 * self.contact_radius + 0.15 * gap, 0.037, 0.075)

                # Re-aim while still a few centimetres away; close to the goal
                # a stable line avoids large swings in the desired hand pose.
                if error > 0.025 and dot(self.direction, cur_dir) > 0.92:
                    self.direction = unit((0.85 * self.direction[0] + 0.15 * cur_dir[0],
                                           0.85 * self.direction[1] + 0.15 * cur_dir[1]))

                # The target for the gripper is one contact gap behind the cube
                # goal.  This yields a position-controlled stop instead of a
                # velocity command that might push through the target.
                # If the arm reaches that pose before the cube reaches goal,
                # gently advance the pose.  This compensates for contact-gap
                # estimation errors without a sudden final shove.
                goal_progress = dot(delta, self.direction)
                if norm(od) < 0.00012 and goal_progress > 0.0011 and self.push_age > 7:
                    self.stall += 1
                else:
                    self.stall = 0
                if self.stall > 5:
                    self.contact_radius = max(0.032, self.contact_radius - 0.001)

                px = target[0] - self.direction[0] * self.contact_radius
                py = target[1] - self.direction[1] * self.contact_radius
                if error > 0.070:
                    xy_limit = 0.43
                elif error > 0.030:
                    xy_limit = 0.26
                elif error > 0.012:
                    xy_limit = 0.12
                elif error > 0.004:
                    xy_limit = 0.055
                else:
                    xy_limit = 0.025
                action = self.pose_action(g, (px, py, push_z), xy_limit=xy_limit, z_limit=0.12)

                speed = norm(od)
                if error < 0.0008 and speed < 0.0003:
                    self.still += 1
                else:
                    self.still = 0
                if self.still >= 2:
                    self.phase = "settle"

        self.last_obj = o
        self.last_grip = g
        return action


def main():
    controller = Controller()
    last_t = None
    last_action = None
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            query = json.loads(line)
            kind = query.get("type")
            if kind == "reset":
                controller = Controller()
                last_t = None
                last_action = None
                reply = {"ok": True}
            elif kind == "act":
                t = int(query["t"])
                if t == last_t:
                    reply = {"action": last_action}
                else:
                    last_action = controller.act(query["obs"])
                    last_t = t
                    reply = {"action": last_action}
            elif kind == "close":
                print(json.dumps({"ok": True}), flush=True)
                return
            else:
                reply = {"ok": False}
            print(json.dumps(reply, separators=(",", ":")), flush=True)
        except Exception as exc:
            print("controller error: " + repr(exc), file=sys.stderr, flush=True)
            print('{"action":[0,0,0,0]}', flush=True)


if __name__ == "__main__":
    main()
