"""Feedback controller for the public square-nut handle-carry task.

The JSON protocol and the controller itself use only the Python standard
library, keeping startup fast in the grader image.
"""

import json
import math
import sys


def clip(x, lo=-1.0, hi=1.0):
    return min(hi, max(lo, x))


def norm2(x, y):
    return math.hypot(x, y)


def norm3(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rotate_by_quat(v, q):
    """Rotate v by a quaternion in robosuite's xyzw convention."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "approach"
        self.mode_t = 0
        self.last_t = -1
        self.attempt = 0
        self.baseline_nut_z = 0.0
        self.lift_z = 0.0
        self.lift_xy = [0.0, 0.0]
        self.carry_z = 1.12
        self.lost_count = 0

    def switch(self, mode, t):
        self.mode = mode
        self.mode_t = t
        print("mode", mode, "t", t, "attempt", self.attempt,
              file=sys.stderr, flush=True)

    def action(self, t, o):
        e = o["eef_pos"]
        h = o["handle_pos"]
        n = o["nut_pos"]
        p = o["peg_pos"]
        q = o["eef_quat"]
        elapsed = t - self.mode_t

        # The handle is a bar radiating outward from the nut. Put the
        # gripper's jaw-closing axis across that bar. A 180 degree yaw is
        # equivalent, so choose the shorter turn.
        bar = [h[0] - n[0], h[1] - n[1]]
        if norm2(*bar) > 0.01:
            target_jaw_angle = math.atan2(bar[1], bar[0]) + math.pi / 2.0
            jaw = rotate_by_quat((0.0, 1.0, 0.0), q)
            jaw_angle = math.atan2(jaw[1], jaw[0])
            yaw_error = wrap_angle(target_jaw_angle - jaw_angle)
            if yaw_error > math.pi / 2.0:
                yaw_error -= math.pi
            elif yaw_error < -math.pi / 2.0:
                yaw_error += math.pi
        else:
            yaw_error = 0.0

        if self.mode == "approach":
            hover = max(h[2] + 0.14, 0.98)
            xy_err = norm2(h[0] - e[0], h[1] - e[1])
            if (xy_err < 0.018 and abs(hover - e[2]) < 0.022
                    and abs(yaw_error) < 0.16) or elapsed > 85:
                self.switch("descend", t)
        elif self.mode == "descend":
            grasp_z = h[2] + (0.003, -0.006, 0.012, -0.012)[min(self.attempt, 3)]
            if (norm2(h[0] - e[0], h[1] - e[1]) < 0.014
                    and abs(grasp_z - e[2]) < 0.014) or elapsed > 75:
                self.baseline_nut_z = n[2]
                self.switch("close", t)
        elif self.mode == "close":
            if elapsed >= 19:
                self.lift_z = max(h[2] + 0.19, 1.07)
                self.lift_xy = [e[0], e[1]]
                self.switch("lift", t)
        elif self.mode == "lift":
            if n[2] > max(0.94, self.baseline_nut_z + 0.075) and norm3(e, h) < 0.095:
                self.carry_z = clip(1.10 + e[2] - n[2], 1.05, 1.28)
                self.lost_count = 0
                self.switch("carry", t)
            elif elapsed > 62 or (e[2] > self.lift_z - 0.025 and elapsed > 30):
                self.attempt += 1
                self.switch("approach", t)
        elif self.mode == "carry":
            if norm3(e, h) > 0.11 or n[2] < 0.88:
                self.lost_count += 1
            else:
                self.lost_count = 0
            if self.lost_count > 7:
                self.attempt += 1
                self.switch("approach", t)

        if self.mode == "approach":
            target = [h[0], h[1], max(h[2] + 0.14, 0.98)]
            grip = -1.0
            xy_speed = 0.75
            z_speed = 0.75
        elif self.mode == "descend":
            target = [h[0], h[1], h[2] +
                      (0.003, -0.006, 0.012, -0.012)[min(self.attempt, 3)]]
            grip = -1.0
            xy_speed = 0.38
            z_speed = 0.43
        elif self.mode == "close":
            target = [h[0], h[1], h[2] +
                      (0.003, -0.006, 0.012, -0.012)[min(self.attempt, 3)]]
            grip = 1.0
            xy_speed = 0.24
            z_speed = 0.24
        elif self.mode == "lift":
            target = [self.lift_xy[0], self.lift_xy[1], self.lift_z]
            grip = 1.0
            xy_speed = 0.32
            z_speed = 0.42
        else:
            # Feedback on the carried nut directly compensates for the
            # small observation bias and any changes in its grip pose.
            target = [e[0] + p[0] - n[0], e[1] + p[1] - n[1], self.carry_z]
            grip = 1.0
            xy_speed = 0.48 if norm2(p[0] - n[0], p[1] - n[1]) > 0.055 else 0.25
            z_speed = 0.43

        dx = clip((target[0] - e[0]) / 0.05, -xy_speed, xy_speed)
        dy = clip((target[1] - e[1]) / 0.05, -xy_speed, xy_speed)
        length = norm2(dx, dy)
        if length > xy_speed:
            dx *= xy_speed / length
            dy *= xy_speed / length
        dz = clip((target[2] - e[2]) / 0.05, -z_speed, z_speed)
        rz = clip(yaw_error / 0.5, -0.7, 0.7) if self.mode in ("approach", "descend") else 0.0
        return [dx, dy, dz, 0.0, 0.0, rz, grip]


def main():
    c = Controller()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            typ = msg.get("type")
            if typ == "reset":
                c.reset()
                out = {"ok": True}
            elif typ == "close":
                out = {"ok": True}
                print(json.dumps(out), flush=True)
                return
            elif typ == "act":
                t = int(msg["t"])
                if t < c.last_t:
                    c.reset()
                c.last_t = max(c.last_t, t)
                out = {"action": c.action(t, msg["obs"])}
            else:
                out = {"error": "unknown message type"}
            print(json.dumps(out, separators=(",", ":")), flush=True)
        except Exception as exc:
            print("controller error", repr(exc), file=sys.stderr, flush=True)
            print('{"action":[0,0,0,0,0,0,1]}', flush=True)


if __name__ == "__main__":
    main()
