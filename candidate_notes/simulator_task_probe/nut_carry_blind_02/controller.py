#!/usr/bin/env python3
"""Feedback pick-and-carry controller for the native square nut scene."""

import json
import math
import sys


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def distance(a, b):
    return norm([x - y for x, y in zip(a, b)])


def quat_matrix(q):
    x, y, z, w = q
    d = x*x + y*y + z*z + w*w
    if d < 1e-12:
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    s = 2. / d
    xx, yy, zz = s*x*x, s*y*y, s*z*z
    xy, xz, yz = s*x*y, s*x*z, s*y*z
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    return [
        [1-yy-zz, xy-wz, xz+wy],
        [xy+wz, 1-xx-zz, yz-wx],
        [xz-wy, yz+wx, 1-xx-yy],
    ]


def matmul_transpose(a, b):
    return [[sum(a[i][k] * b[j][k] for k in range(3))
             for j in range(3)] for i in range(3)]


def rotation_vector(a, b):
    """World-frame rotation taking current matrix b to desired matrix a."""
    r = matmul_transpose(a, b)
    c = clip((r[0][0] + r[1][1] + r[2][2] - 1.) / 2., -1., 1.)
    angle = math.acos(c)
    skew = [r[2][1]-r[1][2], r[0][2]-r[2][0], r[1][0]-r[0][1]]
    if angle < 1e-5:
        return [0.5*x for x in skew]
    if math.pi-angle < 1e-4:
        axis = [math.sqrt(max(0., (r[i][i]+1.)/2.)) for i in range(3)]
        if skew[0] < 0.: axis[0] *= -1.
        if skew[1] < 0.: axis[1] *= -1.
        if skew[2] < 0.: axis[2] *= -1.
        n = norm(axis)
        return [angle*x/max(n, 1e-8) for x in axis]
    scale = angle/(2.*math.sin(angle))
    return [scale*x for x in skew]


def grasp_orientation(nut_quat, current_quat):
    nut = quat_matrix(nut_quat)
    current = quat_matrix(current_quat)
    # Nut local Y is across the narrow handle; gripper local X closes jaws.
    x = [nut[0][1], nut[1][1], 0.]
    n = max(norm(x), 1e-8)
    x = [x[0]/n, x[1]/n, 0.]
    # The two finger directions are physically equivalent. Choose the closer one.
    if x[0]*current[0][0] + x[1]*current[1][0] < 0.:
        x = [-x[0], -x[1], 0.]
    y = [x[1], -x[0], 0.]
    z = [0., 0., -1.]
    return [[x[i], y[i], z[i]] for i in range(3)]


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "approach"
        self.phase_start = 0
        self.last_t = -1
        self.desired_ori = None
        self.attempt = 0
        self.initial_nut_z = None

    def enter(self, phase, t):
        self.phase = phase
        self.phase_start = t

    def command(self, obs, target, grip, orientation=None, speed=1.):
        eef = obs["eef_pos"]
        current_ori = quat_matrix(obs["eef_quat"])
        desired_ori = orientation or grasp_orientation(obs["nut_quat"], obs["eef_quat"])
        error = [target[i]-eef[i] for i in range(3)]
        # OSC input is a desired change of at most 5 cm per policy step.
        # A slightly conservative gain keeps the grasp steady at contact.
        pos_action = [clip(speed * error[i] / .05) for i in range(3)]
        rot_error = rotation_vector(desired_ori, current_ori)
        rot_action = [clip(.72 * x / .5) for x in rot_error]
        return pos_action + rot_action + [float(grip)]

    def act(self, t, obs):
        eef = obs["eef_pos"]
        nut = obs["nut_pos"]
        handle = obs["handle_pos"]
        peg = obs["peg_pos"]
        if self.initial_nut_z is None:
            self.initial_nut_z = nut[2]

        if t != self.last_t:
            elapsed = t - self.phase_start
            xy_handle = distance(eef[:2], handle[:2])
            grasp_distance = distance(eef, handle)
            if self.phase == "approach":
                high = max(handle[2] + .12, .975)
                ori_error = norm(rotation_vector(grasp_orientation(obs["nut_quat"], obs["eef_quat"]), quat_matrix(obs["eef_quat"])))
                if ((xy_handle < .015 and abs(eef[2]-high) < .025 and ori_error < .22)
                        or elapsed > 90):
                    self.enter("descend", t)
            elif self.phase == "descend":
                grip_z = handle[2] - .002
                ori_error = norm(rotation_vector(grasp_orientation(obs["nut_quat"], obs["eef_quat"]), quat_matrix(obs["eef_quat"])))
                if ((elapsed > 8 and xy_handle < .009 and abs(eef[2]-grip_z) < .006 and ori_error < .16)
                        or elapsed > 75):
                    self.desired_ori = grasp_orientation(obs["nut_quat"], obs["eef_quat"])
                    self.enter("close", t)
            elif self.phase == "close":
                if elapsed >= 18:
                    self.enter("lift", t)
            elif self.phase == "lift":
                if nut[2] > 1.035 and grasp_distance < .065:
                    self.enter("carry", t)
                elif ((elapsed > 34 and nut[2] < .97 and grasp_distance > .07)
                      or (elapsed > 65 and nut[2] < 1.025)):
                    self.attempt += 1
                    self.desired_ori = None
                    self.enter("approach", t)
            elif self.phase == "carry":
                if nut[2] < .96 and grasp_distance > .09:
                    self.attempt += 1
                    self.desired_ori = None
                    self.enter("approach", t)
            self.last_t = t

        if self.phase == "approach":
            target = [handle[0], handle[1], max(handle[2]+.12, .975)]
            return self.command(obs, target, -1.)
        if self.phase == "descend":
            target = [handle[0], handle[1], handle[2]-.002]
            return self.command(obs, target, -1., speed=.72)
        if self.phase == "close":
            target = [handle[0], handle[1], handle[2]-.002]
            return self.command(obs, target, 1., self.desired_ori, speed=.50)
        if self.phase == "lift":
            target = [eef[0] + .55*(handle[0]-eef[0]),
                      eef[1] + .55*(handle[1]-eef[1]), 1.105]
            return self.command(obs, target, 1., self.desired_ori, speed=.85)

        # Move by the nut's observed displacement to the peg. This also
        # compensates for rotation of the nut about the grasped handle.
        xy_error = [peg[i]-nut[i] for i in range(2)]
        target = [eef[0]+.7*xy_error[0], eef[1]+.7*xy_error[1],
                  eef[2]+.7*(1.095-nut[2])]
        return self.command(obs, target, 1., self.desired_ori, speed=.82)


def main():
    controller = Controller()
    print("READY", flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request.get("type")
            if kind == "reset":
                controller.reset()
                reply = {"ok": True}
            elif kind == "act":
                action = controller.act(int(request["t"]), request["obs"])
                reply = {"action": [float(clip(x)) for x in action]}
            elif kind == "close":
                print(json.dumps({"ok": True}), flush=True)
                return
            else:
                reply = {"ok": False}
            print(json.dumps(reply, separators=(",", ":")), flush=True)
        except Exception as exc:
            print("controller error: " + repr(exc), file=sys.stderr, flush=True)
            print('{"action":[0,0,0,0,0,0,1]}', flush=True)


if __name__ == "__main__":
    main()
