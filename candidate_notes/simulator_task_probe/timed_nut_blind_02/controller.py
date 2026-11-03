#!/usr/bin/env python3
"""Closed-loop OSC controller for the public NutAssemblySquare observation."""
import json
import math
import sys

import numpy as np


def unit_xy(v):
    n = math.hypot(float(v[0]), float(v[1]))
    return np.array([v[0] / n, v[1] / n]) if n > 1e-8 else np.array([1.0, 0.0])


def wrap(x, period=2 * math.pi):
    return (x + period / 2) % period - period / 2


def quat_matrix(q):
    x, y, z, w = [float(i) for i in q]
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def down_matrix(jaw_angle):
    c, s = math.cos(jaw_angle), math.sin(jaw_angle)
    # X is the jaw-closing axis; Z points toward the tabletop.
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def rotation_vector(desired, current):
    r = desired @ current.T
    v = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
    sn = np.linalg.norm(v) / 2
    cs = float(np.clip((np.trace(r)-1)/2, -1, 1))
    theta = math.atan2(sn, cs)
    if sn < 1e-7:
        return v / 2
    return v * (theta / (2 * sn))


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 'approach_high'
        self.enter_t = 0
        self.last_t = -1
        self.initial_nut_z = None
        self.grasp_z = None
        self.retry = 0
        self.last_nut_z = None
        self.stall = 0

    def phase_to(self, name, t):
        self.phase = name
        self.enter_t = t
        self.stall = 0

    def move(self, eef, quat, target, yaw, grip, z_limit=1.0):
        dp = (np.asarray(target) - eef) / 0.05
        dp = np.clip(dp * 1.15, -1, 1)
        dp[2] = np.clip(dp[2], -z_limit, z_limit)
        current = quat_matrix(quat)
        x_cur = current[:2, 0]
        # A parallel-jaw grasp is unchanged by a half turn; avoid such a flip.
        yaw = math.atan2(x_cur[1], x_cur[0]) + wrap(yaw - math.atan2(x_cur[1], x_cur[0]), math.pi)
        rot = rotation_vector(down_matrix(yaw), current) / 0.5
        rot = np.clip(rot * 0.8, -1, 1)
        return [float(x) for x in dp] + [float(x) for x in rot] + [float(grip)]

    def act(self, t, obs):
        eef = np.asarray(obs['eef_pos'], dtype=float)
        nut = np.asarray(obs['nut_pos'], dtype=float)
        handle = np.asarray(obs['handle_pos'], dtype=float)
        peg = np.asarray(obs['peg_pos'], dtype=float)
        quat = obs['eef_quat']
        r = quat_matrix(quat)
        jaw_yaw = math.atan2(r[1, 0], r[0, 0])
        axis = unit_xy(handle - nut)
        nut_yaw = math.atan2(axis[1], axis[0])
        grasp_jaw_yaw = nut_yaw + math.pi/2
        if self.initial_nut_z is None:
            self.initial_nut_z = float(nut[2])
        if t != self.last_t:
            self.last_t = t
            age = t - self.enter_t
            xy_err = float(np.linalg.norm((eef - handle)[:2]))
            yaw_err = abs(wrap(grasp_jaw_yaw - jaw_yaw, math.pi))
            if self.phase == 'approach_high':
                if (xy_err < 0.014 and abs(eef[2] - (handle[2]+0.085)) < 0.025 and yaw_err < 0.14) or age > 42:
                    self.phase_to('approach_low', t)
            elif self.phase == 'approach_low':
                if (xy_err < 0.008 and abs(eef[2] - (handle[2]+0.002)) < 0.008 and yaw_err < 0.14) or age > 27:
                    self.grasp_z = float(eef[2])
                    self.phase_to('close', t)
            elif self.phase == 'close':
                if age >= 8:
                    self.phase_to('lift', t)
            elif self.phase == 'lift':
                if nut[2] > self.initial_nut_z + 0.095 and eef[2] > 0.98:
                    self.phase_to('transfer', t)
                elif age > 20:
                    if nut[2] < self.initial_nut_z + 0.045 and self.retry < 2:
                        self.retry += 1
                        self.phase_to('approach_high', t)
                    else:
                        self.phase_to('transfer', t)
            elif self.phase == 'transfer':
                err_xy = float(np.linalg.norm((nut-peg)[:2]))
                yaw_res = abs(wrap(nut_yaw, math.pi/2))
                if (err_xy < 0.006 and yaw_res < 0.075 and nut[2] > 0.98) or age > 54:
                    self.phase_to('lower', t)
            elif self.phase == 'lower':
                if self.last_nut_z is not None and abs(nut[2] - self.last_nut_z) < 0.0015 and nut[2] > 0.89:
                    self.stall += 1
                else:
                    self.stall = 0
                if (nut[2] < 0.864 and np.linalg.norm((nut-peg)[:2]) < 0.022) or age > 43:
                    self.phase_to('release', t)
            elif self.phase == 'release':
                if age >= 7:
                    self.phase_to('retract', t)
            self.last_nut_z = float(nut[2])

        if self.phase == 'approach_high':
            target = handle.copy()
            target[2] += 0.085
            return self.move(eef, quat, target, grasp_jaw_yaw, -1)
        if self.phase == 'approach_low':
            target = handle.copy()
            target[2] += 0.002
            return self.move(eef, quat, target, grasp_jaw_yaw, -1, 0.4)
        if self.phase == 'close':
            target = handle.copy()
            target[2] += 0.002
            return self.move(eef, quat, target, grasp_jaw_yaw, 1, 0.3)
        if self.phase == 'lift':
            target = eef.copy()
            target[:2] = handle[:2]
            target[2] = max(1.035, (self.grasp_z or eef[2])+0.18)
            return self.move(eef, quat, target, jaw_yaw, 1)

        # For a held nut, relative EEF-to-nut position is measured every step.
        # This cancels grasp offsets; a few mm of fixed vision bias are tolerated
        # by the nut/peg clearance.
        target = eef + (np.array([peg[0], peg[1], nut[2]]) - nut)
        yaw_correction = -wrap(nut_yaw, math.pi/2)
        desired_jaw_yaw = jaw_yaw + yaw_correction
        if self.phase == 'transfer':
            target[2] = eef[2] + (1.025 - nut[2])
            # Keep the nut safely above the 0.95 m peg top while traversing.
            if nut[2] < 0.975:
                target[:2] = eef[:2]
            return self.move(eef, quat, target, desired_jaw_yaw, 1, 0.8)
        if self.phase == 'lower':
            target[2] = eef[2] + (0.853 - nut[2])
            # The OSC goal can push hard on a peg; descend in small increments.
            return self.move(eef, quat, target, desired_jaw_yaw, 1, 0.24)
        if self.phase == 'release':
            target = eef.copy()
            return self.move(eef, quat, target, jaw_yaw, -1, 0.2)
        # Retract up and away from the center, leaving the nut behind.
        away = unit_xy(handle - nut)
        target = eef.copy()
        target[:2] += away * 0.10
        target[2] = max(eef[2], 1.04)
        return self.move(eef, quat, target, jaw_yaw, -1)


def main():
    policy = Controller()
    print('READY', flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            kind = req.get('type')
            if kind == 'reset':
                policy.reset()
                result = {'ok': True}
            elif kind == 'close':
                result = {'ok': True}
                print(json.dumps(result), flush=True)
                return
            elif kind == 'act':
                result = {'action': policy.act(int(req['t']), req['obs'])}
            else:
                result = {'ok': False}
            print(json.dumps(result, separators=(',', ':')), flush=True)
        except Exception as exc:
            print('controller error:', repr(exc), file=sys.stderr, flush=True)
            print(json.dumps({'action': [0, 0, 0, 0, 0, 0, -1]}), flush=True)


if __name__ == '__main__':
    main()
