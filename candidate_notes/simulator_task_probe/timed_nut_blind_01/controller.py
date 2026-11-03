#!/usr/bin/env python3
"""Observation-only OSC pose controller for the square nut task."""
import json
import math
import sys


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz]


def qinv(q):
    return [-q[0], -q[1], -q[2], q[3]]


def yaw(q):
    x, y, z, w = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def placement_yaw(current):
    # Equivalent quarter-turns fit the square peg. Prefer keeping the handle
    # lateral or on the near side so the Panda need not reach beyond the peg.
    choices = [-math.pi, -math.pi / 2, math.pi / 2, math.pi]
    return min(choices, key=lambda a: abs(wrap_angle(a-current)) +
               (0.12 if abs(a) == math.pi else 0.0))


def grip_quat(nut_yaw):
    # Gripper local x is the finger-closing axis. Point it across the narrow
    # dimension of the handle while local z points down to the tabletop.
    h = nut_yaw / 2.0
    return qmul([0, 0, math.sin(h), math.cos(h)],
                [math.sqrt(0.5), math.sqrt(0.5), 0, 0])


def rot_error(qgoal, qcur):
    q = qmul(qgoal, qinv(qcur))
    if q[3] < 0:
        q = [-v for v in q]
    nv = norm(q[:3])
    if nv < 1e-8:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(nv, q[3])
    return [angle * v / nv for v in q[:3]]


class Controller:
    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 'hover'
        self.phase_t = 0
        self.last_t = -1
        self.initial_z = None
        self.snap_yaw = None
        self.grasp_nut_z = None
        self.grasp_eef_z = None
        self.retry_count = 0
        self.state_before_t = None

    def change(self, phase, t):
        self.phase = phase
        self.phase_t = t

    def pose_action(self, eef, q, target, qtarget, grip, gain=1.0):
        err = [target[i] - eef[i] for i in range(3)]
        r = rot_error(qtarget, q)
        # The stock OSC_POSE maps unit translation to 5 cm and unit rotation
        # to 0.5 rad, in the world-aligned Panda base frame.
        a = [clip(gain * d / 0.05) for d in err]
        a += [clip(0.9 * d / 0.5) for d in r]
        a.append(grip)
        return a

    def act(self, t, obs):
        if t == self.last_t and self.state_before_t is not None:
            self.phase, self.phase_t, self.retry_count = self.state_before_t
        elif t > self.last_t:
            self.state_before_t = (self.phase, self.phase_t, self.retry_count)
        eef = [float(v) for v in obs['eef_pos']]
        eq = [float(v) for v in obs['eef_quat']]
        nut = [float(v) for v in obs['nut_pos']]
        nq = [float(v) for v in obs['nut_quat']]
        handle = [float(v) for v in obs['handle_pos']]
        peg = [float(v) for v in obs['peg_pos']]
        ny = yaw(nq)
        if self.initial_z is None:
            self.initial_z = nut[2]
            self.snap_yaw = placement_yaw(ny)
        if t < self.last_t:
            self.reset()
            self.initial_z = nut[2]
            self.snap_yaw = placement_yaw(ny)
            self.state_before_t = (self.phase, self.phase_t, self.retry_count)
        self.last_t = t

        xyh = norm([eef[0]-handle[0], eef[1]-handle[1]])
        qgrasp = grip_quat(ny)
        qplace = grip_quat(self.snap_yaw)
        re = norm(rot_error(qgrasp, eq))
        elapsed = t - self.phase_t

        # A failed pickup can be retried while sufficient time remains.
        if self.phase in ('lift', 'transfer') and t < 100 and self.retry_count < 1:
            if elapsed > 13 and nut[2] < self.initial_z + 0.025:
                self.retry_count += 1
                self.change('hover', t)
                elapsed = 0

        if self.phase == 'hover':
            target = [handle[0], handle[1], handle[2] + 0.095]
            if (xyh < 0.013 and abs(eef[2]-target[2]) < 0.018 and re < 0.28) or elapsed > 42:
                self.change('descend', t)
            return self.pose_action(eef, eq, target, qgrasp, -1.0)

        if self.phase == 'descend':
            target = [handle[0], handle[1], handle[2] + 0.003]
            if (xyh < 0.009 and abs(eef[2]-target[2]) < 0.012 and re < 0.20) or elapsed > 30:
                self.change('close', t)
            return self.pose_action(eef, eq, target, qgrasp, -1.0, 0.8)

        if self.phase == 'close':
            target = [handle[0], handle[1], handle[2] + 0.003]
            if elapsed >= 10:
                self.grasp_nut_z = nut[2]
                self.grasp_eef_z = eef[2]
                self.change('lift', t)
            return self.pose_action(eef, eq, target, qgrasp, 1.0, 0.65)

        if self.phase == 'lift':
            # Position the nut safely above the peg before horizontal travel.
            target = [handle[0], handle[1],
                      eef[2] + (peg[2] + 0.14 - nut[2])]
            target[2] = min(target[2], 1.12)
            if nut[2] > 0.945 or elapsed > 29:
                self.change('transfer', t)
            return self.pose_action(eef, eq, target, qgrasp, 1.0)

        if self.phase == 'transfer':
            target = [eef[0] + peg[0] - nut[0],
                      eef[1] + peg[1] - nut[1],
                      eef[2] + peg[2] + 0.14 - nut[2]]
            target[2] = min(target[2], 1.12)
            dxy = norm([nut[0]-peg[0], nut[1]-peg[1]])
            if (dxy < 0.006 and nut[2] > peg[2] + 0.112 and elapsed > 7) or elapsed > 49 or t > 119:
                self.change('lower', t)
            return self.pose_action(eef, eq, target, qplace, 1.0)

        if self.phase == 'lower':
            target = [eef[0] + peg[0] - nut[0],
                      eef[1] + peg[1] - nut[1],
                      eef[2] + peg[2] - nut[2]]
            # Keep the descent measured to limit impacts on the peg rim.
            target[2] = max(eef[2] - 0.025, target[2])
            dxy = norm([nut[0]-peg[0], nut[1]-peg[1]])
            if dxy > 0.008 and nut[2] > peg[2] + 0.115 and t < 137:
                target[2] = eef[2] + peg[2] + 0.14 - nut[2]
            if (nut[2] < peg[2] + 0.012 and dxy < 0.022 and elapsed > 6) or elapsed > 32 or t > 146:
                self.change('release', t)
            return self.pose_action(eef, eq, target, qplace, 1.0)

        if self.phase == 'release':
            target = [eef[0] + peg[0] - nut[0],
                      eef[1] + peg[1] - nut[1],
                      eef[2]]
            if elapsed >= 7:
                self.change('retreat', t)
            return self.pose_action(eef, eq, target, qplace, -1.0, 0.7)

        target = [eef[0], eef[1], max(1.00, nut[2] + 0.18)]
        return self.pose_action(eef, eq, target, qplace, -1.0, 1.0)


def main():
    ctrl = Controller()
    print('READY', flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            kind = request.get('type')
            if kind == 'reset':
                ctrl.reset()
                response = {'ok': True}
            elif kind == 'act':
                response = {'action': ctrl.act(int(request['t']), request['obs'])}
            elif kind == 'close':
                response = {'ok': True}
                print(json.dumps(response), flush=True)
                return
            else:
                response = {'error': 'unknown request'}
        except Exception as exc:
            print('controller error:', repr(exc), file=sys.stderr, flush=True)
            response = {'action': [0, 0, 0, 0, 0, 0, -1]}
        print(json.dumps(response, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
