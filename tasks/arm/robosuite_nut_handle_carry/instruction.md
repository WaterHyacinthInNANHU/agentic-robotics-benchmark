# Carry a square nut by its handle

Control a Panda arm in robosuite's native `NutAssemblySquare` MuJoCo scene.
Grasp the square nut by its narrow handle, lift it, and carry it above the
scene's square peg while keeping it in the gripper. The scene is continuous;
there is no reset or teleport during an episode. You have **400 actions at
20 Hz**. Contact, friction, gripper motion, and OSC pose limits come from
robosuite 1.5.2 and MuJoCo 3.3.7.

The verifier succeeds when the nut's true center stays within **3 cm** in
XY of the peg, above **1.0 m** in world Z, and the gripper remains within
**8 cm** of the nut's handle for **10 consecutive simulator steps**.
This checks sustained physical carrying. The verifier evaluates the true
MuJoCo pose; the public nut and handle poses have a fixed per-scene offset
of at most 3 mm per coordinate. The public wrapper is
`/home/user/nut_carry_env.py`, and visible practice seeds can be run with
`python3 /home/user/practice_grader/grade.py`.

Create `/home/user/submission/run.sh`. It starts with cwd
`/home/user/submission`; print `READY` once, then exchange JSON lines:

```text
{"type":"reset"}                  -> {"ok":true}
{"type":"act","t":0,"obs":{...}} -> {"action":[x,y,z,rx,ry,rz,grip]}
{"type":"close"}                  -> {"ok":true}
```

`obs` contains `eef_pos`, `eef_quat`, `joint_pos`, `gripper_qpos`,
`nut_pos`, `nut_quat`, `handle_pos`, `peg_pos`, and `rgb_png_base64`.
Coordinates are world-frame metres. The image is a 128×128 PNG from the
native `agentview` camera. Each action component is clipped to [-1,1].
The first six components request OSC pose motion; the last is the gripper
command (`-1` open, `1` closed). Only the current observation may be used;
the hidden seed is never exposed.

The grader scores the fraction of fresh hidden episodes completed within
the 400-step horizon. Missing, malformed, or late replies fail an episode.
Print diagnostics to stderr, reply within 10 seconds per action, and print
`READY` within 40 seconds. A step can be queried twice with the same `t`;
answer each query from the supplied observation. The grader can restart
your process after an aborted episode.
