# Precise contact push with a Fetch robot

Write a closed-loop controller for the stock Gymnasium Robotics
`FetchPush-v4` MuJoCo environment. Move the robot's gripper behind the cube,
push it across the tabletop, and leave it within **2 cm** of the target at
the end of 80 control actions. The robot runs at 25 Hz, so the horizon is
3.2 seconds of simulated time. The task uses the simulator's real arm,
inverse kinematics, contact, friction, and action limits. It is a precision
variant of FetchPush: native `is_success` uses 5 cm, while this task requires
2 cm and checks the final position.

The simulator stack is pinned to Gymnasium Robotics 1.4.2 and MuJoCo 3.3.7.
The complete public wrapper is `/home/user/fetch_push_env.py`. It provides
the 25-element official kinematic observation, the desired goal, and an RGB
camera frame. You may inspect the wrapper, use the simulator locally, and run
`python3 /home/user/practice_grader/grade.py` on visible practice seeds.
The verifier samples fresh 64-bit seeds after development ends.

Create `/home/user/submission/run.sh`. The verifier starts it with cwd
`/home/user/submission`. Print `READY` once, then exchange one JSON object
per line:

```text
{"type":"reset"}                  -> {"ok":true}
{"type":"act","t":0,"obs":{...}} -> {"action":[dx,dy,dz,grip]}
{"type":"close"}                  -> {"ok":true}
```

`obs["state"]` is the official 25-element observation vector;
`obs["goal"]` is the target xyz; `obs["rgb_png_base64"]` is a 160×160
agent-view PNG. State elements 0:3 give gripper xyz and 3:6 cube xyz, in
metres. Other elements are documented in the simulator's official
documentation. Actions lie in [-1,1] and the final gripper component has
no effect because FetchPush locks the gripper. The verifier clips finite
action values to the simulator's range.

Each scene is scored at the final state only. A reply must arrive within
10 seconds; print `READY` within 40 seconds. A malformed action, crash,
or deadline miss fails the scene. Repeated queries with the same `t`
do not advance physics; answer from the supplied observation. Report
diagnostics to stderr only.
