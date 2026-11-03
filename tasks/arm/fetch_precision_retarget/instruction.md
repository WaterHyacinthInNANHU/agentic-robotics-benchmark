# Late-target precision pushing with a Fetch robot

Write a closed-loop controller for the stock Gymnasium Robotics `FetchPush-v4`
MuJoCo environment. In one continuous physical scene, push the cube to two
targets. You see target 1 at reset. Target 2 is revealed only after action 200;
the cube and robot are **not** reset. You have 200 actions for each target.
Both final positions must be within **3 mm** of the target active at the end
of its stage. The action rate is 25 Hz, so the full scene lasts 16 seconds
of simulated time. The task uses the simulator's arm motion, contact, friction,
and action limits.

The simulator stack is pinned to Gymnasium Robotics 1.4.2 and MuJoCo 3.3.7.
The complete public wrapper is `/home/user/fetch_push_env.py`. It provides
the official 25-element kinematic observation, the active goal, stage number,
stage step, and a rendered RGB camera frame. You may inspect the wrapper,
use the simulator locally, and run
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
`obs["goal"]` is the active target xyz; `obs["stage"]` is 0 or 1;
`obs["stage_step"]` is 0–199; and `obs["rgb_png_base64"]` is a 160×160
agent-view PNG. State elements 0:3 give gripper xyz and 3:6 cube xyz, in
metres. Other elements are documented in the simulator's official
documentation. Actions lie in [-1,1] and the final gripper component has
no effect because FetchPush locks the gripper. The verifier clips finite
action values to this range.

Each stage is scored at its final physical state only, after actions 200
and 400 respectively. The scene succeeds only if **both** stages satisfy
the 3 mm tolerance. The second target is sampled from the simulator's goal
distribution and must be at least 10 cm from the first target. A reply must
arrive within 10 seconds; print `READY` within 40 seconds. A malformed
action, crash, or deadline miss fails the scene. Repeated queries with the
same `t` do not advance physics; answer from the supplied observation.
Report diagnostics to stderr only.
