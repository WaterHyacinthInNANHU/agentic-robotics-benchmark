# Clear a complete tray from multimodal video evidence

Plan a complete robot sorting mission for each of 24 synthetic inspection scenes. Each
scene contains three RGB frames, three aligned depth frames, a natural-language work
order and an inspection CSV. Seven stamped parts move slightly between frames; the work
order selects six of them. Correctly sort **every selected part**. A scene succeeds only
when the whole requested batch is placed and verified safely.

The public evidence is under `/home/user/scene_XX/`. Images are commissioning diagrams,
not photographs. Parts are gold circular gears (`parallel_jaw`, `rim`), blue rounded
plates (`vacuum`, `top`), or red keyed couplers (`three_finger`, `body`). The natural
two-letter stamp establishes identity across frames and inspection rows. A white
triangular notch gives yaw: right/up/left/down are 0/90/180/270 degrees. Use frames 0
and 2 when the moving guard occludes frame 1. In aligned depth, a selected footprint
above intensity 180 requires `clear_guard`; below 150 is clear.

Read each work order's actual `peak_force_n` inequality and threshold. Process every
inspection row satisfying it. Its `vibration_rms` goes to `reject` when at least the
scene threshold, otherwise `ready`.

Write only `/home/user/submission/plan.json` with one row per scene. Actions are:

```json
{"op":"clear_guard","label":"AX","tool":"parallel_jaw"}
{"op":"pick","label":"AX","tool":"vacuum","grasp":"top","yaw_deg":90}
{"op":"place","label":"AX","bin":"reject","tool":"vacuum"}
{"op":"verify","label":"AX","bin":"reject"}
```

At most 28 actions are accepted per scene. Selected parts may be handled in any order.
Clearing a nonexistent guard, selecting the distractor, wrong tool/grasp/yaw/bin,
collision with an uncleared guard, reusing a part, or failing to place and verify all
six selected parts fails the complete scene. Equivalent safe orders are accepted.

Missing, malformed, oversized, non-finite or executable artifacts score zero. The
public checker reports schema/mechanics only and does not expose perception ground
truth. No network or GPU is needed.
