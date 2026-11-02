# Plan robust RGB-D grasping and collision-free full-tray sorting

Plan a complete robot mission for each of 16 synthetic but metrically calibrated
commissioning scenes under `/home/user/scenes/scene_XX/`. Each scene contains three RGB
views, three aligned 16-bit depth-height views, camera projection matrices, an inspection
CSV and a natural-language work order. Six stamped parts sit among tall machine fixtures;
the work order selects five. One lane around the central fence is blocked. Labelled sliding
service guards cover some selected grasp approaches.

This is a Cartesian manipulation-planning task, not visual question answering. Recover each
selected part's metric XYZ and continuous yaw from the calibrated RGB-D views, retract only
the guards that obstruct selected approaches, choose the correct end effector, and provide a
collision-free pregrasp and transport trajectory to the requested bin. The private replay
checks pose accuracy, vertical approach clearance, guard handle/axis/distance, swept-volume
collisions with fixtures, guards and remaining parts, workspace bounds, segment length and
final placement. It repeats the replay under hidden millimetre-scale pose and clearance
perturbations, so grazing paths fail.

Gold gears use `parallel_jaw` / `rim`; blue plates use `vacuum` / `top`; red keyed couplers
use `three_finger` / `body`. The stamped two-letter ID associates pixels with the inspection
row. A bright orientation arrow on each part gives its continuous yaw in tray coordinates.
Depth PNG values are height above the tray in millimetres; zero is invalid. Each view's
`projection_uv_from_xyz_m` in `calibration.json` maps `[x,y,z,1]` to pixel `[u,v]`.
`task_spec.json` supplies public robot radii, tolerances, bounds and trajectory limits.

Read each work order's actual `peak_force_n` inequality and threshold. Process every row
satisfying it. `vibration_rms` at least the scene threshold goes to `reject`; otherwise it
goes to `ready`.

Write only `/home/user/submission/plan.json`:

```json
{
  "schema_version": 2,
  "scenes": [{
    "scene_id": "scene_00",
    "guard_moves": [{
      "guard_id": "G1",
      "handle_position_m": [-0.12, 0.10, 0.115],
      "axis": [1.0, 0.0, 0.0],
      "distance_m": 0.115
    }],
    "picks": [{
      "label": "AX",
      "tool": "vacuum",
      "grasp": "top",
      "grasp_pose": {"position_m": [-0.08, 0.10, 0.026], "yaw_deg": 37.5},
      "pregrasp_position_m": [-0.08, 0.10, 0.116],
      "transport_waypoints_m": [
        [-0.08, 0.10, 0.116],
        [0.04, -0.12, 0.145],
        [0.195, -0.12, 0.145],
        [0.255, 0.112, 0.145],
        [0.255, 0.112, 0.040]
      ],
      "bin": "ready"
    }]
  }]
}
```

Every scene row must contain exactly `scene_id`, `guard_moves` and `picks`. Each pick must
contain exactly the fields shown. Selected parts may be picked in any safe order. Move every
and only required guard once. Use two to eight waypoints per pick and leave clearance margin.
A scene succeeds only when all five selected parts pass every perturbed physical replay.

Missing, malformed, oversized, non-finite or executable artifacts score zero. The public
checker validates schema and basic numeric mechanics only; it does not reveal scene truth.
No network or GPU is needed. Pillow is available at `/opt/venv/bin/python3` for reading the
RGB and 16-bit depth PNGs.
