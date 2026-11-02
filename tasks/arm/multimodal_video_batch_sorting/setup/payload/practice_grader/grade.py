#!/usr/bin/env python3
"""Public schema/numeric checker; perception and collision truth stay private."""
import argparse
import json
import math
from pathlib import Path


def vec(value):
    return (isinstance(value, list) and len(value) == 3
            and all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x)) for x in value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path.home() / "submission/plan.json")
    args = parser.parse_args()
    value = json.loads(args.plan.read_text())
    errors = []
    rows = value.get("scenes", []) if isinstance(value, dict) else []
    expected = {"scene_{:02d}".format(i) for i in range(16)}
    seen = set()
    if not isinstance(value, dict) or set(value) != {"schema_version", "scenes"} or value.get("schema_version") != 2:
        errors.append("top-level schema")
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or set(row) != {"scene_id", "guard_moves", "picks"}:
            errors.append("malformed scene row")
            continue
        seen.add(row["scene_id"])
        if not isinstance(row["guard_moves"], list) or not isinstance(row["picks"], list) or len(row["picks"]) != 5:
            errors.append(row["scene_id"] + ": batch size")
            continue
        for move in row["guard_moves"]:
            if (not isinstance(move, dict) or set(move) != {"guard_id", "handle_position_m", "axis", "distance_m"}
                    or not vec(move.get("handle_position_m")) or not vec(move.get("axis"))):
                errors.append(row["scene_id"] + ": guard move")
        for pick in row["picks"]:
            keys = {"label", "tool", "grasp", "grasp_pose", "pregrasp_position_m", "transport_waypoints_m", "bin"}
            if not isinstance(pick, dict) or set(pick) != keys:
                errors.append(row["scene_id"] + ": pick fields")
                continue
            pose = pick.get("grasp_pose")
            points = pick.get("transport_waypoints_m")
            if (not isinstance(pose, dict) or set(pose) != {"position_m", "yaw_deg"}
                    or not vec(pose.get("position_m")) or not vec(pick.get("pregrasp_position_m"))
                    or not isinstance(points, list) or not 2 <= len(points) <= 8 or not all(vec(x) for x in points)):
                errors.append(row["scene_id"] + ": numeric mechanics")
    if seen != expected:
        errors.append("missing or duplicate scenes")
    print(json.dumps({"schema_mechanics_ok": not errors, "errors": errors[:30],
                      "note": "Pose, guard and swept-volume truth remain private."}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
