"""Private robust Cartesian replay for the complete RGB-D tray mission."""
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_model import evaluate_scene
from robotics_grader import ratio_rewards, read_anchor, stage_config, write_verdict, zero_verdict
from robotics_grader.stage import agent_home


def load_submission(path, scene_ids):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 524288:
        raise ValueError("plan.json must be a regular file no larger than 512 KiB")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"schema_version", "scenes"} or value.get("schema_version") != 2:
        raise ValueError("expected schema_version 2 and scenes")
    rows = value["scenes"]
    if not isinstance(rows, list) or len(rows) != len(scene_ids):
        raise ValueError("one plan row per scene is required")
    plans = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("scene_id") not in scene_ids or row["scene_id"] in plans:
            raise ValueError("invalid or duplicate scene id")
        plans[row["scene_id"]] = row
    if set(plans) != set(scene_ids):
        raise ValueError("missing scene plan")
    return plans


def run_episodes(config, plans=None):
    scenes = json.loads((Path(__file__).parent / "target.json").read_text())["scenes"]
    scene_ids = [scene["scene_id"] for scene in scenes]
    if plans is None:
        plans = load_submission(Path(agent_home()) / "submission/plan.json", scene_ids)
    successes = 0
    reasons = {}
    for scene in scenes:
        ok, reason = evaluate_scene(scene, plans[scene["scene_id"]])
        successes += int(ok)
        reasons[reason] = reasons.get(reason, 0) + 1
    total = len(scenes)
    result = {"success_rate": successes / total, "complete_successes": float(successes)}
    for reason, count in reasons.items():
        result[reason + "_rate"] = count / total
    return result


def main():
    anchor, metadata = read_anchor()
    try:
        result = run_episodes(stage_config())
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        write_verdict(zero_verdict(reason="missing or invalid plan: " + str(exc), anchored=anchor is not None))
        return
    measurement = result.pop("success_rate")
    write_verdict(ratio_rewards(
        measurement,
        anchor,
        cap=stage_config().get("scoring_cap", 1.5),
        higher_is_better=True,
        metric_name="success_rate",
        extra=result,
        full_at=metadata.get("full_at", 1.0),
    ))


if __name__ == "__main__":
    main()
