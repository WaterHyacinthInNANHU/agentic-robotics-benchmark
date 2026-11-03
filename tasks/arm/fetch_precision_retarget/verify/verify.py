"""Hidden two-waypoint, 3 mm evaluator for native FetchPush-v4 physics."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_push_env import PushEpisode
from robotics_grader import (
    Solver, SolverAborted, SolverMissing, ratio_rewards, read_anchor,
    seed_rng, stage_config, write_verdict, zero_verdict,
)


def parse_action(value):
    if not isinstance(value, list) or len(value) != 4:
        raise SolverAborted("expected four action values")
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverAborted("invalid action: {}".format(exc))
    if not all(math.isfinite(v) for v in result):
        raise SolverAborted("non-finite action")
    return result


def run_episodes(config):
    count = int(config.get("episodes", 40))
    seeds = seed_rng()
    solver = Solver(step_timeout=float(config.get("step_timeout_s", 10)))
    wins = stage1_wins = stage2_wins = aborted = 0
    first_errors = []
    second_errors = []
    try:
        for episode in range(count):
            world = PushEpisode(seeds.seed(bits=64))
            try:
                solver.reset()
                obs = world.observe()
                for t in range(world.env.spec.max_episode_steps):
                    action = parse_action(solver.act(t, obs))
                    obs = world.step(action)
                result = world.result()
                wins += int(result["success"])
                stage1_wins += int(result["stage1_success"])
                stage2_wins += int(result["stage2_success"])
                first_errors.append(result["stage1_error_m"])
                second_errors.append(result["stage2_error_m"])
            except (SolverAborted, ValueError) as exc:
                aborted += 1
                print("[grader] episode {} aborted: {}".format(episode, exc),
                      file=sys.stderr, flush=True)
            finally:
                world.close()
    finally:
        solver.close()
    return {
        "success_rate": wins / count,
        "complete_successes": wins,
        "stage1_success_rate": stage1_wins / count,
        "stage2_success_rate": stage2_wins / count,
        "mean_stage1_error_m": sum(first_errors) / len(first_errors) if first_errors else 0.0,
        "mean_stage2_error_m": sum(second_errors) / len(second_errors) if second_errors else 0.0,
        "episodes_aborted": aborted,
    }


def main():
    anchor, metadata = read_anchor()
    try:
        result = run_episodes(stage_config())
    except (SolverMissing, SolverAborted) as exc:
        write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None))
        return
    measured = result.pop("success_rate")
    write_verdict(ratio_rewards(
        measured, anchor, cap=stage_config().get("scoring_cap", 1.5),
        higher_is_better=True, metric_name="success_rate",
        extra=result, full_at=metadata.get("full_at", 1.0),
    ))


if __name__ == "__main__":
    main()
