"""Deterministic Cartesian replay for RGB-D batch-picking plans."""
import math
import random

TOOLS = {"gear": "parallel_jaw", "plate": "vacuum", "coupler": "three_finger"}
GRASPS = {"gear": "rim", "plate": "top", "coupler": "body"}
TOOL_RADIUS = {"parallel_jaw": 0.022, "vacuum": 0.018, "three_finger": 0.024}
WORKSPACE_LO = (-0.225, -0.165, 0.0)
WORKSPACE_HI = (0.310, 0.165, 0.260)
POSITION_TOLERANCE = 0.007
YAW_TOLERANCE = 8.0
HANDLE_TOLERANCE = 0.012
MAX_SEGMENT = 0.26


def finite_vector(value, length=3):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError("expected a {}-vector".format(length))
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError("vectors must contain finite numbers")
        out.append(float(item))
    return out


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def yaw_error(a, b):
    return abs(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def point_in_expanded_box(point, box, radius, extra=0.0):
    lo, hi = box["min_m"], box["max_m"]
    return all(lo[i] - radius - extra <= point[i] <= hi[i] + radius + extra for i in range(3))


def point_hits_part(point, part, radius, extra=0.0):
    px, py, _ = part["position_m"]
    radial = part["radius_m"] + radius + extra
    return ((point[0] - px) ** 2 + (point[1] - py) ** 2 <= radial ** 2
            and -extra <= point[2] <= part["height_m"] + radius + extra)


def segment_samples(a, b):
    length = distance(a, b)
    steps = max(2, int(math.ceil(length / 0.003)))
    for index in range(steps + 1):
        t = index / float(steps)
        yield [a[j] + t * (b[j] - a[j]) for j in range(3)]


def segment_clear(a, b, obstacles, guards, parts, tool_radius, extra):
    if distance(a, b) > MAX_SEGMENT + 1e-9:
        return False, "segment_too_long"
    for point in segment_samples(a, b):
        if (point[0] < WORKSPACE_LO[0] + tool_radius or point[0] > WORKSPACE_HI[0] - tool_radius
                or point[1] < WORKSPACE_LO[1] + tool_radius or point[1] > WORKSPACE_HI[1] - tool_radius
                or point[2] < WORKSPACE_LO[2] or point[2] > WORKSPACE_HI[2] - tool_radius):
            return False, "workspace_collision"
        if any(point_in_expanded_box(point, box, tool_radius, extra) for box in obstacles):
            return False, "obstacle_collision"
        if any(point_in_expanded_box(point, box, tool_radius, extra) for box in guards):
            return False, "guard_collision"
        if any(point_hits_part(point, part, tool_radius, extra) for part in parts):
            return False, "part_collision"
    return True, "ok"


def required_guards(scene):
    labels = {target["label"] for target in scene["targets"]}
    return {guard["id"]: guard for guard in scene["guards"] if guard["covers"] in labels}


def validate_guard_moves(scene, moves):
    if not isinstance(moves, list):
        return False, "malformed_guard_moves", set()
    expected = required_guards(scene)
    seen = set()
    for move in moves:
        if not isinstance(move, dict) or set(move) != {"guard_id", "handle_position_m", "axis", "distance_m"}:
            return False, "malformed_guard_move", set()
        guard_id = move.get("guard_id")
        if guard_id not in expected or guard_id in seen:
            return False, "unneeded_guard_move", set()
        seen.add(guard_id)
        guard = expected[guard_id]
        try:
            handle = finite_vector(move["handle_position_m"])
            axis = finite_vector(move["axis"])
            amount = float(move["distance_m"])
        except (ValueError, TypeError, OverflowError):
            return False, "bad_guard_pose", set()
        norm = math.sqrt(sum(x * x for x in axis))
        if norm < 1e-9:
            return False, "bad_guard_axis", set()
        axis = [x / norm for x in axis]
        if distance(handle, guard["handle_position_m"]) > HANDLE_TOLERANCE:
            return False, "bad_guard_handle", set()
        if sum(axis[i] * guard["axis"][i] for i in range(3)) < 0.985:
            return False, "bad_guard_axis", set()
        if not math.isfinite(amount) or amount < guard["distance_m"] - 0.012 or amount > guard["distance_m"] + 0.030:
            return False, "bad_guard_distance", set()
    if seen != set(expected):
        return False, "missing_guard_move", set()
    return True, "ok", seen


def perturb_scene(scene, seed):
    rng = random.Random(seed)
    parts = []
    for part in scene["parts"]:
        copy = dict(part)
        copy["position_m"] = [part["position_m"][0] + rng.uniform(-0.0018, 0.0018),
                              part["position_m"][1] + rng.uniform(-0.0018, 0.0018),
                              part["position_m"][2]]
        copy["yaw_deg"] = part["yaw_deg"] + rng.uniform(-1.8, 1.8)
        parts.append(copy)
    return parts, rng.uniform(0.0015, 0.0038)


def replay(scene, plan, perturbation_seed):
    if not isinstance(plan, dict) or set(plan) != {"scene_id", "guard_moves", "picks"}:
        return False, "malformed_scene_plan"
    guard_ok, reason, moved = validate_guard_moves(scene, plan["guard_moves"])
    if not guard_ok:
        return False, reason
    picks = plan.get("picks")
    targets = {target["label"]: target for target in scene["targets"]}
    if not isinstance(picks, list) or len(picks) != len(targets):
        return False, "incomplete_batch"
    labels = [pick.get("label") for pick in picks if isinstance(pick, dict)]
    if len(labels) != len(set(labels)) or set(labels) != set(targets):
        return False, "wrong_parts"

    perturbed, extra = perturb_scene(scene, perturbation_seed)
    parts = {part["label"]: part for part in perturbed}
    remaining = set(parts)
    obstacles = list(scene["obstacles"])
    active_guards = [guard for guard in scene["guards"] if guard["id"] not in moved]

    required_keys = {"label", "tool", "grasp", "grasp_pose", "pregrasp_position_m",
                     "transport_waypoints_m", "bin"}
    for pick in picks:
        if not isinstance(pick, dict) or set(pick) != required_keys:
            return False, "malformed_pick"
        label = pick["label"]
        physical = parts[label]
        if pick.get("tool") != TOOLS[physical["kind"]] or pick.get("grasp") != GRASPS[physical["kind"]]:
            return False, "wrong_grasp"
        if pick.get("bin") != targets[label]["destination"]:
            return False, "wrong_destination"
        pose = pick.get("grasp_pose")
        if not isinstance(pose, dict) or set(pose) != {"position_m", "yaw_deg"}:
            return False, "malformed_grasp_pose"
        try:
            grasp = finite_vector(pose["position_m"])
            pregrasp = finite_vector(pick["pregrasp_position_m"])
            yaw = float(pose["yaw_deg"])
            waypoints = [finite_vector(x) for x in pick["transport_waypoints_m"]]
        except (ValueError, TypeError, OverflowError):
            return False, "nonfinite_plan"
        expected_position = [physical["position_m"][0], physical["position_m"][1], physical["height_m"]]
        if distance(grasp, expected_position) > POSITION_TOLERANCE or yaw_error(yaw, physical["yaw_deg"]) > YAW_TOLERANCE:
            return False, "pose_error"
        dz = pregrasp[2] - grasp[2]
        if math.hypot(pregrasp[0] - grasp[0], pregrasp[1] - grasp[1]) > 0.006 or not 0.065 <= dz <= 0.125:
            return False, "bad_approach"
        if not isinstance(pick["transport_waypoints_m"], list) or not 2 <= len(waypoints) <= 8:
            return False, "bad_waypoints"

        radius = TOOL_RADIUS[pick["tool"]]
        other_parts = [parts[x] for x in remaining if x != label]
        clear, why = segment_clear(pregrasp, grasp, obstacles, active_guards, other_parts, radius, extra)
        if not clear:
            return False, "approach_" + why
        remaining.remove(label)
        route = [grasp] + waypoints
        for a, b in zip(route, route[1:]):
            clear, why = segment_clear(a, b, obstacles, active_guards,
                                       [parts[x] for x in remaining], radius + physical["radius_m"] * .35, extra)
            if not clear:
                return False, "transport_" + why
        destination = scene["bins"][pick["bin"]]
        if distance(waypoints[-1], destination) > 0.018:
            return False, "bad_place_pose"
    return True, "ok"


def evaluate_scene(scene, plan):
    # Five replay variants keep the score hardware-invariant while requiring millimetres of
    # clearance margin.  Seeds are grader-owned and are not present in the agent workspace.
    for index in range(5):
        ok, reason = replay(scene, plan, 990001 + scene["index"] * 101 + index * 17)
        if not ok:
            return False, reason
    return True, "ok"
