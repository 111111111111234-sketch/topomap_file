#!/usr/bin/env python3
"""Analyze closed-loop GOAT smoke trace with Level-4 validation metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional


def _dist3(a: List[float], b: List[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_node_index(step: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    nodes = _safe_get(step, ["topo", "nodes"], [])
    out: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id:
            out[node_id] = node
    return out


def analyze(trace: Dict[str, Any]) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = trace.get("steps", [])
    if not steps:
        return {"error": "trace has no steps"}

    total_steps = len(steps)
    first = steps[0]
    last = steps[-1]

    map_total = []
    map_rooms = []
    map_objects = []
    map_frontiers = []
    map_visited = []
    map_cross_links = []

    intent_counter: Counter[str] = Counter()
    target_type_counter: Counter[str] = Counter()

    frontier_selected_count = 0
    blocked_frontier_still_selected = 0

    prev_target_id: Optional[str] = None
    same_target_repeat_steps = 0
    target_switch_count = 0
    run_len = 0
    max_same_target_streak = 0

    prev_pos = _safe_get(first, ["position"], [0.0, 0.0, 0.0])
    first_pos = prev_pos
    last_pos = prev_pos
    position_change_count = 0
    total_movement = 0.0
    collision_like_count = 0

    prev_stuck_recoveries = 0
    recovery_events: List[Dict[str, Any]] = []
    recovery_step_set = set()
    last_failed_target = None

    for idx, step in enumerate(steps):
        mem = step.get("memory", {}) if isinstance(step.get("memory"), dict) else {}
        map_total.append(int(mem.get("total_nodes", 0) or 0))
        map_rooms.append(int(mem.get("rooms", 0) or 0))
        map_objects.append(int(mem.get("objects", 0) or 0))
        map_frontiers.append(int(mem.get("frontiers", 0) or 0))
        map_visited.append(int(mem.get("visited_waypoints", 0) or 0))
        map_cross_links.append(int(mem.get("cross_layer_links", 0) or 0))

        intent = str(_safe_get(step, ["sticky_debug", "selected_intent", "intent_type"], "none"))
        intent_counter[intent] += 1

        target_type = str(step.get("target_type", "none"))
        target_type_counter[target_type] += 1
        if target_type == "frontier":
            frontier_selected_count += 1

        target_id_raw = step.get("target_node_id")
        target_id = str(target_id_raw) if target_id_raw is not None else None
        if idx > 0:
            if target_id == prev_target_id and target_id is not None:
                same_target_repeat_steps += 1
                run_len += 1
            else:
                if prev_target_id is not None and target_id is not None and target_id != prev_target_id:
                    target_switch_count += 1
                run_len = 1 if target_id is not None else 0
            if run_len > max_same_target_streak:
                max_same_target_streak = run_len
        else:
            run_len = 1 if target_id is not None else 0
            max_same_target_streak = run_len
        prev_target_id = target_id

        if target_type == "frontier" and target_id is not None and idx not in recovery_step_set:
            node_index = _build_node_index(step)
            node = node_index.get(target_id)
            attrs = (node or {}).get("attributes", {}) if isinstance((node or {}).get("attributes", {}), dict) else {}
            if bool(attrs.get("blocked", False)) or bool(attrs.get("consumed", False)):
                blocked_frontier_still_selected += 1

        pos = _safe_get(step, ["position"], prev_pos)
        step_move = _dist3(pos, prev_pos)
        total_movement += step_move
        if step_move > 1e-3:
            position_change_count += 1

        # A simple closed-loop proxy: navigate/forward command but almost no movement.
        low_action = str(step.get("low_action", ""))
        if low_action in {"move_forward", "navigate"} and step_move < 0.01:
            collision_like_count += 1

        prev_pos = pos
        last_pos = pos

        stuck_recoveries = int(_safe_get(step, ["sticky_debug", "stuck_recoveries"], 0) or 0)
        if stuck_recoveries > prev_stuck_recoveries:
            recovery_events.append(
                {
                    "idx": idx,
                    "step": int(step.get("step", idx)),
                    "target_at_recovery": str(step.get("target_node_id")) if step.get("target_node_id") is not None else None,
                }
            )
            recovery_step_set.add(idx)
        prev_stuck_recoveries = stuck_recoveries

        candidate_failed = _safe_get(step, ["sticky_debug", "last_failed_target"], None)
        if candidate_failed is not None:
            last_failed_target = candidate_failed

    intent_object_count = intent_counter.get("object", 0)
    intent_room_count = intent_counter.get("room", 0)
    intent_explore_count = intent_counter.get("explore", 0)

    stuck_recovery_target_switch_count = 0
    for event in recovery_events:
        idx = int(event["idx"])
        recovery_target = event.get("target_at_recovery")
        if recovery_target is None:
            continue
        if idx + 1 >= total_steps:
            continue
        next_target_raw = steps[idx + 1].get("target_node_id")
        next_target = str(next_target_raw) if next_target_raw is not None else None
        if next_target is not None and next_target != recovery_target:
            stuck_recovery_target_switch_count += 1

    # Reasonableness checks from the Level-4 checklist.
    alerts = {
        "always_room_intent": intent_room_count == total_steps,
        "same_target_100_steps": max_same_target_streak >= 100,
        "frontier_selected_too_low": frontier_selected_count < max(1, int(0.1 * total_steps)),
        "blocked_frontier_still_selected": blocked_frontier_still_selected > 0,
        "position_almost_static": total_movement < 0.1,
        "visited_waypoints_too_low": int(map_visited[-1]) <= 2,
        "stuck_recoveries_without_target_switch": prev_stuck_recoveries > 0 and stuck_recovery_target_switch_count == 0,
    }

    result = {
        "trace_meta": {
            "scene": trace.get("scene"),
            "episode_index": trace.get("episode_index"),
            "steps": total_steps,
            "goals": trace.get("goals", []),
            "goals_success": _safe_get(trace, ["smoke_summary", "goals_success"], None),
        },
        "map_metrics": {
            "structure_node_count": int(_safe_get(last, ["topo", "layers_summary", "structure", "nodes"], map_rooms[-1] + map_objects[-1])),
            "navigation_node_count": int(_safe_get(last, ["topo", "layers_summary", "navigation", "nodes"], map_visited[-1] + map_frontiers[-1])),
            "room_node_count": int(map_rooms[-1]),
            "object_node_count": int(map_objects[-1]),
            "frontier_node_count": int(map_frontiers[-1]),
            "cross_layer_edge_count": int(map_cross_links[-1]),
            "trend": {
                "visited_waypoints_start_end": [int(map_visited[0]), int(map_visited[-1])],
                "frontier_start_end": [int(map_frontiers[0]), int(map_frontiers[-1])],
                "room_start_end": [int(map_rooms[0]), int(map_rooms[-1])],
                "object_start_end": [int(map_objects[0]), int(map_objects[-1])],
                "cross_layer_start_end": [int(map_cross_links[0]), int(map_cross_links[-1])],
                "object_peak": int(max(map_objects)),
            },
        },
        "planner_metrics": {
            "selected_intent_distribution": dict(intent_counter),
            "target_type_distribution": dict(target_type_counter),
            "target_node_id_switch_count": int(target_switch_count),
            "same_target_repeat_steps": int(same_target_repeat_steps),
            "max_same_target_streak": int(max_same_target_streak),
            "frontier_selected_count": int(frontier_selected_count),
            "object_intent_count": int(intent_object_count),
            "room_intent_count": int(intent_room_count),
            "explore_intent_count": int(intent_explore_count),
            "blocked_frontier_still_selected_count": int(blocked_frontier_still_selected),
        },
        "behavior_metrics": {
            "position_change_count": int(position_change_count),
            "total_movement": round(float(total_movement), 4),
            "net_displacement": round(float(_dist3(first_pos, last_pos)), 4),
            "visited_waypoints": int(map_visited[-1]),
            "collision_like_count": int(collision_like_count),
            "stuck_recoveries": int(prev_stuck_recoveries),
            "stuck_recovery_target_switch_count": int(stuck_recovery_target_switch_count),
            "stuck_recovery_event_count": int(len(recovery_events)),
            "last_failed_target": last_failed_target,
        },
        "alerts": alerts,
        "alert_count": int(sum(1 for v in alerts.values() if v)),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze GOAT smoke closed-loop metrics")
    parser.add_argument("--trace", required=True, help="Path to trace.json")
    parser.add_argument("--output", default="", help="Optional path to save JSON report")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON")
    args = parser.parse_args()

    with open(args.trace, "r", encoding="utf-8") as f:
        trace = json.load(f)

    report = analyze(trace)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=True, indent=2)

    if args.compact:
        print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
    else:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
