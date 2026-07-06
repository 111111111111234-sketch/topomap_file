#!/usr/bin/env python3
"""Offline planner probe for GOAT dual-layer planning.

This script freezes a single trace step (map snapshot + pose + goal),
reconstructs planner inputs, and runs:
1) StructurePlanner: choose semantic intent (where to search)
2) NavigationPlanner: resolve to executable target (how to move)

It is designed to validate the dual-layer invariant offline, without running
the closed-loop agent in simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from conftopo.agents.goat_agent_1 import (
    AgentPose,
    AgentState,
    MemoryManager,
    NavigationPlanner,
    ObjectMemoryEntry,
    SearchIntent,
    StructurePlanner,
)
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType
from conftopo.core.instruction_graph import GoalNode


NAV_NODE_TYPES = {
    NodeType.WAYPOINT_VISITED,
    NodeType.WAYPOINT_FRONTIER,
    NodeType.WAYPOINT_CANDIDATE,
    NodeType.WAYPOINT_APPROACH,
}


def _parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_vec3(value: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
        except Exception:
            pass
    if default is not None:
        return default.astype(np.float32)
    return np.zeros(3, dtype=np.float32)


def _intent_debug(intent: SearchIntent) -> Dict[str, Any]:
    return {
        "intent_type": intent.intent_type,
        "target_label": intent.target_label,
        "structure_node_id": intent.structure_node_id,
        "preferred_anchor_waypoint_id": intent.preferred_anchor_waypoint_id,
        "preferred_room": intent.preferred_room,
        "score": round(float(intent.score), 4),
        "reason": intent.reason,
        "evidence_refs": list(intent.evidence_refs),
    }


def _load_trace(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_step(trace: Dict[str, Any], step_idx: int) -> Dict[str, Any]:
    steps = trace.get("steps") or []
    if not steps:
        raise ValueError("trace has no steps")
    if step_idx < 0:
        return steps[-1]
    if step_idx >= len(steps):
        raise ValueError(f"step index out of range: {step_idx} >= {len(steps)}")
    return steps[step_idx]


def _restore_topo_from_snapshot(step_data: Dict[str, Any]) -> DynamicTopoMap:
    config = ConfTopoConfig()
    topo = DynamicTopoMap(config.memory)
    topo._current_step = _safe_int(step_data.get("step"), 0)

    topo_snapshot = step_data.get("topo") or {}
    nodes = topo_snapshot.get("nodes") or []
    edges = topo_snapshot.get("edges") or []

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        node_type_str = str(node.get("type", "")).strip()
        if not node_id or not node_type_str:
            continue
        try:
            node_type = NodeType(node_type_str)
        except ValueError:
            continue

        topo.add_node(
            node_type=node_type,
            position=_to_vec3(node.get("position")),
            confidence=_safe_float(node.get("confidence"), 0.5),
            label=str(node.get("label", "") or ""),
            node_id=node_id,
            attributes=dict(node.get("attributes") or {}),
        )
        restored = topo.get_node(node_id)
        if restored is not None:
            restored.step_id = _safe_int(node.get("step_id"), topo.current_step)
            restored.visit_count = _safe_int(node.get("visit_count"), restored.visit_count)

    for edge in edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        if not topo.has_node(source) or not topo.has_node(target):
            continue

        edge_type_str = str(edge.get("type", "")).strip()
        if edge_type_str in {item.value for item in EdgeType}:
            topo.add_edge(
                source,
                target,
                EdgeType(edge_type_str),
                weight=_safe_float(edge.get("weight"), 1.0),
                relation=str(edge.get("relation", "") or ""),
                confidence=_safe_float(edge.get("confidence"), 0.5),
            )
            if topo.graph.has_edge(source, target):
                restored_edge = topo.graph.edges[source, target]
                for key, value in edge.items():
                    if key in {"source", "target"}:
                        continue
                    restored_edge[key] = value
        else:
            topo.graph.add_edge(source, target, **dict(edge))

    return topo


def _hydrate_object_memory_pool(memory: MemoryManager) -> None:
    topo = memory.topo_map
    pool = memory.object_memory_pool

    for obj_node in topo.get_nodes_by_type(NodeType.OBJECT):
        attrs = dict(obj_node.attributes or {})

        anchor_id = str(
            attrs.get("best_anchor_waypoint_id")
            or attrs.get("anchor_waypoint_id")
            or ""
        ).strip()
        if not anchor_id:
            anchor = topo.navigation_target_for_structure_node(obj_node.node_id)
            anchor_id = anchor.node_id if anchor is not None else ""
        anchor_node = topo.get_node(anchor_id) if anchor_id else None

        anchor_pos = (
            anchor_node.position.copy()
            if anchor_node is not None
            else _to_vec3(attrs.get("anchor_waypoint_position"), default=obj_node.position)
        )
        est_pos = _to_vec3(attrs.get("estimated_object_position"), default=obj_node.position)

        observed_from_history = attrs.get("observed_from_history") or []
        observed_from: List[np.ndarray] = []
        for item in observed_from_history:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                observed_from.append(_to_vec3(item))
        if not observed_from:
            observed_from.append(anchor_pos.copy())

        observed_from_wp_ids = [
            str(item)
            for item in (attrs.get("observed_from_waypoint_ids") or [])
            if str(item).strip()
        ]
        if anchor_id and anchor_id not in observed_from_wp_ids:
            observed_from_wp_ids.append(anchor_id)

        bbox_history = attrs.get("bbox_history") or []
        if not isinstance(bbox_history, list):
            bbox_history = [attrs.get("bbox")] if attrs.get("bbox") is not None else []

        range_bin_history = attrs.get("range_bin_history") or []
        if not isinstance(range_bin_history, list):
            range_bin_history = [str(attrs.get("range_bin", "unknown"))]

        bearing_history = attrs.get("bearing_history") or []
        if not isinstance(bearing_history, list):
            bearing = str(attrs.get("bearing", "")).strip()
            bearing_history = [bearing] if bearing else []

        entry = ObjectMemoryEntry(
            object_id=obj_node.node_id,
            node_id=obj_node.node_id,
            label=str(obj_node.label or ""),
            confidence=_safe_float(obj_node.confidence, 0.0),
            attributes=attrs,
            bbox_history=list(bbox_history),
            anchor_waypoint_id=anchor_id or obj_node.node_id,
            anchor_waypoint_position=anchor_pos.copy(),
            estimated_object_position=est_pos.copy(),
            observed_from=[vec.copy() for vec in observed_from],
            first_seen_step=_safe_int(attrs.get("first_seen_step"), _safe_int(obj_node.step_id, 0)),
            last_seen_step=_safe_int(attrs.get("last_seen_step"), _safe_int(obj_node.step_id, 0)),
            observed_from_waypoint_ids=list(observed_from_wp_ids),
            best_anchor_waypoint_id=(anchor_id or None),
            range_bin_history=[str(item) for item in range_bin_history],
            bearing_history=[str(item) for item in bearing_history if str(item).strip()],
            room_probs=dict(attrs.get("room_probs") or {}),
            position_uncertainty=_safe_float(attrs.get("position_uncertainty"), 1.0),
            seen_count=max(1, _safe_int(attrs.get("seen_count"), 1)),
            failed_count=max(0, _safe_int(attrs.get("failed_count"), 0)),
            unreachable_count=max(0, _safe_int(attrs.get("unreachable_count"), 0)),
        )
        pool._entries[obj_node.node_id] = entry


def _build_goal(
    *,
    trace: Dict[str, Any],
    step_data: Dict[str, Any],
    goal_label: Optional[str],
    room_prior: Optional[List[str]],
    landmarks: Optional[List[str]],
) -> GoalNode:
    label = (goal_label or step_data.get("goal") or trace.get("current_goal") or "").strip()
    if not label:
        raise ValueError("goal label is empty; pass --goal-label explicitly")
    return GoalNode(
        target_object=label,
        room_prior=list(room_prior or []),
        landmarks=list(landmarks or []),
    )


def _run_once(
    *,
    memory: MemoryManager,
    goal: GoalNode,
    pose: AgentPose,
) -> Dict[str, Any]:
    structure = StructurePlanner()
    navigation = NavigationPlanner()
    state = AgentState()

    intents = structure.build_search_intents(memory=memory, goal=goal, pose=pose)
    selected_intent = intents[0] if intents else structure._explore_intent(memory)
    nav_target = navigation.select_navigation_target(
        intent=selected_intent,
        memory=memory,
        pose=pose,
        state=state,
    )

    nav_node = memory.topo_map.get_node(nav_target.node_id) if nav_target.node_id else None
    structure_node_id = selected_intent.structure_node_id
    separation_ok = not (
        structure_node_id is not None
        and nav_target.node_id is not None
        and structure_node_id == nav_target.node_id
    )
    nav_layer_ok = (
        nav_node is not None and nav_node.node_type in NAV_NODE_TYPES
    ) if nav_target.node_id is not None else True
    not_structure_target_type = nav_target.target_type not in {
        "object",
        "room",
        "landmark",
        "portal",
    }

    result = {
        "structure_planner": {
            "selected_intent": _intent_debug(selected_intent),
            "top_intents": [_intent_debug(intent) for intent in intents[:6]],
        },
        "navigation_planner": {
            "selected_target": {
                "target_node_id": nav_target.node_id,
                "target_type": nav_target.target_type,
                "target_position": (
                    nav_target.position.tolist()
                    if isinstance(nav_target.position, np.ndarray)
                    else nav_target.position
                ),
                "score": round(float(nav_target.score), 4),
                "reason": nav_target.reason,
                "structure_node_id": nav_target.structure_node_id,
                "evidence_refs": list(nav_target.evidence_refs),
                "target_node_type": nav_node.node_type.value if nav_node is not None else None,
                "target_node_layer": (
                    (nav_node.attributes or {}).get("layer") if nav_node is not None else None
                ),
            },
            "top_proposals": list(navigation.last_debug.get("top_proposals", [])),
            "proposal_count": _safe_int(navigation.last_debug.get("proposal_count"), 0),
        },
        "checks": {
            "structure_nav_id_separated": {
                "pass": separation_ok,
                "structure_node_id": structure_node_id,
                "target_node_id": nav_target.node_id,
            },
            "navigation_target_is_navigation_layer": {
                "pass": nav_layer_ok,
                "target_node_id": nav_target.node_id,
                "target_node_type": nav_node.node_type.value if nav_node is not None else None,
            },
            "navigation_target_not_structure_type": {
                "pass": not_structure_target_type,
                "target_type": nav_target.target_type,
            },
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline dual-layer planner validator")
    parser.add_argument("--trace", required=True, help="Path to trace.json")
    parser.add_argument(
        "--step-index",
        type=int,
        default=-1,
        help="Step index to freeze (default: -1, use last step)",
    )
    parser.add_argument("--goal-label", default=None, help="Override goal label")
    parser.add_argument(
        "--goal-room-prior",
        default="",
        help="Comma-separated room prior labels (optional)",
    )
    parser.add_argument(
        "--goal-landmarks",
        default="",
        help="Comma-separated landmark labels (optional)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat planner calls on the same frozen snapshot (default: 1)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any core check fails",
    )
    args = parser.parse_args()

    trace = _load_trace(args.trace)
    step_data = _pick_step(trace, args.step_index)

    topo = _restore_topo_from_snapshot(step_data)
    config = ConfTopoConfig()
    memory = MemoryManager(topo, config)
    memory.topo_map._current_step = _safe_int(step_data.get("step"), topo.current_step)
    _hydrate_object_memory_pool(memory)

    goal = _build_goal(
        trace=trace,
        step_data=step_data,
        goal_label=args.goal_label,
        room_prior=_parse_csv(args.goal_room_prior),
        landmarks=_parse_csv(args.goal_landmarks),
    )
    pose = AgentPose(
        position=_to_vec3(step_data.get("position")),
        heading=_safe_float(step_data.get("heading"), 0.0),
    )

    runs: List[Dict[str, Any]] = []
    repeat = max(1, args.repeat)
    for _ in range(repeat):
        runs.append(_run_once(memory=memory, goal=goal, pose=pose))

    first = runs[0]
    stable = True
    if len(runs) > 1:
        baseline = json.dumps(first, sort_keys=True)
        for run in runs[1:]:
            if json.dumps(run, sort_keys=True) != baseline:
                stable = False
                break

    output = {
        "offline_planner_probe": {
            "trace": args.trace,
            "step_index": _safe_int(step_data.get("step"), args.step_index),
            "goal": {
                "target_object": goal.target_object,
                "room_prior": list(goal.room_prior),
                "landmarks": list(goal.landmarks),
            },
            "pose": {
                "position": pose.position.tolist(),
                "heading": round(float(pose.heading), 4),
            },
            "snapshot_stats": {
                "total_nodes": len(step_data.get("topo", {}).get("nodes", []) or []),
                "total_edges": len(step_data.get("topo", {}).get("edges", []) or []),
                "object_pool_entries": len(memory.object_memory_pool.entries()),
            },
            "repeat": repeat,
            "deterministic": stable,
            "result": first,
        }
    }

    print(json.dumps(output, ensure_ascii=True, indent=2))

    if args.strict:
        checks = first.get("checks", {})
        all_ok = all(bool(item.get("pass", False)) for item in checks.values())
        if not all_ok:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
