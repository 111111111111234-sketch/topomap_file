"""Smoke runner for clean Agent1 (conftopo/agents/goat_agent_1.py).

Per-step trace follows the clean ConfTopo pipeline:
perception / memory / structure_intents / navigation_proposals / stop_verifier.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAV_REACH_RADIUS = 0.40

from conftopo.agents.goat_agent_1 import ConfTopoGOATAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.navigation.pathfinder_executor import PathfinderExecutor
from conftopo.perception import (
    GoatModalityClipEncoder,
    encode_agent_image_goal_embed,
    encode_agent_rgb_embed,
)
from run_goat_minimal import (
    angular_diff,
    apply_episode_rotation,
    find_scene_file,
    load_json_gz,
    make_sim,
    pick_episode,
    quat_to_heading,
    rgb_to_embedding,
)
from run_goat_topo_trace import snapshot_topo

_INF = float("inf")
_PAPER_SUCCESS_DISTANCE = 1.0


def _memory_stats(agent: ConfTopoGOATAgent) -> dict[str, Any]:
    topo = agent.topo_map
    
    # 结构层节点（语义知识）
    structure_nodes = topo.structure_layer.nodes()
    structure_rooms = [n for n in structure_nodes if n.node_type == NodeType.ROOM]
    structure_objects = [n for n in structure_nodes if n.node_type == NodeType.OBJECT]
    structure_landmarks = [n for n in structure_nodes if n.node_type == NodeType.LANDMARK]
    structure_portals = [n for n in structure_nodes if n.node_type == NodeType.PORTAL]
    
    # 导航层节点（可执行路径）
    navigation_nodes = topo.navigation_layer.nodes()
    visited_wp = [n for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_VISITED]
    frontier_wp = [n for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_FRONTIER]
    candidate_wp = [n for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_CANDIDATE]
    approach_wp = [n for n in navigation_nodes if n.node_type == NodeType.WAYPOINT_APPROACH]
    
    # 跨层链接
    cross_layer_links = list(topo.cross_layer.edges())
    
    # 计数值
    num_visited = len(visited_wp)
    num_frontiers = len(frontier_wp)
    num_candidates = len(candidate_wp)
    num_rooms = len(structure_rooms)
    num_objects = len(structure_objects)
    num_landmarks = len(structure_landmarks)
    num_hypotheses = len(agent.memory.hypothesis_pool.get_active(goal_id=None)) if hasattr(agent, 'memory') and hasattr(agent.memory, 'hypothesis_pool') else 0
    num_object_memory = len(topo.object_memory.get_all())

    return {
        # 向后兼容字段（顶级）- 用于可视化脚本
        "total_nodes": len(topo._nodes),
        "visited_waypoints": num_visited,
        "frontiers": num_frontiers,
        "candidate_waypoints": num_candidates,
        "objects": num_objects,
        "rooms": num_rooms,
        "landmarks": num_landmarks,
        "hypotheses": num_hypotheses,
        "object_memory": num_object_memory,
        
        # 新的分层结构
        "structure_layer": {
            "total": len(structure_nodes),
            "rooms": num_rooms,
            "objects": num_objects,
            "landmarks": num_landmarks,
            "portals": len(structure_portals),
        },
        "navigation_layer": {
            "total": len(navigation_nodes),
            "visited_waypoints": num_visited,
            "frontiers": num_frontiers,
            "candidate_waypoints": num_candidates,
            "approach_waypoints": len(approach_wp),
        },
        "cross_layer_links": len(cross_layer_links),
        "failed_targets": list(topo.failed_target_history[-8:]),
    }


def _snapshot_rgb(obs: dict[str, Any]) -> np.ndarray | None:
    rgb = obs.get("color_sensor")
    if rgb is None:
        return None
    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.array(arr, copy=True)


def controller_step(sim, target: np.ndarray, origin: np.ndarray | None = None) -> str:
    if not hasattr(controller_step, "_last_action"):
        controller_step._last_action = None
        controller_step._last_pos = None
        controller_step._flip_streak = 0
        controller_step._pathfinder = PathfinderExecutor(
            reach_radius=NAV_REACH_RADIUS,
            waypoint_radius=NAV_REACH_RADIUS,
            turn_threshold=0.35,
        )

    if origin is not None:
        try:
            action, _debug = controller_step._pathfinder.step(sim, target, origin)
            controller_step._last_action = action
            state = sim.get_agent(0).get_state()
            pos = np.asarray(state.position, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
            controller_step._last_pos = pos.copy()
            return action
        except Exception:
            pass

    state = sim.get_agent(0).get_state()
    pos = np.array(state.position, dtype=np.float32)
    if origin is not None:
        pos -= np.array(origin, dtype=np.float32)
    delta = target - pos
    if np.linalg.norm(delta) < NAV_REACH_RADIUS:
        controller_step._flip_streak = 0
        controller_step._last_action = "target_reached"
        controller_step._last_pos = pos.copy()
        return "target_reached"
    target_heading = math.atan2(-float(delta[0]), -float(delta[2]))
    q = state.rotation
    diff = angular_diff(
        target_heading,
        quat_to_heading([q.real, q.imag[0], q.imag[1], q.imag[2]]),
    )
    # Keep a wider forward band than one discrete turn step to avoid left/right oscillation.
    turn_threshold = 0.35
    if diff > turn_threshold:
        action = "turn_left"
    elif diff < -turn_threshold:
        action = "turn_right"
    else:
        action = "move_forward"

    moved = 1.0
    if controller_step._last_pos is not None:
        last_pos = np.asarray(controller_step._last_pos, dtype=np.float32)
        moved = float(np.linalg.norm(pos - last_pos))
    delta_norm = float(np.linalg.norm(delta))
    is_turn = action in {"turn_left", "turn_right"}
    last_is_turn = controller_step._last_action in {"turn_left", "turn_right"}
    alternating_turn = (
        is_turn
        and last_is_turn
        and action != controller_step._last_action
        and moved < 0.03
        and delta_norm > 0.6
    )
    if alternating_turn:
        controller_step._flip_streak += 1
    else:
        controller_step._flip_streak = 0 if not is_turn else max(0, controller_step._flip_streak - 1)

    # Break repeated left-right loops by forcing a forward step.
    if controller_step._flip_streak >= 2:
        action = "move_forward"
        controller_step._flip_streak = 0

    controller_step._last_action = action
    controller_step._last_pos = pos.copy()
    return action


def _scene_basename(episode: dict[str, Any]) -> str:
    scene_id = str(episode.get("scene_id", ""))
    return scene_id.split("/")[-1].replace(".basis.glb", "").replace(".glb", "")


def _scene_goal_key_prefixes(episode: dict[str, Any]) -> list[str]:
    scene_file = str(episode.get("scene_id", "")).split("/")[-1]
    candidates = [scene_file]
    scene_base = _scene_basename(episode)
    if scene_base and scene_base not in candidates:
        candidates.append(scene_base)
    return candidates


def _goal_entries_for_key(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    key_suffix: str | None,
) -> list[dict[str, Any]]:
    if not key_suffix:
        return []
    for prefix in _scene_goal_key_prefixes(episode):
        entries = dataset_goals.get(f"{prefix}_{key_suffix}")
        if isinstance(entries, list):
            return [g for g in entries if isinstance(g, dict)]
    return []


def _resolve_goal_entries(
    dataset_goals: dict[str, Any],
    episode: dict[str, Any],
    task_index: int,
) -> list[dict[str, Any]]:
    tasks = episode.get("tasks") or []
    if task_index >= len(tasks) or len(tasks[task_index]) < 2:
        return []
    goal_category, goal_type, *rest = tasks[task_index]
    goal_inst_id = rest[0] if rest else None

    goal_entries = _goal_entries_for_key(dataset_goals, episode, str(goal_category))
    if not goal_entries:
        return []

    for child_cat in goal_entries[0].get("children_object_categories") or []:
        goal_entries.extend(_goal_entries_for_key(dataset_goals, episode, str(child_cat)))

    if goal_type == "object":
        return [g for g in goal_entries if isinstance(g, dict)]
    if goal_inst_id is None:
        return []

    instance_key_entries = _goal_entries_for_key(dataset_goals, episode, str(goal_inst_id))
    if instance_key_entries:
        return instance_key_entries
    return [
        g
        for g in goal_entries
        if isinstance(g, dict) and str(g.get("object_id")) == str(goal_inst_id)
    ]


def _normalize_episode_task_goal_type(goal_type: str) -> str:
    mapping = {
        "object": "category",
        "description": "description",
        "image": "image",
    }
    return mapping.get(str(goal_type).strip().lower(), str(goal_type).strip().lower())


def _normalize_graph_goal_type(goal_type: str) -> str:
    value = str(goal_type).strip().lower()
    if value == "object":
        return "category"
    return value


def _assert_goal_graph_alignment(
    episode: dict[str, Any],
    goals: list[GoalNode],
    *,
    max_goals: int,
) -> None:
    tasks = episode.get("tasks") or []
    expected_tasks = tasks[:max_goals]
    if len(goals) < len(expected_tasks):
        raise RuntimeError(
            "GoalGraph shorter than episode tasks: "
            f"goal_nodes={len(goals)} expected_at_least={len(expected_tasks)}"
        )

    mismatches: list[str] = []
    for idx, task in enumerate(expected_tasks):
        if len(task) < 2:
            mismatches.append(f"idx={idx} malformed_task={task}")
            continue
        expected_target = str(task[0]).strip().lower()
        expected_type = _normalize_episode_task_goal_type(str(task[1]))
        actual_goal = goals[idx]
        actual_target = str(actual_goal.target_object).strip().lower()
        actual_type = _normalize_graph_goal_type(actual_goal.goal_type)
        if expected_target != actual_target or expected_type != actual_type:
            mismatches.append(
                f"idx={idx} episode=({expected_target},{expected_type}) "
                f"goal_graph=({actual_target},{actual_type})"
            )

    if mismatches:
        raise RuntimeError(
            "GoalGraph / episode task order mismatch: " + "; ".join(mismatches[:6])
        )


def _instance_positions(goal_entries: list[dict[str, Any]]) -> list[np.ndarray]:
    seen: set[tuple] = set()
    positions: list[np.ndarray] = []
    for g in goal_entries:
        pos_raw = g.get("position")
        if pos_raw is None:
            continue
        pos = np.asarray(pos_raw, dtype=np.float32)
        key = tuple(np.round(pos, 4).tolist())
        if key not in seen:
            seen.add(key)
            positions.append(pos)
    return positions


def _view_points_from_goal_entries(goal_entries: list[dict[str, Any]]) -> list[np.ndarray]:
    seen: set[tuple] = set()
    viewpoints: list[np.ndarray] = []
    for goal in goal_entries:
        for view_point in goal.get("view_points") or []:
            if not isinstance(view_point, dict):
                continue
            agent_state = view_point.get("agent_state") or {}
            position = agent_state.get("position")
            if position is None:
                continue
            pos = np.asarray(position, dtype=np.float32)
            key = tuple(np.round(pos, 4).tolist())
            if key in seen:
                continue
            seen.add(key)
            viewpoints.append(pos)
    return viewpoints


def _same_floor_positions(
    positions: list[np.ndarray],
    *,
    floor_y: float,
    tolerance: float,
) -> list[np.ndarray]:
    return [
        np.asarray(pos, dtype=np.float32)
        for pos in positions
        if abs(float(np.asarray(pos, dtype=np.float32)[1]) - float(floor_y)) <= float(tolerance)
    ]


def _planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm((aa - bb)[[0, 2]]))


def _relative_position(world_pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    rel = np.asarray(world_pos, dtype=np.float32) - np.asarray(origin, dtype=np.float32)
    return rel


def _min_distance(agent_pos: np.ndarray, positions: list[np.ndarray]) -> float:
    if not positions:
        return _INF
    agent = np.asarray(agent_pos, dtype=np.float32)
    return min(float(np.linalg.norm(agent - np.asarray(p, dtype=np.float32))) for p in positions)


def _geodesic_distance_to_viewpoints(
    sim,
    agent_world: np.ndarray,
    viewpoints: list[np.ndarray],
    episode: Any = None,
) -> float:
    if not viewpoints:
        return _INF
    try:
        agent_world_np = np.asarray(agent_world, dtype=np.float32)
        viewpoints_np = [np.asarray(v, dtype=np.float32) for v in viewpoints]
        if hasattr(sim, "geodesic_distance"):
            geo = float(sim.geodesic_distance(agent_world_np, viewpoints_np, episode))
            return geo if np.isfinite(geo) else _INF

        from habitat_sim.nav import MultiGoalShortestPath

        path = MultiGoalShortestPath()
        path.requested_start = agent_world_np
        path.requested_ends = np.asarray(viewpoints_np, dtype=np.float32)
        if not sim.pathfinder.find_path(path):
            return _INF
        geo = float(path.geodesic_distance)
        return geo if np.isfinite(geo) else _INF
    except Exception as exc:
        print(f"[eval] geodesic distance failed: {exc}", file=sys.stderr)
        return _INF


def _goal_summary(goal: GoalNode) -> dict[str, Any]:
    return {
        "target_object": goal.target_object,
        "goal_type": goal.goal_type,
        "attributes": goal.attributes,
        "room_prior": goal.room_prior,
        "landmarks": goal.landmarks or [],
        "has_target_embedding": goal.target_embedding is not None,
        "target_embedding_dim": None
        if goal.target_embedding is None
        else int(goal.target_embedding.shape[-1]),
    }


def _object_anchors(
    agent: ConfTopoGOATAgent,
    *,
    label_filter: str | None = None,
) -> list[dict[str, Any]]:
    result = []
    label_key = None if label_filter is None else str(label_filter).lower().strip()
    for node in agent.topo_map._nodes.values():
        if node.node_type != NodeType.OBJECT:
            continue
        if label_key is not None and node.label.lower().strip() != label_key:
            continue
        attrs = node.attributes
        role = str(attrs.get("semantic_role", "")).strip()
        anchor_waypoint_id = attrs.get("anchor_waypoint_id") or attrs.get("best_anchor_waypoint_id")
        is_anchored_object = (
            role == "object_anchor"
            or (
                role in {"object", "context_object"}
                and bool(anchor_waypoint_id or attrs.get("approach_point_id"))
            )
        )
        if not is_anchored_object:
            continue
        result.append(
            {
                "node_id": node.node_id,
                "label": node.label,
                "source": attrs.get("source"),
                "confidence": float(node.confidence),
                "semantic_role": role,
                "anchor_waypoint_id": anchor_waypoint_id,
                "best_anchor_waypoint_id": attrs.get("best_anchor_waypoint_id"),
                "seen_count": int(attrs.get("seen_count", 0) or 0),
                "last_seen_step": attrs.get("last_seen_step"),
            }
        )
    return result


def _step_record(
    *,
    global_step: int,
    goal_index: int,
    goal_name: str,
    local_step: int,
    out: dict[str, Any],
    low_action: str,
    position: np.ndarray,
    world_position: np.ndarray,
    distance_to_target: float | None,
    goal_min_viewpoint_geodesic: float | None,
    memory: dict[str, Any],
    rgb_frame: str | None = None,
    topo_snapshot: dict[str, Any] | None = None,
    heading: float | None = None,
) -> dict[str, Any]:
    debug = dict(out.get("debug") or {})
    state = str(debug.get("state", ""))
    perception = dict(debug.get("perception") or {})
    selected_intent = dict(debug.get("selected_intent") or {})
    selected_target = dict(debug.get("selected_navigation_target") or {})
    map_audit = dict(debug.get("map_audit") or {})
    target_node_id = out.get("target_node_id") or selected_target.get("node_id")
    target_pos = out.get("target_position")
    if target_pos is None:
        target_pos = selected_target.get("target_position")

    rec: dict[str, Any] = {
        "step": global_step,
        "global_step": global_step,
        "goal_index": goal_index,
        "goal": goal_name,
        "local_step": local_step,
        "agent_action": out.get("action"),
        "low_action": low_action,
        "state": state,
        "mode": "verified_stop" if state == "STOP" or out.get("action") == "stop" else state.lower(),
        "nav_phase": state,
        "phase_reason": debug.get("transition_reason"),
        "state_steps": debug.get("state_steps"),
        "target_node_id": target_node_id,
        "target_type": out.get("target_type") or selected_target.get("target_type"),
        "target_position": (
            None
            if target_pos is None
            else np.asarray(target_pos, dtype=np.float32).round(4).tolist()
        ),
        "position": position.round(4).tolist(),
        "world_position": world_position.round(4).tolist(),
        "distance_to_target": None
        if distance_to_target is None
        else round(float(distance_to_target), 4),
        "distance_to_target_metric": "viewpoint_geodesic",
        "goal_min_viewpoint_geodesic": None
        if goal_min_viewpoint_geodesic is None
        else round(float(goal_min_viewpoint_geodesic), 4),
        "goal_min_distance": None
        if goal_min_viewpoint_geodesic is None
        else round(float(goal_min_viewpoint_geodesic), 4),
        "action": {
            "high_level": out.get("action"),
            "low_level": low_action,
            "reason": out.get("reason"),
            "target_node_id": target_node_id,
            "target_type": out.get("target_type") or selected_target.get("target_type"),
        },
        "perception": perception,
        "memory": memory,
        "structure_intents": list(debug.get("structure_intents") or []),
        "selected_intent": selected_intent,
        "navigation_proposals": list(debug.get("navigation_proposals") or []),
        "selected_navigation_target": selected_target,
        "stop_verifier": dict(debug.get("stop_verifier") or {}),
        "trajectory": list(map_audit.get("trajectory") or []),
        "map_context": {
            "edges": list(map_audit.get("map_edges") or []),
            "current_options": list(map_audit.get("current_options") or []),
        },
        "supplementary": list(map_audit.get("supplementary_candidates") or []),
        "recent_failures": list(map_audit.get("recent_failures") or []),
        "scan_context": dict(out.get("scan_context") or debug.get("scan_context") or {}),
        "frontier_rejected": list(debug.get("frontier_rejected") or []),
        "reachability_probe": dict(debug.get("reachability_probe") or {}),
        "topo_snapshot": topo_snapshot,
        "debug": debug,
        "proposal_count": len(debug.get("navigation_proposals") or []),
        "proposal_type": selected_intent.get("intent_type"),
        "proposal_source": "structure_planner",
        "proposal_score": selected_target.get("score"),
        "proposal_structure_node_id": selected_target.get("structure_node_id"),
        "verify_hits": debug.get("verify_hits"),
        "verify_attempts": debug.get("verify_attempts"),
        "track_hits": debug.get("track_hits"),
        "target_lost_steps": debug.get("target_lost_steps"),
        "active_anchor_id": debug.get("active_object_node_id"),
        "vlm_fresh": perception.get("fresh_vlm"),
        "vlm_trigger_reason": perception.get("vlm_reason"),
        "vlm_goal_visible": perception.get("vlm_goal_visible"),
        "verified_stop": bool(state == "STOP" or out.get("action") == "stop"),
    }

    if rgb_frame is not None:
        rec["rgb_frame"] = rgb_frame
    if topo_snapshot is not None:
        rec["topo"] = topo_snapshot
    if heading is not None:
        rec["heading"] = round(float(heading), 4)
        rec["world_heading"] = round(float(heading), 4)
    return rec


def _summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    steps = trace.get("steps", [])
    summaries = trace.get("task_summaries", [])

    phases = {s.get("nav_phase") for s in steps if s.get("nav_phase")}
    modes = {s.get("mode") for s in steps if s.get("mode")}
    
    # 双层架构统计
    navigation_layer_changes = 0
    cross_layer_activations = 0
    structure_layer_proposals = 0
    
    for step in steps:
        topo = step.get("topo_snapshot") or step.get("topo") or {}
        layers_summary = topo.get("layers_summary", {})
        if layers_summary:
            navigation_layer_changes += 1

        intent = step.get("selected_intent") or {}
        intent_type = intent.get("intent_type")
        if intent_type and intent_type != "explore":
            cross_layer_activations += 1
            structure_layer_proposals += 1

    active_topo_stats = _summarize_active_topo(steps)

    return {
        "total_steps": len(steps),
        "vlm_fresh_steps": sum(1 for s in steps if s.get("vlm_fresh")),
        "vlm_goal_visible_steps": sum(1 for s in steps if s.get("vlm_goal_visible")),
        "vlm_trigger_reasons": sorted(
            {s.get("vlm_trigger_reason") for s in steps if s.get("vlm_trigger_reason")}
        ),
        "nav_phases": sorted(phases),
        "modes": sorted(modes),
        "saw_approach": "CONFIRM" in phases,
        "saw_verify": "CONFIRM" in phases,
        "saw_stop": "STOP" in phases or any(s.get("verified_stop") for s in steps),
        "goals_success": sum(1 for t in summaries if t.get("goal_success")),
        "goals_repo_success": sum(1 for t in summaries if t.get("repo_success")),
        "goals_paper_success": sum(1 for t in summaries if t.get("paper_success")),
        # 双层架构工作统计
        "dual_layer_stats": {
            "cross_layer_activations": cross_layer_activations,
            "navigation_layer_updates": navigation_layer_changes,
            "structure_layer_proposals": structure_layer_proposals,
        },
        # ActiveTopo (Phase B-G) observability roll-up.
        "active_topo_stats": active_topo_stats,
    }


def _summarize_active_topo(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-step ActiveTopo debug (view / verify / recovery)."""
    max_folded = max_suppressed = max_verify_nodes = 0
    max_recovery = 0
    needs_verify_steps = 0
    verify_success = verify_fail = 0
    active_ratios: list[float] = []
    seen = False
    for step in steps:
        debug = step.get("debug") or {}
        av = debug.get("active_topo")
        if av:
            seen = True
            dist = av.get("activity_distribution", {})
            max_folded = max(max_folded, int(dist.get("folded", 0)))
            max_suppressed = max(max_suppressed, int(dist.get("suppressed", 0)))
            max_verify_nodes = max(max_verify_nodes, int(dist.get("verify", 0)))
            persistent = int(av.get("persistent_node_count", 0) or 0)
            if persistent > 0:
                active_ratios.append(int(av.get("active_node_count", 0) or 0) / persistent)
        ve = debug.get("verify_events")
        if ve:
            needs_verify_steps += 1 if ve.get("needs_verify_now") else 0
            verify_success = max(verify_success, int(ve.get("success_count", 0) or 0))
            verify_fail = max(verify_fail, int(ve.get("fail_count", 0) or 0))
        rec = debug.get("recovery")
        if rec:
            max_recovery = max(max_recovery, int(rec.get("level", 0) or 0))
    if not seen:
        return {"present": False}
    avg_active_ratio = round(sum(active_ratios) / len(active_ratios), 4) if active_ratios else None
    return {
        "present": True,
        "max_folded_nodes": max_folded,
        "max_suppressed_nodes": max_suppressed,
        "max_verify_nodes": max_verify_nodes,
        "avg_active_ratio": avg_active_ratio,
        "verify_success_total": verify_success,
        "verify_fail_total": verify_fail,
        "needs_verify_steps": needs_verify_steps,
        "max_recovery_level": max_recovery,
    }


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke for refactored goat_agent_1.py")
    p.add_argument("--split", default="val_seen")
    p.add_argument("--scene", default="4ok3usBNeis")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--max-goals", type=int, default=3)
    p.add_argument("--steps-per-goal", type=int, default=120)
    p.add_argument("--dataset-dir", default="data/datasets/goat_bench/hm3d/v1")
    p.add_argument("--scene-root", default="data/scene_datasets/hm3d_val/val")
    p.add_argument("--goal-graph-dir", default="data/goal_graphs/goat")
    p.add_argument(
        "--output",
        default="data/logs/goat_topo/agent1_smoke_4ok3usBNeis_ep0_v2/trace.json",
    )
    p.add_argument(
        "--perception-backend", default="vlm", choices=["vlm", "clip_groundingdino"]
    )
    p.add_argument("--vlm-api-base", default="http://localhost:8000/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--vlm-timeout", type=float, default=30.0)
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Pin the regression baseline: temperature=0 greedy VLM decoding, "
             "fixed VLM/global seeds. Overrides --vlm-temperature.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed applied to python/numpy/torch and the VLM server "
             "when --deterministic is set.",
    )
    p.add_argument(
        "--vlm-temperature",
        type=float,
        default=None,
        help="Override VLM sampling temperature (default: backend default 0.1). "
             "Ignored when --deterministic is set (forced to 0.0).",
    )
    p.add_argument("--clip-model", default="ViT-B/32")
    p.add_argument("--clip-image-model", default="RN50")
    p.add_argument("--clip-device", default="cuda")
    p.add_argument("--use-placeholder-embed", action="store_true")
    p.add_argument("--success-distance", type=float, default=0.25,
                   help="Official GOAT eval threshold: geodesic distance to nearest goal viewpoint.")
    p.add_argument(
        "--same-floor-y-tolerance",
        type=float,
        default=2.0,
        help="Used only for visualization and diagnostic grouping of object centers; does not affect official viewpoint-based success evaluation.",
    )
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--frame-dir", default=None)
    p.add_argument("--viz", action="store_true",
                   help="Run visualize_goat_topo_trace.py after trace is saved.")
    p.add_argument("--viz-stride", type=int, default=2)
    p.add_argument("--viz-fps", "--fps", type=int, default=6, dest="viz_fps")
    p.add_argument("--viz-out-dir", default=None,
                   help="Visualization output directory (default: <output_parent>/viz).")
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    """Seed all host-side RNGs so non-VLM randomness is reproducible."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def main() -> None:
    args = _build_args()

    scene_path, episode = pick_episode(
        ROOT / args.dataset_dir, args.split, args.scene, args.episode_index
    )
    dataset = load_json_gz(scene_path)
    scene_file = find_scene_file(episode["scene_id"], ROOT / args.scene_root)

    from run_goat_topo_trace import load_goal_graph

    ig = load_goal_graph(
        ROOT / args.goal_graph_dir, args.split, scene_path, episode["episode_id"]
    )
    goals = [g for g in ig.goal_nodes if isinstance(g, GoalNode)][: args.max_goals]
    if not goals:
        raise RuntimeError("No goals found in goal graph")
    _assert_goal_graph_alignment(episode, goals, max_goals=args.max_goals)

    config = ConfTopoConfig()
    config.perception.backend = args.perception_backend
    config.perception.vlm_api_base = args.vlm_api_base
    config.perception.vlm_model = args.vlm_model
    config.perception.vlm_timeout = args.vlm_timeout
    config.perception.clip_device = args.clip_device
    config.perception.heavy_enabled = args.perception_backend == "vlm"

    if args.deterministic:
        _seed_everything(args.seed)
        config.perception.vlm_temperature = 0.0
        config.perception.vlm_seed = args.seed
        print(
            f"[deterministic] temperature=0.0 seed={args.seed} "
            "(greedy VLM decoding; host RNGs seeded)"
        )
    elif args.vlm_temperature is not None:
        config.perception.vlm_temperature = args.vlm_temperature

    agent = ConfTopoGOATAgent(config)

    encoder = None
    if not args.use_placeholder_embed:
        encoder = GoatModalityClipEncoder(
            args.clip_model, args.clip_image_model, args.clip_device
        )
        if hasattr(agent, "set_clip_encoder"):
            agent.set_clip_encoder(encoder)

    sim = make_sim(scene_file)
    sim_agent = sim.initialize_agent(0)
    init_state = sim_agent.get_state()
    init_state.position = np.array(episode["start_position"], dtype=np.float32)
    apply_episode_rotation(init_state, episode)
    sim_agent.set_state(init_state)
    origin = np.array(init_state.position, dtype=np.float32)
    if hasattr(agent, "set_navigation_context"):
        agent.set_navigation_context(sim, origin)
    elif hasattr(agent, "set_pathfinder"):
        agent.set_pathfinder(sim)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_dir: Path | None = None
    if not args.no_save_frames:
        frame_dir = Path(args.frame_dir) if args.frame_dir else output.parent / "frames"
        if not frame_dir.is_absolute():
            frame_dir = ROOT / frame_dir
        frame_dir.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "agent": "goat_agent_1.ConfTopoGOATAgent",
        "smoke_type": "agent1_v2",
        "split": args.split,
        "scene": args.scene,
        "episode_index": args.episode_index,
        "scene_file": str(scene_file),
        "episode_file": str(scene_path),
        "episode_id": episode["episode_id"],
        "origin_world": origin.round(4).tolist(),
        "coordinate_frame": "episode_start_relative",
        "topo_coordinate_frame": "episode_start_relative",
        "success_metric": "official_goat_eval_viewpoint_geodesic",
        "goal_sequence_validated": True,
        "success_distance": float(args.success_distance),
        "paper_success_distance": float(_PAPER_SUCCESS_DISTANCE),
        "same_floor_y_tolerance": float(args.same_floor_y_tolerance),
        "same_floor_y_tolerance_usage": "visualization_and_diagnostics_only",
        "pose_sources": {
            "position": "episode_start_relative_xz_grounded",
            "world_position": "habitat_gt_world_position",
            "heading": "habitat_gt_heading",
        },
        "perception_backend": args.perception_backend,
        "deterministic": bool(args.deterministic),
        "vlm_temperature": float(config.perception.vlm_temperature),
        "vlm_seed": config.perception.vlm_seed,
        "max_goals": args.max_goals,
        "steps_per_goal": args.steps_per_goal,
        "goals": [g.target_object for g in goals],
        "current_goal": _goal_summary(goals[0]),
        "steps": [],
        "task_summaries": [],
        "goal_instances": [],
    }

    global_step = 0
    try:
        for goal_index, goal in enumerate(goals):
            agent.set_new_goal(goal)
            trace["current_goal"] = _goal_summary(goal)

            goal_entries = _resolve_goal_entries(dataset.get("goals", {}), episode, goal_index)
            instance_positions = _instance_positions(goal_entries)
            goal_view_points = _view_points_from_goal_entries(goal_entries)
            same_floor_instance_positions = _same_floor_positions(
                instance_positions,
                floor_y=float(origin[1]),
                tolerance=float(args.same_floor_y_tolerance),
            )
            off_floor_instance_positions = [
                pos for pos in instance_positions if not any(np.allclose(pos, keep) for keep in same_floor_instance_positions)
            ]
            goal_object_ids = [
                str(g.get("object_id")) for g in goal_entries if g.get("object_id")
            ]
            goal_min_dist = _INF
            goal_stop_dist = _INF
            goal_min_object_dist = _INF
            goal_stop_object_dist = _INF
            goal_stopped = False
            local_step = 0
            goal_trace_start = len(trace["steps"])
            progress_target_id: str | None = None
            progress_best_distance = _INF
            progress_stale_steps = 0
            reached_target_counts: dict[tuple[str, str, str], int] = {}

            for local_step in range(args.steps_per_goal):
                obs = sim.get_sensor_observations()
                state = sim_agent.get_state()
                world_pos = np.asarray(state.position, dtype=np.float32)
                rel_pos = _relative_position(world_pos, origin)
                rgb = _snapshot_rgb(obs)
                if rgb is not None and int(rgb.max()) == 0:
                    obs = sim.get_sensor_observations()
                    rgb = _snapshot_rgb(obs)

                rgb_embed = encode_agent_rgb_embed(
                    encoder,
                    rgb,
                    agent,
                    use_placeholder=args.use_placeholder_embed,
                    placeholder_fn=rgb_to_embedding,
                )
                image_goal_embed = (
                    None
                    if args.use_placeholder_embed or rgb is None
                    else encode_agent_image_goal_embed(encoder, rgb, agent)
                )
                conf_obs = {
                    "rgb": rgb,
                    "rgb_embed": rgb_embed,
                    "image_goal_embed": image_goal_embed,
                    "position": rel_pos.copy(),
                    "heading": quat_to_heading(
                        [state.rotation.real, *state.rotation.imag]
                    ),
                }

                out = agent.step(conf_obs)
                action = out.get("action")
                target = out.get("target_position")
                debug = out.get("debug") or {}
                phase = str(debug.get("state", ""))

                if action in ("move_forward", "turn_left", "turn_right"):
                    low_action = action
                elif action == "hold_and_verify":
                    low_action = "hold"
                elif action == "stop" or phase == "STOP":
                    low_action = "stop"
                    goal_stopped = True
                elif action == "navigate" and target is not None:
                    low_action = controller_step(sim, np.asarray(target), origin=origin)
                else:
                    low_action = "turn_left"

                target_node_id = out.get("target_node_id")
                if low_action == "unreachable" and hasattr(agent, "on_navigation_event"):
                    agent.on_navigation_event(target_node_id, "unreachable")
                    progress_target_id = None
                    progress_best_distance = _INF
                    progress_stale_steps = 0
                elif low_action == "target_reached" and hasattr(agent, "on_navigation_event"):
                    agent.on_navigation_event(target_node_id, "target_reached")
                    selected_target = debug.get("selected_navigation_target") or {}
                    target_type = str(out.get("target_type") or selected_target.get("target_type") or "")
                    structure_key = str(
                        selected_target.get("structure_node_id")
                        or "|".join(str(x) for x in selected_target.get("evidence_refs", [])[:2])
                        or "none"
                    )
                    reach_key = (str(target_node_id), target_type, structure_key)
                    reached_target_counts[reach_key] = reached_target_counts.get(reach_key, 0) + 1
                    if (
                        reached_target_counts[reach_key] >= 4
                        and target_type not in {"frontier", "candidate", "waypoint_frontier"}
                    ):
                        agent.on_navigation_event(target_node_id, "navigation_failed")
                        reached_target_counts[reach_key] = 0
                    progress_target_id = None
                    progress_best_distance = _INF
                    progress_stale_steps = 0
                elif action == "navigate" and target is not None and target_node_id:
                    nav_dist = float(np.linalg.norm(np.asarray(target, dtype=np.float32) - rel_pos))
                    if progress_target_id != str(target_node_id):
                        progress_target_id = str(target_node_id)
                        progress_best_distance = nav_dist
                        progress_stale_steps = 0
                    elif nav_dist < progress_best_distance - 0.08:
                        progress_best_distance = nav_dist
                        progress_stale_steps = 0
                    else:
                        progress_stale_steps += 1
                    if progress_stale_steps >= 12:
                        progress_target_id = None
                        progress_best_distance = _INF
                        progress_stale_steps = 0

                dist = None
                if goal_view_points:
                    geo = _geodesic_distance_to_viewpoints(sim, world_pos, goal_view_points)
                    if np.isfinite(geo):
                        dist = geo
                if dist is not None and dist < goal_min_dist:
                    goal_min_dist = dist
                object_dist = None
                if instance_positions:
                    obj = _min_distance(world_pos, instance_positions)
                    if np.isfinite(obj):
                        object_dist = obj
                if object_dist is not None and object_dist < goal_min_object_dist:
                    goal_min_object_dist = object_dist
                if low_action == "stop" and dist is not None:
                    goal_stop_dist = dist
                if low_action == "stop" and object_dist is not None:
                    goal_stop_object_dist = object_dist

                frame_rel = None
                if frame_dir is not None and rgb is not None:
                    frame_path = frame_dir / f"rgb_{global_step:04d}.png"
                    imageio.imwrite(frame_path, rgb)
                    frame_rel = str(frame_path.relative_to(ROOT))

                topo_snapshot = snapshot_topo(agent)

                trace["steps"].append(
                    _step_record(
                        global_step=global_step,
                        goal_index=goal_index,
                        goal_name=str(goal.target_object),
                        local_step=local_step,
                        out=out,
                        low_action=low_action,
                        position=rel_pos,
                        world_position=world_pos,
                        distance_to_target=dist,
                        goal_min_viewpoint_geodesic=None if goal_min_dist == _INF else goal_min_dist,
                        memory=_memory_stats(agent),
                        rgb_frame=frame_rel,
                        topo_snapshot=topo_snapshot,
                        heading=float(conf_obs["heading"]),
                    )
                )
                global_step += 1

                if goal_stopped or low_action == "stop":
                    break
                if low_action not in {"target_reached", "unreachable", "hold"}:
                    sim.step(low_action)

            goal_steps = trace["steps"][goal_trace_start:]
            selected_intent_counts: dict[str, int] = {}
            for step_row in goal_steps:
                intent_type = (step_row.get("selected_intent") or {}).get("intent_type")
                if intent_type:
                    selected_intent_counts[str(intent_type)] = selected_intent_counts.get(str(intent_type), 0) + 1
            stop_rows = [s for s in goal_steps if s.get("verified_stop")]
            object_matches = [
                node
                for node in agent.topo_map.object_memory.get_all()
                if str(node.label).lower().strip() == str(goal.target_object).lower().strip()
            ]
            repo_success = bool(
                goal_stopped
                and goal_stop_dist != _INF
                and goal_stop_dist < args.success_distance
            )
            paper_success = bool(
                goal_stopped
                and goal_stop_object_dist != _INF
                and goal_stop_object_dist < _PAPER_SUCCESS_DISTANCE
            )
            trace["task_summaries"].append(
                {
                    "goal_index": goal_index,
                    "target_object": goal.target_object,
                    "target": goal.target_object,
                    "steps": local_step + 1,
                    "goal_stopped": goal_stopped,
                    "goal_stop_viewpoint_geodesic": None
                    if goal_stop_dist == _INF
                    else round(float(goal_stop_dist), 4),
                    "goal_min_viewpoint_geodesic": None
                    if goal_min_dist == _INF
                    else round(float(goal_min_dist), 4),
                    "goal_stop_object_euclidean": None
                    if goal_stop_object_dist == _INF
                    else round(float(goal_stop_object_dist), 4),
                    "goal_min_object_euclidean": None
                    if goal_min_object_dist == _INF
                    else round(float(goal_min_object_dist), 4),
                    "goal_stop_distance": None
                    if goal_stop_dist == _INF
                    else round(float(goal_stop_dist), 4),
                    "goal_min_distance": None
                    if goal_min_dist == _INF
                    else round(float(goal_min_dist), 4),
                    "repo_success": repo_success,
                    "paper_success": paper_success,
                    "goal_success": repo_success,
                    "goal_instance_count": len(instance_positions),
                    "same_floor_goal_instance_count": len(same_floor_instance_positions),
                    "off_floor_goal_instance_count": len(off_floor_instance_positions),
                    "goal_object_ids": goal_object_ids,
                    "goal_view_point_count": len(goal_view_points),
                    "repo_success_distance": float(args.success_distance),
                    "paper_success_distance": float(_PAPER_SUCCESS_DISTANCE),
                    "object_memory_matches": len(object_matches),
                    "selected_intent_counts": selected_intent_counts,
                    "frontier_selected_count": selected_intent_counts.get("explore", 0),
                    "verify_count": sum(1 for s in goal_steps if s.get("nav_phase") == "CONFIRM"),
                    "stop_reason": (
                        ((stop_rows[-1].get("stop_verifier") or {}).get("reason") if stop_rows else None)
                        or ("verified_stop" if goal_stopped else None)
                    ),
                    "goal_instance_positions_world": [
                        np.asarray(p, dtype=np.float32).round(4).tolist() for p in instance_positions
                    ],
                    "goal_instance_positions_world_same_floor": [
                        np.asarray(p, dtype=np.float32).round(4).tolist()
                        for p in same_floor_instance_positions
                    ],
                    "goal_instance_positions_world_off_floor": [
                        np.asarray(p, dtype=np.float32).round(4).tolist()
                        for p in off_floor_instance_positions
                    ],
                    "goal_instance_positions": [
                        _relative_position(p, origin).round(4).tolist()
                        for p in same_floor_instance_positions
                    ],
                    "object_anchors": _object_anchors(agent, label_filter=goal.target_object),
                }
            )
            trace["goal_instances"].append(
                {
                    "goal_index": goal_index,
                    "target_object": goal.target_object,
                    "positions_world": [
                        np.asarray(p, dtype=np.float32).round(4).tolist() for p in instance_positions
                    ],
                    "positions_world_same_floor": [
                        np.asarray(p, dtype=np.float32).round(4).tolist()
                        for p in same_floor_instance_positions
                    ],
                    "positions_world_off_floor": [
                        np.asarray(p, dtype=np.float32).round(4).tolist()
                        for p in off_floor_instance_positions
                    ],
                    "positions_relative": [
                        _relative_position(p, origin).round(4).tolist()
                        for p in same_floor_instance_positions
                    ],
                    "view_points_world": [
                        np.asarray(p, dtype=np.float32).round(4).tolist() for p in goal_view_points
                    ],
                    "view_points": [
                        _relative_position(p, origin).round(4).tolist() for p in goal_view_points
                    ],
                }
            )
    finally:
        sim.close()

    trace["final_memory"] = _memory_stats(agent)
    trace["final_object_anchors"] = _object_anchors(agent)
    trace["final_summary"] = {"episode_length": len(trace["steps"])}
    trace["smoke_summary"] = _summarize_trace(trace)

    output.write_text(json.dumps(trace, indent=2))

    viz_dir: Path | None = None
    if args.viz:
        viz_dir = Path(args.viz_out_dir) if args.viz_out_dir else output.parent / "viz"
        if not viz_dir.is_absolute():
            viz_dir = ROOT / viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/visualize_goat_topo_trace.py",
                "--trace",
                str(output.relative_to(ROOT)),
                "--out-dir",
                str(viz_dir.relative_to(ROOT)),
                "--stride",
                str(args.viz_stride),
                "--fps",
                str(args.viz_fps),
            ],
            cwd=ROOT,
            check=True,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "frame_dir": None if frame_dir is None else str(frame_dir),
                "viz_dir": None if viz_dir is None else str(viz_dir),
                "smoke_summary": trace["smoke_summary"],
                "goals": trace["goals"],
                "task_summaries": [
                    {
                        "target": t["target_object"],
                        "steps": t["steps"],
                        "goal_stopped": t["goal_stopped"],
                        "goal_stop_viewpoint_geodesic": t.get("goal_stop_viewpoint_geodesic"),
                        "goal_min_viewpoint_geodesic": t.get("goal_min_viewpoint_geodesic"),
                        "goal_stop_object_euclidean": t.get("goal_stop_object_euclidean"),
                        "repo_success": t.get("repo_success"),
                        "paper_success": t.get("paper_success"),
                        "object_anchors": len(t["object_anchors"]),
                    }
                    for t in trace["task_summaries"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
