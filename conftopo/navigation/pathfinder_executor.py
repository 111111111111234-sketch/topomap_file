"""Navmesh-aware low-level executor for Phase 2 GOAT smoke tests.

The executor keeps trace-facing coordinates episode-start-relative while using
Habitat-Sim world coordinates for pathfinder queries.
"""

from __future__ import annotations
from conftopo.core.dynamic_topo_map import EdgeType

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def relative_to_world(position_relative: Sequence[float], origin_world: Sequence[float]) -> np.ndarray:
    return np.asarray(origin_world, dtype=np.float32) + np.asarray(position_relative, dtype=np.float32)


def world_to_relative(position_world: Sequence[float], origin_world: Sequence[float]) -> np.ndarray:
    return np.asarray(position_world, dtype=np.float32) - np.asarray(origin_world, dtype=np.float32)


def _quat_to_heading(q) -> float:
    w, x, y, z = float(q.real), float(q.imag[0]), float(q.imag[1]), float(q.imag[2])
    return float(math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def _angular_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _distance3d(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _tolist(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_tolist(v) for v in value]
    if isinstance(value, tuple):
        return [_tolist(v) for v in value]
    if isinstance(value, dict):
        return {k: _tolist(v) for k, v in value.items()}
    return value


@dataclass
class ReachabilityProbe:
    reachable: bool
    geodesic_distance: float = float("inf")
    path_points_relative: List[List[float]] = field(default_factory=list)
    reason: str = ""
    target_position_relative: Optional[List[float]] = None
    target_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _tolist({
            "reachable": self.reachable,
            "geodesic_distance": self.geodesic_distance,
            "path_points_relative": self.path_points_relative,
            "reason": self.reason,
            "target_position": self.target_position_relative,
            "target_node_id": self.target_node_id,
        })


class PathfinderExecutor:
    """Use Habitat pathfinder to execute relative topo targets safely."""

    def __init__(
        self,
        reach_radius: float = 0.40,
        waypoint_radius: float = 0.40,
        turn_threshold: float = 0.25,
        max_geodesic_distance: float = 80.0,
    ):
        self.reach_radius = reach_radius
        self.waypoint_radius = waypoint_radius
        self.turn_threshold = turn_threshold
        self.max_geodesic_distance = max_geodesic_distance

    def probe(self, sim, target_relative: Sequence[float], origin_world: Sequence[float], target_node_id: Optional[str] = None) -> ReachabilityProbe:
        target_relative_arr = np.asarray(target_relative, dtype=np.float32)
        state = sim.get_agent(0).get_state()
        start_world = np.asarray(state.position, dtype=np.float32)
        origin_world_arr = np.asarray(origin_world, dtype=np.float32)
        target_world = relative_to_world(target_relative_arr, origin_world_arr)
        snapped_target_world = None
        try:
            snapped = np.asarray(sim.pathfinder.snap_point(target_world), dtype=np.float32)
            if snapped.shape == (3,) and np.all(np.isfinite(snapped)):
                snapped_target_world = snapped
                target_world = snapped
                target_relative_arr = world_to_relative(target_world, origin_world_arr)
        except Exception:
            snapped_target_world = None

        if snapped_target_world is None:
            return ReachabilityProbe(
                False,
                reason="snap_failed",
                target_position_relative=target_relative_arr.round(4).tolist(),
                target_node_id=target_node_id,
            )

        try:
            if not sim.pathfinder.is_navigable(snapped_target_world):
                return ReachabilityProbe(
                    False,
                    reason="not_navigable",
                    target_position_relative=target_relative_arr.round(4).tolist(),
                    target_node_id=target_node_id,
                )
        except Exception:
            pass

        start_relative = world_to_relative(start_world, origin_world_arr)

        dist3d = _distance3d(target_relative_arr, start_relative)
        if dist3d <= self.reach_radius:
            return ReachabilityProbe(
                reachable=True,
                geodesic_distance=0.0,
                path_points_relative=[start_relative.round(4).tolist(), target_relative_arr.round(4).tolist()],
                reason="target_reached",
                target_position_relative=target_relative_arr.round(4).tolist(),
                target_node_id=target_node_id,
            )

        try:
            from habitat_sim.nav import ShortestPath
            path = ShortestPath()
            path.requested_start = start_world
            path.requested_end = target_world
            found = bool(sim.pathfinder.find_path(path))
            points_world = [np.asarray(p, dtype=np.float32) for p in getattr(path, "points", [])]
            geo = float(getattr(path, "geodesic_distance", float("inf")))
        except Exception as exc:
            return ReachabilityProbe(False, reason=f"path_error:{type(exc).__name__}", target_position_relative=target_relative_arr.round(4).tolist(), target_node_id=target_node_id)

        if not found or not np.isfinite(geo) or not points_world:
            return ReachabilityProbe(False, reason="unreachable", target_position_relative=target_relative_arr.round(4).tolist(), target_node_id=target_node_id)
        if geo > self.max_geodesic_distance:
            return ReachabilityProbe(False, geodesic_distance=geo, reason="too_far_geodesic", target_position_relative=target_relative_arr.round(4).tolist(), target_node_id=target_node_id)

        points_relative = [world_to_relative(p, origin_world_arr).round(4).tolist() for p in points_world]
        return ReachabilityProbe(True, geodesic_distance=geo, path_points_relative=points_relative, reason="reachable", target_position_relative=target_relative_arr.round(4).tolist(), target_node_id=target_node_id)

    def choose_next_waypoint(self, probe: ReachabilityProbe, current_relative: Sequence[float]) -> np.ndarray:
        current = np.asarray(current_relative, dtype=np.float32)
        if not probe.path_points_relative:
            return np.asarray(probe.target_position_relative, dtype=np.float32)
        min_step_radius = max(self.waypoint_radius, self.reach_radius)
        for point in probe.path_points_relative[1:]:
            arr = np.asarray(point, dtype=np.float32)
            if _distance3d(arr, current) > min_step_radius:
                return arr
        return np.asarray(probe.path_points_relative[-1], dtype=np.float32)

    def low_level_action_to(self, sim, waypoint_relative: Sequence[float], origin_world: Sequence[float]) -> str:
        state = sim.get_agent(0).get_state()
        current_relative = world_to_relative(np.asarray(state.position, dtype=np.float32), origin_world)
        target = np.asarray(waypoint_relative, dtype=np.float32)
        delta = target - current_relative
        if float(np.linalg.norm(delta)) <= self.reach_radius:
            return "target_reached"
        target_heading = math.atan2(-float(delta[0]), -float(delta[2]))
        heading = _quat_to_heading(state.rotation)
        diff = _angular_diff(target_heading, heading)
        if diff > self.turn_threshold:
            return "turn_left"
        if diff < -self.turn_threshold:
            return "turn_right"
        return "move_forward"

    def step(self, sim, target_relative: Sequence[float], origin_world: Sequence[float], target_node_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        probe = self.probe(sim, target_relative, origin_world, target_node_id=target_node_id)
        debug = probe.to_dict()
        debug["controller_mode"] = "pathfinder_next_waypoint"
        debug["target_snapped_to_navmesh"] = (
            probe.target_position_relative is not None
            and not np.allclose(np.asarray(target_relative, dtype=np.float32), np.asarray(probe.target_position_relative, dtype=np.float32), atol=1e-3)
        )
        if not probe.reachable:
            debug["low_action"] = "unreachable"
            return "unreachable", debug
        state = sim.get_agent(0).get_state()
        current_relative = world_to_relative(np.asarray(state.position, dtype=np.float32), origin_world)
        next_wp = self.choose_next_waypoint(probe, current_relative)
        action = self.low_level_action_to(sim, next_wp, origin_world)
        if action == "target_reached" and probe.reason != "target_reached":
            action = "move_forward"
        debug["next_waypoint_relative"] = next_wp.round(4).tolist()
        debug["low_action"] = action
        return action, _tolist(debug)

    def select_reachable_candidate(
        self,
        sim,
        origin_world: Sequence[float],
        topo_map,
        candidate_ids: Sequence[str],
        scores: Sequence[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        ranked = sorted(zip(candidate_ids, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        reachable: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        for node_id, score in ranked:
            node = topo_map.get_node(node_id)
            if node is None:
                skipped.append({"node_id": node_id, "score": float(score), "reason": "missing_node"})
                continue
            probe_node_id = node_id
            probe_node = node
            if getattr(node, "node_type", None) is not None and node.node_type.value == "object":
                resolved = topo_map.navigation_target_for_structure_node(node_id)
                if resolved is None or getattr(resolved, "node_type", None).value == "object":
                    skipped.append({"node_id": node_id, "score": float(score), "reason": "object_without_navigation_anchor"})
                    continue
                probe_node = resolved
                probe_node_id = resolved.node_id
            probe = self.probe(sim, probe_node.position, origin_world, target_node_id=probe_node_id)
            row = probe.to_dict()
            row.update({
                "node_id": probe_node_id,
                "source_node_id": node_id,
                "score": float(score),
                "type": probe_node.node_type.value,
            })
            if probe.reachable:
                reachable.append(row)
            else:
                skipped.append(row)
        selected = reachable[0] if reachable else None
        return {"selected": selected, "reachable_candidates": reachable, "unreachable_candidates": skipped}


class CollisionLikeTracker:
    """Detect repeated forward commands with nearly no relative displacement."""

    def __init__(self, move_epsilon: float = 0.03, trigger_steps: int = 3):
        self.move_epsilon = move_epsilon
        self.trigger_steps = trigger_steps
        self._last_position: Optional[np.ndarray] = None
        self.stuck_steps = 0

    def update(self, previous_action: Optional[str], current_position_relative: Sequence[float]) -> Dict[str, Any]:
        pos = np.asarray(current_position_relative, dtype=np.float32)
        movement = None if self._last_position is None else float(np.linalg.norm(pos - self._last_position))
        if previous_action == "move_forward" and movement is not None and movement < self.move_epsilon:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        self._last_position = pos.copy()
        return {
            "movement": movement,
            "stuck_steps": self.stuck_steps,
            "collision_like": self.stuck_steps >= self.trigger_steps,
        }


class GlobalGraphPlanner:
    # Plan shortest paths on topo graph navigable edges.
    def plan_to_target(self, topo_map, current_node_id, target_node_id):
        path = topo_map.shortest_path(current_node_id, target_node_id, EdgeType.NAVIGABLE)
        if path is None or len(path) < 2:
            return None
        return path[1:]
