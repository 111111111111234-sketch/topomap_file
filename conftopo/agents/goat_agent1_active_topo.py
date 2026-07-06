"""ActiveTopoView projector: a thin, goal-conditioned view over the persistent
DynamicTopoMap.

Design contract (see project plan):
  - The persistent graph stores only environment facts: map confidence (C) and
    reachability/accessibility (A).
  - Task relevance (R) and the per-node ActivityState live *only* here, in the
    view, and are discarded when the goal changes.
  - There is one persistent graph and one projected view -- never four physical
    maps.

Phase B scope: the projector classifies every node as ACTIVE and annotates it
with (R, C, A) for observability. It does not fold, suppress, or reroute, so a
planner reading the full graph behaves identically. Later phases tighten
``ProjectionConfig`` to actually drive folding / verification / recovery.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from conftopo.agents.goat_agent1_planner import (
    frontier_map_confidence,
    frontier_task_relevance,
    landmark_map_confidence,
    landmark_task_relevance,
    object_map_confidence,
    object_task_relevance,
    portal_map_confidence,
    portal_task_relevance,
    room_map_confidence,
    room_task_relevance,
)
from conftopo.agents.goat_agent1_types import (
    ActiveNodeInfo,
    ActiveTopoView,
    ActivityState,
    AgentPose,
    ProjectionConfig,
)
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalNode


def _score_dict(value) -> dict:
    if isinstance(value, dict):
        return {str(k).lower().strip(): float(v) for k, v in value.items()}
    return {}


def _object_view_count(node: SemanticNode) -> int:
    """How many distinct views corroborate this object node (>=1)."""
    return max(
        int(node.attributes.get("multi_view_count", 0) or 0),
        int(node.attributes.get("seen_count", 0) or 0),
        int(node.visit_count or 0),
        1,
    )


def _generic_reachability(node: SemanticNode) -> float:
    """A: node-level reachability penalty proxy from failure history."""
    failed = float(node.attributes.get("failed_count", 0) or 0)
    unreachable = float(node.attributes.get("unreachable_count", 0) or 0)
    return 0.25 * failed + 0.75 * unreachable


def _frontier_semantic(node: SemanticNode, goal: Optional[GoalNode]) -> float:
    semantic = float(node.attributes.get("semantic_value", 0.0) or 0.0)
    if goal is not None:
        priors = {str(room).lower().strip() for room in goal.room_prior or []}
        expected = _score_dict(node.attributes.get("expected_room_probs", {}))
        room_match = max((float(expected.get(room, 0.0)) for room in priors), default=0.0)
        semantic = max(semantic, room_match)
    return semantic


_FOLDABLE_NAV_TYPES = (NodeType.WAYPOINT_FRONTIER, NodeType.WAYPOINT_CANDIDATE)


class ActiveTopoProjector:
    """Builds an :class:`ActiveTopoView` from the persistent map for a goal."""

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()

    def _foldable_rooms(
        self,
        topo_map: DynamicTopoMap,
        goal: Optional[GoalNode],
        current_room_id: Optional[str],
        recovery_level: int,
    ) -> set:
        """Rooms whose task relevance is low enough to fold away.

        The room the agent is currently in is never folded (so it can always be
        explored), and recovery relaxes folding entirely so exhausted searches
        can reach back into low-relevance regions.
        """
        if not self.config.fold_enabled or goal is None or recovery_level > 0:
            return set()
        foldable = set()
        for room in topo_map.structure_layer.nodes():
            if room.node_type != NodeType.ROOM:
                continue
            if room.node_id == current_room_id:
                continue
            if room_task_relevance(room, goal) < self.config.relevance_fold_threshold:
                foldable.add(room.node_id)
        return foldable

    @staticmethod
    def _node_room_id(node: SemanticNode, topo_map: DynamicTopoMap) -> Optional[str]:
        room_id = node.attributes.get("room_id") or node.attributes.get("room_region_id")
        if room_id:
            return str(room_id)
        created = node.attributes.get("created_from_waypoint")
        if created:
            wp = topo_map.get_node(str(created))
            if wp is not None:
                rid = wp.attributes.get("room_region_id") or wp.attributes.get("room_id")
                return str(rid) if rid else None
        return None

    def _structure_relevance_confidence(
        self, node: SemanticNode, goal: Optional[GoalNode]
    ) -> tuple[float, float]:
        if goal is None:
            return 0.0, float(node.confidence)
        if node.node_type == NodeType.OBJECT:
            return object_task_relevance(node, goal), object_map_confidence(node)
        if node.node_type == NodeType.ROOM:
            return room_task_relevance(node, goal), room_map_confidence(node)
        if node.node_type == NodeType.LANDMARK:
            return landmark_task_relevance(node, goal), landmark_map_confidence(node)
        if node.node_type == NodeType.PORTAL:
            return portal_task_relevance(node, goal), portal_map_confidence(node)
        return 0.0, float(node.confidence)

    def _classify(
        self,
        node: SemanticNode,
        relevance: float,
        map_confidence: float,
        *,
        is_navigation: bool,
    ) -> ActivityState:
        """Classify a node's task-conditioned activity state.

        Phase C only introduces *behavior-neutral* suppression: nodes the
        planner already skips unconditionally (blocked/consumed navigation
        nodes, blocked portals) are marked SUPPRESSED so ``active_node_count``
        reflects the real candidate set. Everything else stays ACTIVE. Folding
        / verification (relevance-driven) arrive in later phases via the
        (relevance, map_confidence) inputs already threaded here.
        """
        if self.config.suppress_enabled:
            blocked = bool(node.attributes.get("blocked", False))
            consumed = bool(node.attributes.get("consumed", False))
            if is_navigation:
                if blocked or consumed:
                    return ActivityState.SUPPRESSED
            elif node.node_type == NodeType.PORTAL and blocked:
                return ActivityState.SUPPRESSED
        # VERIFY: an object we care about (high R) whose map confidence (C) is
        # not yet corroborated by multiple views (single glimpse). It stays a
        # candidate, but the control layer must re-observe it before trusting it
        # enough to STOP. Multi-view objects STOP normally.
        if (
            self.config.verify_enabled
            and not is_navigation
            and node.node_type == NodeType.OBJECT
            and relevance >= self.config.relevance_verify_threshold
            and _object_view_count(node) < self.config.verify_min_view_count
        ):
            return ActivityState.VERIFY
        return ActivityState.ACTIVE

    def build(
        self,
        *,
        topo_map: DynamicTopoMap,
        goal: Optional[GoalNode],
        pose: Optional[AgentPose],
        recovery_level: int = 0,
        prev_view: Optional[ActiveTopoView] = None,
        goal_id: Optional[str] = None,
        step_id: int = -1,
    ) -> ActiveTopoView:
        view = ActiveTopoView(goal_id=goal_id, built_step=step_id, recovery_level=recovery_level)
        view.rebuild_count = (prev_view.rebuild_count + 1) if prev_view is not None else 1

        entry_waypoint = self._nearest_visited(topo_map, pose)
        current_room_id = self._room_of_waypoint(topo_map, entry_waypoint)
        foldable_rooms = self._foldable_rooms(topo_map, goal, current_room_id, recovery_level)

        total = 0
        for node in topo_map.structure_layer.nodes():
            total += 1
            relevance, map_conf = self._structure_relevance_confidence(node, goal)
            reach = _generic_reachability(node)
            state = self._classify(node, relevance, map_conf, is_navigation=False)
            # Fold low-relevance objects that live inside a folded room. Keep the
            # room node + portals so the skeleton stays traversable.
            if (
                state == ActivityState.ACTIVE
                and foldable_rooms
                and node.node_type == NodeType.OBJECT
                and relevance < self.config.relevance_fold_threshold
                and self._node_room_id(node, topo_map) in foldable_rooms
            ):
                state = ActivityState.FOLDED
            view.node_infos[node.node_id] = ActiveNodeInfo(
                node_id=node.node_id,
                node_kind="structure",
                relevance=float(relevance),
                map_confidence=float(map_conf),
                reachability=float(reach),
                activity_state=state,
                utility=float(relevance + map_conf - reach),
            )
            if state == ActivityState.ACTIVE:
                view.active_structure.add(node.node_id)
            elif state == ActivityState.VERIFY:
                view.verification_nodes.add(node.node_id)
            elif state == ActivityState.FOLDED:
                view.folded_regions.add(node.node_id)

        for node in topo_map.navigation_layer.nodes():
            total += 1
            if node.node_type == NodeType.WAYPOINT_FRONTIER:
                relevance = frontier_task_relevance(_frontier_semantic(node, goal))
                map_conf = frontier_map_confidence(float(node.confidence))
            else:
                relevance = 0.0
                map_conf = float(node.confidence)
            reach = _generic_reachability(node)
            state = self._classify(node, relevance, map_conf, is_navigation=True)
            # Fold exploration targets (frontiers/candidates) inside folded rooms
            # so the planner stops chasing them into irrelevant regions.
            if (
                state == ActivityState.ACTIVE
                and foldable_rooms
                and node.node_type in _FOLDABLE_NAV_TYPES
                and self._node_room_id(node, topo_map) in foldable_rooms
            ):
                state = ActivityState.FOLDED
            view.node_infos[node.node_id] = ActiveNodeInfo(
                node_id=node.node_id,
                node_kind="navigation",
                relevance=float(relevance),
                map_confidence=float(map_conf),
                reachability=float(reach),
                activity_state=state,
                utility=float(relevance + map_conf - reach),
            )
            if state == ActivityState.ACTIVE:
                view.active_navigation.add(node.node_id)
            elif state == ActivityState.FOLDED:
                view.folded_regions.add(node.node_id)

        view.persistent_node_count = total
        view.entry_waypoint = entry_waypoint
        return view

    @staticmethod
    def _room_of_waypoint(
        topo_map: DynamicTopoMap, waypoint_id: Optional[str]
    ) -> Optional[str]:
        if not waypoint_id:
            return None
        wp = topo_map.get_node(str(waypoint_id))
        if wp is None:
            return None
        rid = wp.attributes.get("room_region_id") or wp.attributes.get("room_id")
        return str(rid) if rid else None

    @staticmethod
    def _nearest_visited(
        topo_map: DynamicTopoMap, pose: Optional[AgentPose]
    ) -> Optional[str]:
        if pose is None:
            return None
        best_id: Optional[str] = None
        best_dist = float("inf")
        for node in topo_map.navigation_layer.nodes():
            if node.node_type != NodeType.WAYPOINT_VISITED:
                continue
            d = float(np.linalg.norm(np.asarray(pose.position, dtype=np.float32) - node.position))
            if d < best_dist:
                best_dist = d
                best_id = node.node_id
        return best_id
