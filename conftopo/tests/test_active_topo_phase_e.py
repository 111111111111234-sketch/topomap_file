"""Phase E: folding low-relevance rooms.

A room whose task relevance is low (does not match the goal room priors) has its
internal exploration targets (frontiers/candidates) and low-relevance objects
folded out of the candidate set. The room node, portals and visited waypoints
stay so the skeleton remains traversable. The room the agent is currently in is
never folded, and recovery relaxes folding entirely.
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_active_topo import ActiveTopoProjector
from conftopo.agents.goat_agent1_types import AgentPose, ProjectionConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode

FOLD_CFG = ProjectionConfig(suppress_enabled=True, fold_enabled=True, relevance_fold_threshold=0.45)


def _goal():
    return GoalNode(target_object="sofa", room_prior=["living room"])


def _build_map():
    m = DynamicTopoMap()
    # goal-relevant room (label matches prior -> relevance ~0.70)
    m.add_node(NodeType.ROOM, np.array([0.0, 0.0, 0.0]), confidence=0.8,
               label="living room", node_id="room_goal")
    # irrelevant room (relevance floors at 0.35 -> foldable)
    m.add_node(NodeType.ROOM, np.array([5.0, 0.0, 0.0]), confidence=0.8,
               label="bathroom", node_id="room_other")

    m.add_node(NodeType.WAYPOINT_VISITED, np.array([0.2, 0.0, 0.0]), node_id="wp_goal")
    m.add_node(NodeType.WAYPOINT_VISITED, np.array([5.0, 0.0, 0.0]), node_id="wp_other")
    m.cross_layer.bind_room_member("room_goal", "wp_goal")
    m.cross_layer.bind_room_member("room_other", "wp_other")

    m.add_node(NodeType.WAYPOINT_FRONTIER, np.array([1.0, 0.0, 0.0]), confidence=0.5,
               node_id="fr_goal", attributes={"created_from_waypoint": "wp_goal", "semantic_value": 0.3})
    m.add_node(NodeType.WAYPOINT_FRONTIER, np.array([6.0, 0.0, 0.0]), confidence=0.5,
               node_id="fr_other", attributes={"created_from_waypoint": "wp_other", "semantic_value": 0.3})
    m.add_node(NodeType.OBJECT, np.array([6.0, 0.0, 0.5]), confidence=0.8,
               label="toilet", node_id="obj_other", attributes={"room_id": "room_other"})
    return m


def _view(cfg, recovery_level=0):
    return ActiveTopoProjector(cfg).build(
        topo_map=_build_map(), goal=_goal(),
        pose=AgentPose(position=np.array([0.2, 0.0, 0.0], dtype=np.float32)),
        recovery_level=recovery_level,
    )


def test_folds_irrelevant_room_internals():
    v = _view(FOLD_CFG)
    # folded: frontier + object inside the irrelevant room
    assert not v.is_candidate("fr_other"), v.node_infos["fr_other"]
    assert not v.is_candidate("obj_other"), v.node_infos["obj_other"]
    # kept: relevant-room frontier, both room nodes, visited waypoints
    assert v.is_candidate("fr_goal")
    assert v.is_candidate("room_goal") and v.is_candidate("room_other")
    assert v.is_candidate("wp_goal") and v.is_candidate("wp_other")
    assert v.stats()["folded_node_count"] >= 2


def test_fold_disabled_keeps_all():
    v = _view(ProjectionConfig(suppress_enabled=True, fold_enabled=False))
    assert v.is_candidate("fr_other")
    assert v.is_candidate("obj_other")
    assert v.stats()["folded_node_count"] == 0


def test_current_room_never_folded():
    # Even if we make the current room low-relevance, standing in it exempts it.
    m = DynamicTopoMap()
    m.add_node(NodeType.ROOM, np.array([0.0, 0.0, 0.0]), confidence=0.8,
               label="bathroom", node_id="room_here")
    m.add_node(NodeType.WAYPOINT_VISITED, np.array([0.1, 0.0, 0.0]), node_id="wp_here")
    m.cross_layer.bind_room_member("room_here", "wp_here")
    m.add_node(NodeType.WAYPOINT_FRONTIER, np.array([1.0, 0.0, 0.0]), confidence=0.5,
               node_id="fr_here", attributes={"created_from_waypoint": "wp_here", "semantic_value": 0.3})
    v = ActiveTopoProjector(FOLD_CFG).build(
        topo_map=m, goal=_goal(),
        pose=AgentPose(position=np.array([0.1, 0.0, 0.0], dtype=np.float32)),
    )
    assert v.is_candidate("fr_here"), "frontier in the current room must stay active"


def test_recovery_relaxes_folding():
    v = _view(FOLD_CFG, recovery_level=1)
    assert v.is_candidate("fr_other"), "recovery should re-expose folded regions"
    assert v.stats()["folded_node_count"] == 0


if __name__ == "__main__":
    test_folds_irrelevant_room_internals()
    test_fold_disabled_keeps_all()
    test_current_room_never_folded()
    test_recovery_relaxes_folding()
    print("PASS: Phase E folding (low-relevance rooms folded, skeleton preserved)")
