"""Phase F: recovery ladder.

When the active region is exhausted (no search progress for a budget of steps),
the recovery level escalates (0 -> 1 -> 2, monotonic within a goal). Escalation
records the exhausted region and, via the projector, relaxes folding so the agent
can reach previously folded / next-relevant regions.
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_active_topo import ActiveTopoProjector
from conftopo.agents.goat_agent1_types import AgentPose, ProjectionConfig, RecoveryState
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode

BUDGET = 20


def test_escalates_only_after_budget():
    r = RecoveryState()
    assert not r.maybe_escalate(no_progress_steps=BUDGET - 1, budget=BUDGET, current_region="room_x")
    assert r.level == 0 and not r.exhausted_regions
    assert r.maybe_escalate(no_progress_steps=BUDGET, budget=BUDGET, current_region="room_x")
    assert r.level == 1 and "room_x" in r.exhausted_regions


def test_level_is_monotonic_and_capped():
    r = RecoveryState()
    assert r.maybe_escalate(no_progress_steps=99, budget=BUDGET, current_region="a")
    assert r.maybe_escalate(no_progress_steps=99, budget=BUDGET, current_region="b")
    assert r.level == 2
    # capped at max_level
    assert not r.maybe_escalate(no_progress_steps=99, budget=BUDGET, current_region="c")
    assert r.level == 2
    assert {"a", "b"} <= r.exhausted_regions


def test_reset_clears_ladder():
    r = RecoveryState()
    r.maybe_escalate(no_progress_steps=99, budget=BUDGET, current_region="a")
    r.reset()
    assert r.level == 0 and not r.exhausted_regions


def test_recovery_reexposes_folded_region():
    """The concrete pay-off: folded frontiers in an irrelevant room become
    candidates again once recovery kicks in."""
    m = DynamicTopoMap()
    m.add_node(NodeType.ROOM, np.array([0.0, 0.0, 0.0]), confidence=0.8,
               label="living room", node_id="room_goal")
    m.add_node(NodeType.ROOM, np.array([5.0, 0.0, 0.0]), confidence=0.8,
               label="bathroom", node_id="room_other")
    m.add_node(NodeType.WAYPOINT_VISITED, np.array([0.2, 0.0, 0.0]), node_id="wp_goal")
    m.add_node(NodeType.WAYPOINT_VISITED, np.array([5.0, 0.0, 0.0]), node_id="wp_other")
    m.cross_layer.bind_room_member("room_goal", "wp_goal")
    m.cross_layer.bind_room_member("room_other", "wp_other")
    m.add_node(NodeType.WAYPOINT_FRONTIER, np.array([6.0, 0.0, 0.0]), confidence=0.5,
               node_id="fr_other", attributes={"created_from_waypoint": "wp_other", "semantic_value": 0.3})

    goal = GoalNode(target_object="sofa", room_prior=["living room"])
    pose = AgentPose(position=np.array([0.2, 0.0, 0.0], dtype=np.float32))
    proj = ActiveTopoProjector(ProjectionConfig(suppress_enabled=True, fold_enabled=True,
                                                relevance_fold_threshold=0.45))

    v0 = proj.build(topo_map=m, goal=goal, pose=pose, recovery_level=0)
    assert not v0.is_candidate("fr_other"), "folded before recovery"

    v1 = proj.build(topo_map=m, goal=goal, pose=pose, recovery_level=1)
    assert v1.is_candidate("fr_other"), "recovery must re-expose the folded frontier"
    assert v1.stats()["recovery_level"] == 1


if __name__ == "__main__":
    test_escalates_only_after_budget()
    test_level_is_monotonic_and_capped()
    test_reset_clears_ladder()
    test_recovery_reexposes_folded_region()
    print("PASS: Phase F recovery ladder (escalation + fold relaxation)")
