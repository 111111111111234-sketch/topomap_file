"""Phase C regression: routing planner candidates through the ActiveTopoView is
behavior-neutral.

The projector (suppress_enabled) only marks nodes the planner already skips
unconditionally (blocked / consumed navigation nodes, blocked portals) as
SUPPRESSED. Therefore gating candidate generation on the view must produce the
exact same intents and navigation target as ignoring it, while
active_node_count drops below the full graph.

Run directly or via pytest.
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_active_topo import ActiveTopoProjector
from conftopo.agents.goat_agent1_planner import NavigationPlanner, StructurePlanner
from conftopo.agents.goat_agent1_types import AgentPose, ProjectionConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.hypothesis_pool import HypothesisPool
from conftopo.core.instruction_graph import GoalNode


def _build_map() -> DynamicTopoMap:
    m = DynamicTopoMap()
    m.add_node(NodeType.WAYPOINT_VISITED, np.array([0.0, 0.0, 0.0]), node_id="v0")
    # a clean frontier, a consumed one, a blocked one
    m.add_node(
        NodeType.WAYPOINT_FRONTIER, np.array([2.0, 0.0, 0.0]), confidence=0.7,
        node_id="f_ok", attributes={"semantic_value": 0.3},
    )
    m.add_node(
        NodeType.WAYPOINT_FRONTIER, np.array([2.5, 0.0, 1.0]), confidence=0.9,
        node_id="f_consumed", attributes={"semantic_value": 0.9, "consumed": True},
    )
    m.add_node(
        NodeType.WAYPOINT_FRONTIER, np.array([1.0, 0.0, 2.0]), confidence=0.9,
        node_id="f_blocked", attributes={"semantic_value": 0.9, "blocked": True},
    )
    return m


def _goal() -> GoalNode:
    return GoalNode(target_object="chair", room_prior=["living room"])


def _pose() -> AgentPose:
    return AgentPose(position=np.array([0.0, 0.0, 0.0], dtype=np.float32))


def _intent_key(intents):
    return [
        (i.intent_type, i.preferred_anchor_waypoint_id, round(float(i.score), 9))
        for i in intents
    ]


def test_view_gating_is_behavior_neutral():
    m = _build_map()
    goal, pose = _goal(), _pose()
    view = ActiveTopoProjector(ProjectionConfig(suppress_enabled=True)).build(
        topo_map=m, goal=goal, pose=pose,
    )

    sp = StructurePlanner()
    hp = HypothesisPool()
    ungated = sp.build_intents(topo_map=m, hypotheses=hp, goal=goal, pose=pose)
    gated = sp.build_intents(
        topo_map=m, hypotheses=hp, goal=goal, pose=pose, active_view=view,
    )
    assert _intent_key(ungated) == _intent_key(gated), (_intent_key(ungated), _intent_key(gated))

    npl = NavigationPlanner()
    t_ungated = npl.select_from_intents(topo_map=m, intents=ungated[:3], pose=pose)
    t_gated = npl.select_from_intents(topo_map=m, intents=gated[:3], pose=pose, active_view=view)
    assert (t_ungated.node_id, round(t_ungated.score, 9)) == (t_gated.node_id, round(t_gated.score, 9))
    # the only viable frontier is the clean one
    assert t_gated.node_id == "f_ok", t_gated.node_id


def test_suppression_shrinks_active_set():
    m = _build_map()
    view = ActiveTopoProjector(ProjectionConfig(suppress_enabled=True)).build(
        topo_map=m, goal=_goal(), pose=_pose(),
    )
    stats = view.stats()
    # 4 nodes total; consumed + blocked frontiers suppressed -> 2 active.
    assert stats["persistent_node_count"] == 4, stats
    assert stats["active_node_count"] == 2, stats
    assert stats["activity_distribution"]["suppressed"] == 2, stats
    assert not view.is_candidate("f_consumed")
    assert not view.is_candidate("f_blocked")
    assert view.is_candidate("f_ok")


def test_disabled_suppression_keeps_all_active():
    m = _build_map()
    view = ActiveTopoProjector(ProjectionConfig(suppress_enabled=False)).build(
        topo_map=m, goal=_goal(), pose=_pose(),
    )
    stats = view.stats()
    assert stats["active_node_count"] == stats["persistent_node_count"] == 4, stats
    assert stats["activity_distribution"]["suppressed"] == 0, stats


if __name__ == "__main__":
    test_view_gating_is_behavior_neutral()
    test_suppression_shrinks_active_set()
    test_disabled_suppression_keeps_all_active()
    print("PASS: Phase C view gating is behavior-neutral; suppression shrinks active set")
