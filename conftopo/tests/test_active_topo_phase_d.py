"""Phase D: VERIFY closed loop.

Deterministic tests (no VLM) covering:
  1. projector classifies high-relevance / low-confidence objects as VERIFY;
  2. memory.record_object_verification moves confidence up (success) / down +
     strong_negative_evidence (failure) via confidence.py;
  3. the StateMachine raises the STOP bar for VERIFY targets and fires the
     success / failure callbacks (the actual closed loop).
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_active_topo import ActiveTopoProjector
from conftopo.agents.goat_agent1_control import StateMachine
from conftopo.agents.goat_agent1_memory import Agent1MemoryWriter
from conftopo.agents.goat_agent1_types import (
    AgentPose,
    NavigationTarget,
    ProjectionConfig,
    SearchIntent,
)
from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport

VERIFY_CFG = ProjectionConfig(suppress_enabled=True, verify_enabled=True)


def _goal():
    g = GoalNode(target_object="chair", room_prior=["living room"])
    g.target_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return g


def _pose():
    return AgentPose(position=np.zeros(3, dtype=np.float32))


# --------------------------------------------------------------------------- #
# 1. projector VERIFY classification
# --------------------------------------------------------------------------- #
def test_projector_marks_single_view_object_verify():
    m = DynamicTopoMap()
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    # single-view goal object (seen once) -> VERIFY, even at high detection conf
    m.add_node(NodeType.OBJECT, np.array([1.0, 0.0, 0.0]), embedding=emb,
               confidence=0.90, label="chair", node_id="obj_single",
               attributes={"seen_count": 1})
    # multi-view goal object -> ACTIVE (stops normally)
    m.add_node(NodeType.OBJECT, np.array([1.5, 0.0, 0.0]), embedding=emb,
               confidence=0.70, label="chair", node_id="obj_multi",
               attributes={"seen_count": 3, "multi_view_count": 3})
    view = ActiveTopoProjector(VERIFY_CFG).build(topo_map=m, goal=_goal(), pose=_pose())
    assert view.is_verify("obj_single"), view.node_infos["obj_single"]
    assert not view.is_verify("obj_multi"), view.node_infos["obj_multi"]
    # VERIFY nodes are still candidates (planner behavior unchanged in Phase C/D)
    assert view.is_candidate("obj_single")


def test_verify_disabled_keeps_active():
    m = DynamicTopoMap()
    emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    m.add_node(NodeType.OBJECT, np.array([1.0, 0.0, 0.0]), embedding=emb,
               confidence=0.90, label="chair", node_id="obj_single",
               attributes={"seen_count": 1})
    view = ActiveTopoProjector(ProjectionConfig(suppress_enabled=True, verify_enabled=False)).build(
        topo_map=m, goal=_goal(), pose=_pose(),
    )
    assert not view.is_verify("obj_single")


# --------------------------------------------------------------------------- #
# 2. memory verification confidence updates
# --------------------------------------------------------------------------- #
def test_verification_success_raises_confidence():
    m = DynamicTopoMap()
    m.add_node(NodeType.OBJECT, np.array([1.0, 0.0, 0.0]), confidence=0.40,
               label="chair", node_id="obj")
    mem = Agent1MemoryWriter(m)
    new_conf = mem.record_object_verification("obj", success=True, detection_score=0.9)
    assert new_conf > 0.40, new_conf
    assert int(m.get_node("obj").attributes.get("verify_success_count")) == 1


def test_verification_failure_lowers_confidence_and_marks_negative():
    m = DynamicTopoMap()
    m.add_node(NodeType.OBJECT, np.array([1.0, 0.0, 0.0]), confidence=0.40,
               label="chair", node_id="obj")
    mem = Agent1MemoryWriter(m)
    new_conf = mem.record_object_verification("obj", success=False, detection_score=0.0)
    node = m.get_node("obj")
    assert new_conf < 0.40, new_conf
    assert float(node.attributes.get("strong_negative_evidence", 0.0)) >= 0.5
    assert int(node.attributes.get("verify_fail_count")) == 1


# --------------------------------------------------------------------------- #
# 3. state-machine VERIFY loop (deterministic)
# --------------------------------------------------------------------------- #
def _report(*, range_bin="near"):
    obj = ObjectObservation(
        label="chair",
        bbox=[0.40, 0.40, 0.60, 0.70],
        confidence=0.80,
        visibility="visible",
        range_bin=range_bin,
        bbox_confidence="high",
    )
    obj.bearing = "center"
    return PerceptionReport(
        objects=[obj], goal_visible=True, goal_match_confidence=0.8,
        is_full=True, source="vlm",
    )


def _target():
    intent = SearchIntent(intent_type="object", structure_node_id="obj")
    return NavigationTarget(
        node_id="way", position=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        target_type="object_anchor", source_intent=intent, structure_node_id="obj",
    )


def test_verify_target_requires_full_buffer_and_fires_success():
    goal, target = _goal(), _target()

    # Baseline: no verify -> STOP after 2 hits.
    sm_base = StateMachine()
    stops_base = None
    for step in range(1, 8):
        act = sm_base.act(step_id=step, target=target, report=_report(), goal=goal,
                          fresh_report=True, needs_verify=False)
        if act["action"] == "stop":
            stops_base = step
            break

    # Verify path: needs one extra confirming frame, and commits success.
    sm = StateMachine()
    fired = {"ok": 0, "fail": 0}
    stops_verify = None
    for step in range(1, 8):
        act = sm.act(
            step_id=step, target=target, report=_report(), goal=goal,
            fresh_report=True, needs_verify=True,
            on_verify_success=lambda: fired.__setitem__("ok", fired["ok"] + 1),
            on_verify_failure=lambda: fired.__setitem__("fail", fired["fail"] + 1),
        )
        if act["action"] == "stop":
            stops_verify = step
            break

    assert stops_base is not None and stops_verify is not None
    assert stops_verify > stops_base, (stops_base, stops_verify)
    assert fired["ok"] == 1 and fired["fail"] == 0, fired


def test_verify_target_failure_writes_negative():
    goal, target = _goal(), _target()
    sm = StateMachine()
    fired = {"ok": 0, "fail": 0}
    for step in range(1, 15):
        # visible but never stoppable (mid range -> not "near")
        sm.act(
            step_id=step, target=target, report=_report(range_bin="mid"), goal=goal,
            fresh_report=True, needs_verify=True,
            on_verify_success=lambda: fired.__setitem__("ok", fired["ok"] + 1),
            on_verify_failure=lambda: fired.__setitem__("fail", fired["fail"] + 1),
        )
        if fired["fail"]:
            break
    assert fired["fail"] == 1 and fired["ok"] == 0, fired


if __name__ == "__main__":
    test_projector_marks_single_view_object_verify()
    test_verify_disabled_keeps_active()
    test_verification_success_raises_confidence()
    test_verification_failure_lowers_confidence_and_marks_negative()
    test_verify_target_requires_full_buffer_and_fires_success()
    test_verify_target_failure_writes_negative()
    print("PASS: Phase D VERIFY loop (projector + memory + state machine)")
