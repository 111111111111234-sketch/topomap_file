"""STOP gate for clean Agent1: a lean, general three-condition gate.

STOP fires iff:
  1. the goal is visible,
  2. the detected object really is the goal (semantic relevance >= 0.70), and
  3. navigation has ARRIVED, where arrival is EITHER a pose-stall (forward
     progress stalled = as close as the navmesh lets us get) OR VLM proximity
     (range_bin near/very_near AND the bbox roughly centered).

There is deliberately NO absolute center-depth cutoff, no bbox-area distance
estimate, no bbox-confidence gate, and no CLIP crop-vs-goal veto: those stacked
AND-conditions special-cased distances (a 1.0 m depth cutoff can never fire at a
legal GOAT viewpoint, which sits ~1.5 m off the object) and made valid stops
unreachable. Arrival is scale- and placement-fair and needs no per-scene tuning.
The agent takes no depth input at all.

Run directly (``python conftopo/tests/test_stop_clip_gate.py``) or via pytest.
"""

from __future__ import annotations

import numpy as np

from conftopo.agents.goat_agent1_control import StopVerifier, median_bbox
from conftopo.perception.heavy_perceiver import normalize_bbox_flexible
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.heavy_perceiver import ObjectObservation
from conftopo.perception.perception_report import PerceptionReport


def _goal():
    g = GoalNode(target_object="chair", room_prior=["living room"])
    g.target_embedding = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return g


def _report(label="chair", goal_visible=True, is_full=True, bbox=None, range_bin="near"):
    obj = ObjectObservation(
        label=label,
        bbox=bbox if bbox is not None else [0.35, 0.15, 0.65, 0.90],
        confidence=0.9,
        visibility="visible",
        bearing="center",
        range_bin=range_bin,
        bbox_confidence="high",
        embedding=np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
    )
    return PerceptionReport(
        objects=[obj], goal_visible=goal_visible, is_full=is_full, source="vlm"
    )


def test_visible_relevant_arrived_stops():
    """The three-condition gate: goal visible + relevant + arrived -> STOP."""
    v = StopVerifier()
    assert v.can_stop(_report(), _goal(), arrived=True) is True
    assert v.last_debug["can_stop"] is True
    assert v.last_debug["arrived"] is True
    assert v.last_debug["reason"] == "visual_confirmed_stop"


def test_not_arrived_far_does_not_stop():
    """Visible + relevant but still far (no pose-stall, VLM range not near) -> no
    stop: neither arrival path is satisfied while genuinely closing distance."""
    v = StopVerifier()
    assert v.can_stop(_report(range_bin="mid"), _goal(), arrived=False) is False
    assert v.last_debug["reason"] == "visual_evidence_insufficient"


def test_visual_proximity_stops_without_pose_stall():
    """Visible + relevant + VLM says near AND centered -> STOP even without a
    pose-stall. This is the visual arrival path: while actively servoing toward
    the goal the pose keeps changing (pose-stall only flickers), so the coarse
    VLM proximity judgement is what latches the stop (the v21 rack sat ~0.39 m
    from the goal viewpoint, near+centered, yet never stopped before this)."""
    v = StopVerifier()
    assert v.can_stop(_report(range_bin="near"), _goal(), arrived=False) is True
    assert v.last_debug["visual_arrived"] is True
    assert v.last_debug["reason"] == "visual_confirmed_stop"


def test_near_but_off_center_needs_pose_stall():
    """Near but pushed to the frame edge is NOT a visual arrival on its own; it
    still needs a genuine pose-stall to stop (guards against stopping while the
    goal is only glancingly in view at the edge)."""
    v = StopVerifier()
    edge = _report(bbox=[0.72, 0.15, 0.98, 0.90], range_bin="near")
    assert v.can_stop(edge, _goal(), arrived=False) is False


def test_not_visible_does_not_stop():
    v = StopVerifier()
    assert v.can_stop(_report(goal_visible=False), _goal(), arrived=True) is False


def test_irrelevant_label_does_not_stop():
    """A detection that isn't the goal yields no target object at all."""
    v = StopVerifier()
    assert v.can_stop(_report(label="lamp"), _goal(), arrived=True) is False
    assert v.last_debug["reason"] == "missing_goal_or_target"


def test_empty_report_missing_target():
    v = StopVerifier()
    empty = PerceptionReport(objects=[], goal_visible=True, is_full=True, source="vlm")
    assert v.can_stop(empty, _goal(), arrived=True) is False
    assert v.last_debug["reason"] == "missing_goal_or_target"


def test_off_center_still_stops():
    """A target pushed to the frame edge (VLM bbox jitter) no longer blocks a
    stop: horizontal centering is not a STOP condition."""
    v = StopVerifier()
    assert v.can_stop(_report(bbox=[0.72, 0.15, 0.98, 0.90]), _goal(), arrived=True) is True


def test_elevated_object_stops():
    """Wall-mounted / elevated target (bbox high in frame, tiny area) that would
    fail every bbox-geometry gate still STOPS on arrival alone."""
    v = StopVerifier()
    assert v.can_stop(_report(bbox=[0.45, 0.20, 0.60, 0.30]), _goal(), arrived=True) is True


def test_bbox_scale_autodetect():
    """VLM bboxes may arrive in 0-1000 (Qwen native) or 0-100 scale instead of
    [0,1]; they should be rescaled, not dropped. Pixel coords stay rejected."""
    assert normalize_bbox_flexible([170, 0, 830, 990]) == [0.17, 0.0, 0.83, 0.99]
    assert normalize_bbox_flexible([17, 0, 83, 99]) == [0.17, 0.0, 0.83, 0.99]
    assert normalize_bbox_flexible([0.17, 0.0, 0.83, 0.99]) == [0.17, 0.0, 0.83, 0.99]
    assert normalize_bbox_flexible([170, 0, 8300, 9900]) is None  # unknown image size


def test_bbox_median_smoothing_damps_jitter():
    """Per-coordinate median rejects a single jittered frame (observed VLM
    behavior: center_x alternating 0.5 / 0.67 on a static scene)."""
    hist = [[0.20, 0.1, 0.80, 0.80], [0.33, 0.0, 1.00, 0.99], [0.20, 0.1, 0.80, 0.80]]
    assert median_bbox(hist) == [0.20, 0.1, 0.80, 0.80]
    assert median_bbox([]) is None


if __name__ == "__main__":
    test_visible_relevant_arrived_stops()
    test_not_arrived_far_does_not_stop()
    test_visual_proximity_stops_without_pose_stall()
    test_near_but_off_center_needs_pose_stall()
    test_not_visible_does_not_stop()
    test_irrelevant_label_does_not_stop()
    test_empty_report_missing_target()
    test_off_center_still_stops()
    test_elevated_object_stops()
    test_bbox_scale_autodetect()
    test_bbox_median_smoothing_damps_jitter()
    print("PASS: lean 3-condition STOP gate (visible + relevant + arrived)")
