"""Controller and visual stop verification for clean Agent1."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from conftopo.agents.goat_agent1_memory import (
    GoalEvidenceFilter,
    GoalRelevanceScorer,
    bbox_center_x,
)
from conftopo.agents.goat_agent1_types import GOATState, NavigationTarget
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.perception_report import PerceptionReport


# STOP proximity is decided purely by ARRIVAL: the agent kept trying to move
# forward but its own pose stopped advancing (it is physically standing at / as
# close to the target as the navmesh allows). This is scale-fair and
# placement-fair and needs no per-scene tuning. The agent takes NO depth input:
# an absolute center-depth cutoff could never fire at a legal GOAT viewpoint
# (which sits ~1.5 m off the object), so depth was removed entirely.

# VLM bboxes jitter heavily frame-to-frame even on a static scene (measured
# consecutive-frame IoU 0.44-1.0, center +/-0.17). We smooth the target bbox with
# a short per-coordinate median before using it for centering.
BBOX_SMOOTH_WINDOW = 3


def median_bbox(history: List[List[float]]) -> Optional[List[float]]:
    """Per-coordinate median of recent bboxes; damps VLM localization jitter."""
    valid = [b for b in history if b and len(b) == 4]
    if not valid:
        return None
    return [float(statistics.median(coord)) for coord in zip(*valid)]


@dataclass
class ControlState:
    mode: GOATState = GOATState.SEARCH
    transition_reason: str = "init"
    state_steps: int = 0
    visible_window: List[bool] = field(default_factory=list)
    confirm_buffer: List[bool] = field(default_factory=list)
    confirm_attempts: int = 0
    verify_committed: bool = False
    last_goal_seen_step: int = -1
    target_lost_steps: int = 0
    active_object_node_id: Optional[str] = None
    active_anchor_waypoint_id: Optional[str] = None
    active_track_key: Optional[str] = None
    confirm_cooldown_until_step: int = -1
    confirm_turn_streak: int = 0


class Controller:
    def search_action(self, target: NavigationTarget) -> Dict[str, Any]:
        if target.position is None:
            return {"action": "turn_left", "reason": "search_no_navigation_target"}
        return {
            "action": "navigate",
            "target_position": target.position,
            "target_node_id": target.node_id,
            "target_type": target.target_type,
            "reason": target.reason,
        }

    def confirm_action(
        self,
        report: Optional[PerceptionReport],
        target: NavigationTarget,
        goal: Optional[GoalNode],
        arrived: bool = False,
        bbox_override: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Center the visible target, close distance, hold still when near."""
        obj = target_object(report, goal)
        if obj is None:
            return self.search_action(target)
        box = bbox_override if bbox_override is not None else obj.get("bbox")
        center = bbox_center_x(box)
        if center is None:
            bearing = str(obj.get("bearing", "")).lower()
            if bearing == "left":
                return {"action": "turn_left", "reason": "confirm_bearing_left"}
            if bearing == "right":
                return {"action": "turn_right", "reason": "confirm_bearing_right"}
            return {"action": "move_forward", "reason": "confirm_visible_no_bbox"}
        if center < 0.35:
            return {"action": "turn_left", "reason": "confirm_center_target"}
        if center > 0.65:
            return {"action": "turn_right", "reason": "confirm_center_target"}
        # Hold once we have arrived at the object (within reach of the nearest
        # goal object node); otherwise keep closing the distance. No depth gate.
        if arrived:
            return {"action": "hold_and_verify", "reason": "confirm_hold_centered"}
        return {"action": "move_forward", "reason": "confirm_approach_target"}

    def stop_action(self) -> Dict[str, Any]:
        return {"action": "stop", "reason": "verified_goal"}


class StopVerifier:
    """Single, lean, general STOP gate.

    STOP requires three conditions: the goal is visible, the detection really is
    the goal (semantic relevance >= 0.70), and the agent has ARRIVED. Arrival is
    satisfied by EITHER a pose-stall (kept trying to move forward but pose stopped
    advancing, supplied by the agent) OR a VLM-proximity signal (range_bin
    near/very_near AND the smoothed bbox roughly centered). No absolute-depth
    cutoff, no bbox-area distance estimate, no bbox-confidence gate, no CLIP veto
    -- those stacked AND-conditions special-cased distances and made valid stops
    unreachable. Multi-frame agreement is enforced by the state machine's confirm
    buffer, not here.
    """

    def __init__(self):
        self.last_debug: Dict[str, Any] = {}

    def can_stop(
        self,
        report: Optional[PerceptionReport],
        goal: Optional[GoalNode],
        arrived: bool = False,
        bbox_override: Optional[List[float]] = None,
    ) -> bool:
        obj = target_object(report, goal)
        if report is None or goal is None or obj is None:
            self.last_debug = {"can_stop": False, "reason": "missing_goal_or_target"}
            return False
        relevance = GoalRelevanceScorer.label_relevance(str(obj.get("label", "")), goal)
        # General STOP gate -- goal visible, semantically the goal, and ARRIVED.
        # "Arrived" is satisfied by EITHER of two placement-fair signals:
        #   (a) pose-stall: the agent kept trying to move forward but its own pose
        #       stopped advancing (physically at / blocked by the target), OR
        #   (b) VLM proximity: the goal is qualitatively CLOSE (range_bin
        #       near / very_near) AND roughly CENTERED in view.
        # Path (a) alone is too brittle in CONFIRM: while the agent is still
        # visually servoing (turn / step / step) its pose keeps changing, so the
        # stall only flickers True for a single frame and never fills the state
        # machine's 2/3 confirm buffer (the v21 rack sat ~0.39 m from the goal
        # viewpoint with the VLM screaming "near, centered, stop_candidate" for
        # ~10 frames yet never stopped). Path (b) trusts the VLM's own coarse
        # distance judgement -- which is what the whole approach is built on -- so
        # we stop once we are demonstrably close and looking at the goal. We still
        # avoid the discredited absolute-depth / bbox-area / CLIP gates.
        box = bbox_override if bbox_override is not None else obj.get("bbox")
        center = bbox_center_x(box)
        range_bin = str(obj.get("range_bin", "")).lower()
        centered = center is not None and abs(float(center) - 0.5) <= 0.30
        visual_arrived = range_bin in {"near", "very_near"} and centered
        arrived_final = bool(arrived) or visual_arrived
        ok = bool(report.goal_visible) and relevance >= 0.70 and arrived_final
        self.last_debug = {
            "can_stop": bool(ok),
            "goal_visible": bool(report.goal_visible),
            "relevance": round(float(relevance), 4),
            "arrived": bool(arrived),
            "visual_arrived": bool(visual_arrived),
            "range_bin": range_bin,
            "bbox_center": None if center is None else round(float(center), 4),
            "bbox_centered": bool(centered),
            "bbox_smoothed": bbox_override is not None,
            "bbox_confidence": str(obj.get("bbox_confidence", "")).lower(),
            "reason": "visual_confirmed_stop" if ok else "visual_evidence_insufficient",
        }
        return bool(ok)


class StateMachine:
    """Three-state control: SEARCH -> CONFIRM -> STOP.

    SEARCH drives global navigation to the planner target. Once the goal is
    visually detected, CONFIRM centers/approaches and accumulates STOP votes in
    a single buffer. STOP fires only after multi-frame visual confirmation.
    """

    CONFIRM_BUFFER = 3
    CONFIRM_STOP_HITS = 2
    LOST_LIMIT = 4
    CONFIRM_ATTEMPT_LIMIT = 8
    # Break an endless centering spin: if CONFIRM keeps issuing in-place turns
    # (VLM bbox missing or stuck at the frame edge), force a forward step after
    # this many consecutive turns so we actually close distance on the object.
    CONFIRM_MAX_TURN_STREAK = 3

    def __init__(self):
        self.state = ControlState()
        self.controller = Controller()
        self.stop_verifier = StopVerifier()
        self._bbox_hist: Deque[List[float]] = deque(maxlen=BBOX_SMOOTH_WINDOW)

    def reset_for_goal(self) -> None:
        self.state = ControlState()
        self.stop_verifier.last_debug = {}
        self._bbox_hist.clear()

    def _smoothed_target_bbox(
        self,
        report: Optional[PerceptionReport],
        goal: Optional[GoalNode],
        fresh_report: bool,
    ) -> Optional[List[float]]:
        """Update the rolling bbox history from this fresh frame and return the
        per-coordinate median (damps VLM localization jitter)."""
        if fresh_report:
            obj = target_object(report, goal)
            raw = obj.get("bbox") if obj else None
            if raw and len(raw) == 4:
                try:
                    self._bbox_hist.append([float(v) for v in raw])
                except (TypeError, ValueError):
                    pass
        return median_bbox(list(self._bbox_hist))

    def act(
        self,
        *,
        step_id: int,
        target: NavigationTarget,
        report: Optional[PerceptionReport],
        goal: Optional[GoalNode],
        fresh_report: bool = True,
        needs_verify: bool = False,
        on_verify_success: Optional[Callable[[], None]] = None,
        on_verify_failure: Optional[Callable[[], None]] = None,
        arrived: bool = False,
    ) -> Dict[str, Any]:
        if fresh_report:
            self._observe_target(step_id, target, report, goal)
        else:
            self.state.state_steps += 1
        smoothed_bbox = self._smoothed_target_bbox(report, goal, fresh_report)
        mode = self.state.mode
        visible = self.state.visible_window[-1:] == [True]

        if mode == GOATState.STOP:
            return self.controller.stop_action()

        if mode == GOATState.SEARCH:
            if fresh_report and visible and step_id >= self.state.confirm_cooldown_until_step:
                self._enter(GOATState.CONFIRM, "target_visible")
                return self._confirm_step(report, target, goal, arrived, smoothed_bbox)
            return self.controller.search_action(target)

        # mode == CONFIRM
        if self.state.target_lost_steps >= self.LOST_LIMIT:
            if needs_verify and not self.state.verify_committed and self.state.confirm_attempts >= 1:
                self._fire_verify(on_verify_failure)
            self._enter(GOATState.SEARCH, "target_lost")
            return self.controller.search_action(target)
        if fresh_report:
            self.state.confirm_attempts += 1
            can_stop = self.stop_verifier.can_stop(
                report, goal, arrived=arrived, bbox_override=smoothed_bbox,
            )
            self.state.confirm_buffer.append(can_stop)
            self.state.confirm_buffer = self.state.confirm_buffer[-self.CONFIRM_BUFFER:]
        else:
            self.stop_verifier.last_debug = {
                "can_stop": False,
                "reason": "awaiting_fresh_vlm",
                "fresh_report": False,
            }
        hits = sum(1 for item in self.state.confirm_buffer if item)
        # Low-confidence (VERIFY) targets must clear a stricter, full-buffer bar
        # before we trust them; confident targets keep the normal 2/3 STOP.
        pending_verify = bool(needs_verify and not self.state.verify_committed)
        required = self.CONFIRM_BUFFER if pending_verify else self.CONFIRM_STOP_HITS
        if hits >= required:
            if pending_verify:
                self.state.verify_committed = True
                self._fire_verify(on_verify_success)
            self._enter(GOATState.STOP, "multi_frame_verified")
            return self.controller.stop_action()
        if self.state.confirm_attempts >= self.CONFIRM_ATTEMPT_LIMIT:
            self.state.confirm_cooldown_until_step = step_id + 4
            if pending_verify:
                self._fire_verify(on_verify_failure)
            self._enter(GOATState.SEARCH, "confirm_failed")
            return self.controller.search_action(target)
        return self._confirm_step(report, target, goal, arrived, smoothed_bbox)

    def _confirm_step(
        self,
        report: Optional[PerceptionReport],
        target: NavigationTarget,
        goal: Optional[GoalNode],
        arrived: bool,
        smoothed_bbox: Optional[List[float]],
    ) -> Dict[str, Any]:
        """Run the centering/approach controller, but never let it spin forever.

        When the VLM bbox is missing or pinned to the frame edge the raw
        controller keeps issuing in-place turns and the agent never advances (the
        v18 rack stalled ~3 m out doing exactly this). After a few consecutive
        turns we override with a forward step so we keep closing distance; the
        next fresh detection re-centers. Forcing forward is suppressed once
        ``arrived`` so we can hold and verify at the object.
        """
        action = self.controller.confirm_action(
            report, target, goal, arrived=arrived, bbox_override=smoothed_bbox,
        )
        if str(action.get("action", "")) in ("turn_left", "turn_right"):
            self.state.confirm_turn_streak += 1
            if not arrived and self.state.confirm_turn_streak > self.CONFIRM_MAX_TURN_STREAK:
                self.state.confirm_turn_streak = 0
                return {"action": "move_forward", "reason": "confirm_force_forward"}
        else:
            self.state.confirm_turn_streak = 0
        return action

    @staticmethod
    def _fire_verify(callback: Optional[Callable[[], None]]) -> None:
        if callback is not None:
            callback()

    def _observe_target(
        self,
        step_id: int,
        target: NavigationTarget,
        report: Optional[PerceptionReport],
        goal: Optional[GoalNode],
    ) -> None:
        self.state.state_steps += 1
        obj = target_object(report, goal)
        visible = obj is not None
        track_key = self._track_key(target, obj, goal) if visible else None
        if visible and self.state.active_track_key is not None and track_key != self.state.active_track_key:
            self._clear_confirm()
        if visible and track_key is not None:
            self.state.active_track_key = track_key
        self.state.visible_window.append(visible)
        self.state.visible_window = self.state.visible_window[-5:]
        if visible:
            self.state.last_goal_seen_step = step_id
            self.state.target_lost_steps = 0
            self.state.active_anchor_waypoint_id = target.node_id
            if target.source_intent and target.source_intent.intent_type == "object":
                self.state.active_object_node_id = target.structure_node_id or (str(target.evidence_refs[0]) if target.evidence_refs else None)
        else:
            self.state.target_lost_steps += 1

    def _enter(self, mode: GOATState, reason: str) -> None:
        if self.state.mode == mode:
            return
        self.state.mode = mode
        self.state.transition_reason = reason
        self.state.state_steps = 0
        if mode != GOATState.CONFIRM:
            self._clear_confirm()

    def _clear_confirm(self) -> None:
        self.state.confirm_buffer.clear()
        self.state.confirm_attempts = 0
        self.state.verify_committed = False
        self.state.confirm_turn_streak = 0

    def debug(self) -> Dict[str, Any]:
        confirm_hits = sum(1 for item in self.state.confirm_buffer if item)
        return {
            "state": self.state.mode.value,
            "transition_reason": self.state.transition_reason,
            "state_steps": self.state.state_steps,
            "visible_window": list(self.state.visible_window),
            "confirm_buffer": list(self.state.confirm_buffer),
            # Legacy keys kept so existing trace columns stay populated.
            "verify_hits": confirm_hits,
            "verify_attempts": self.state.confirm_attempts,
            "verify_committed": bool(self.state.verify_committed),
            "track_hits": sum(1 for item in self.state.visible_window[-3:] if item),
            "target_lost_steps": self.state.target_lost_steps,
            "active_object_node_id": self.state.active_object_node_id,
            "active_anchor_waypoint_id": self.state.active_anchor_waypoint_id,
            "active_track_key": self.state.active_track_key,
        }

    def _track_key(
        self,
        target: NavigationTarget,
        obj: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> str:
        goal_id = str(getattr(goal, "target_object", "") or "")
        label = str((obj or {}).get("label", "")).lower().strip()
        return "|".join([
            goal_id,
            str(target.structure_node_id or ""),
            str(target.node_id or ""),
            label,
        ])


def target_object(report: Optional[PerceptionReport], goal: Optional[GoalNode]) -> Optional[Dict[str, Any]]:
    if report is None or goal is None:
        return None
    if not bool(getattr(report, "is_full", True)):
        return None
    candidates = []
    for obj in report.objects:
        data = obj.to_dict()
        relevance = GoalRelevanceScorer.label_relevance(str(data.get("label", "")), goal)
        if relevance < 0.65:
            continue
        if not GoalEvidenceFilter.is_usable_detection(data, goal):
            continue
        candidates.append((relevance, float(data.get("confidence", 0.0) or 0.0), data))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]
    return None
