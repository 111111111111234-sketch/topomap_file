"""Controller and visual stop verification for clean Agent1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from conftopo.agents.goat_agent1_memory import (
    GoalEvidenceFilter,
    GoalRelevanceScorer,
    bbox_center_x,
)
from conftopo.agents.goat_agent1_types import GOATState, NavigationTarget
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.perception_report import PerceptionReport


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
    ) -> Dict[str, Any]:
        """Center the visible target, close distance, hold still when near."""
        obj = target_object(report, goal)
        if obj is None:
            return self.search_action(target)
        center = bbox_center_x(obj.get("bbox"))
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
        range_bin = str(obj.get("range_bin", "")).lower()
        if range_bin in {"very_near", "close", "near"}:
            return {"action": "hold_and_verify", "reason": "confirm_hold_centered"}
        return {"action": "move_forward", "reason": "confirm_approach_target"}

    def stop_action(self) -> Dict[str, Any]:
        return {"action": "stop", "reason": "verified_goal"}


class StopVerifier:
    """Single, deliberately conservative STOP gate.

    STOP requires explicit visual confirmation only: goal visible, high task
    relevance, near range, and roughly centered. Multi-frame agreement is
    enforced by the state machine's confirm buffer, not here. No implicit
    distance / anchor / bbox-trend signals -- we would rather not stop than
    stop on weak evidence.
    """

    def __init__(self):
        self.last_debug: Dict[str, Any] = {}

    def can_stop(self, report: Optional[PerceptionReport], goal: Optional[GoalNode]) -> bool:
        obj = target_object(report, goal)
        if report is None or goal is None or obj is None:
            self.last_debug = {"can_stop": False, "reason": "missing_goal_or_target"}
            return False
        relevance = GoalRelevanceScorer.label_relevance(str(obj.get("label", "")), goal)
        range_bin = str(obj.get("range_bin", "")).lower()
        bbox_conf = str(obj.get("bbox_confidence", "")).lower()
        center = bbox_center_x(obj.get("bbox"))
        near = range_bin in {"very_near", "close", "near"}
        center_ok = center is not None and 0.30 <= center <= 0.70
        ok = (
            bool(report.goal_visible)
            and relevance >= 0.70
            and near
            and center_ok
            and bbox_conf != "low"
        )
        self.last_debug = {
            "can_stop": bool(ok),
            "goal_visible": bool(report.goal_visible),
            "relevance": round(float(relevance), 4),
            "range_bin": range_bin,
            "bbox_center": None if center is None else round(float(center), 4),
            "bbox_confidence": bbox_conf,
            "near": bool(near),
            "center_ok": bool(center_ok),
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

    def __init__(self):
        self.state = ControlState()
        self.controller = Controller()
        self.stop_verifier = StopVerifier()

    def reset_for_goal(self) -> None:
        self.state = ControlState()
        self.stop_verifier.last_debug = {}

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
    ) -> Dict[str, Any]:
        if fresh_report:
            self._observe_target(step_id, target, report, goal)
        else:
            self.state.state_steps += 1
        mode = self.state.mode
        visible = self.state.visible_window[-1:] == [True]

        if mode == GOATState.STOP:
            return self.controller.stop_action()

        if mode == GOATState.SEARCH:
            if fresh_report and visible and step_id >= self.state.confirm_cooldown_until_step:
                self._enter(GOATState.CONFIRM, "target_visible")
                return self.controller.confirm_action(report, target, goal)
            return self.controller.search_action(target)

        # mode == CONFIRM
        if self.state.target_lost_steps >= self.LOST_LIMIT:
            if needs_verify and not self.state.verify_committed and self.state.confirm_attempts >= 1:
                self._fire_verify(on_verify_failure)
            self._enter(GOATState.SEARCH, "target_lost")
            return self.controller.search_action(target)
        if fresh_report:
            self.state.confirm_attempts += 1
            can_stop = self.stop_verifier.can_stop(report, goal)
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
        return self.controller.confirm_action(report, target, goal)

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
