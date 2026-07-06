"""Single GOAT agent with an explicit perception-memory-planning loop.

This file is intentionally a clean first pass.  It keeps GOAT as one agent,
but separates the internal responsibilities so the step loop reads like the
method description:

observe -> CLIP -> optional VLM -> memory -> proposals -> planner -> state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from conftopo.agents.base_agent import ConfTopoBaseAgent
from conftopo.config import ConfTopoConfig
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.instruction_graph import GoalNode, InstructionGraph
from conftopo.perception.light_perceiver import LightPerceiver
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import Qwen3VLBackend
from conftopo.perception.vlm_perceiver import VLMPerceiver


class GOATState(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    APPROACH = "APPROACH"
    VERIFY_STOP = "VERIFY_STOP"
    STOP = "STOP"


@dataclass
class AgentPose:
    position: np.ndarray
    heading: float = 0.0


@dataclass
class ClipHint:
    room_label: str = "unknown"
    room_confidence: float = 0.0
    room_scores: List[Any] = field(default_factory=list)
    goal_scores: List[Any] = field(default_factory=list)
    landmark_scores: List[Any] = field(default_factory=list)
    best_goal_sim: float = 0.0
    best_landmark_sim: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def goal_candidate(self) -> bool:
        return self.best_goal_sim >= 0.35


@dataclass
class NavigationTarget:
    node_id: Optional[str]
    position: Optional[np.ndarray]
    target_type: str = "none"
    reason: str = ""
    source_intent: Optional["SearchIntent"] = None
    structure_node_id: Optional[str] = None
    evidence_refs: List[Any] = field(default_factory=list)
    score: float = 0.0


@dataclass
class SearchIntent:
    intent_type: str
    target_label: str = ""
    structure_node_id: Optional[str] = None
    preferred_anchor_waypoint_id: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    score: float = 0.0
    reason: str = ""
    preferred_room: Optional[str] = None
    related_landmarks: List[str] = field(default_factory=list)
    evidence_refs: List[Any] = field(default_factory=list)


@dataclass
class StructureProposal:
    proposal_type: str
    target_label: str
    structure_node_id: Optional[str]
    preferred_anchor_waypoint_id: Optional[str]
    score: float
    reason: str
    evidence_refs: List[Any] = field(default_factory=list)
    preferred_room: Optional[str] = None
    related_landmarks: List[str] = field(default_factory=list)


@dataclass
class NavigationProposal:
    target_node_id: Optional[str]
    target_position: Optional[np.ndarray]
    target_type: str
    score: float
    reason: str
    source_intent: SearchIntent
    structure_node_id: Optional[str] = None
    evidence_refs: List[Any] = field(default_factory=list)


@dataclass
class AgentState:
    mode: GOATState = GOATState.SEARCH
    transition_reason: str = "init"
    verify_hits: int = 0
    verify_attempts: int = 0
    verify_cooldown_until_step: int = -1
    verify_window: List[bool] = field(default_factory=list)
    target_visible_window: List[bool] = field(default_factory=list)
    target_area_window: List[float] = field(default_factory=list)
    verify_area_window: List[float] = field(default_factory=list)
    target_area_trend: float = 0.0
    track_hits: int = 0
    target_lost_steps: int = 0
    state_steps: int = 0
    last_goal_seen_step: int = -1
    active_object_node_id: Optional[str] = None
    stuck_steps: int = 0
    stuck_recoveries: int = 0
    stuck_target_node_id: Optional[str] = None
    last_motion_delta: float = 0.0
    last_stuck_reason: str = ""


@dataclass
class Hypothesis:
    hypothesis_id: str
    label: str
    kind: str
    confidence: float
    position: np.ndarray
    anchor_waypoint_id: Optional[str]
    source: str
    first_seen_step: int
    last_seen_step: int
    seen_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decayed_confidence(self, current_step: int, decay: float) -> float:
        age = max(0, current_step - self.last_seen_step)
        return float(self.confidence * (decay ** age))


@dataclass
class ObjectMemoryEntry:
    object_id: str
    node_id: str
    label: str
    confidence: float
    attributes: Dict[str, Any]
    bbox_history: List[Any]
    anchor_waypoint_id: str
    anchor_waypoint_position: np.ndarray
    estimated_object_position: np.ndarray
    observed_from: List[np.ndarray]
    first_seen_step: int
    last_seen_step: int
    observed_from_waypoint_ids: List[str] = field(default_factory=list)
    best_anchor_waypoint_id: Optional[str] = None
    range_bin_history: List[str] = field(default_factory=list)
    bearing_history: List[str] = field(default_factory=list)
    room_probs: Dict[str, float] = field(default_factory=dict)
    position_uncertainty: float = 1.0
    seen_count: int = 1
    failed_count: int = 0
    unreachable_count: int = 0

    @property
    def latest_bbox(self) -> Any:
        return self.bbox_history[-1] if self.bbox_history else None


class GoalManager:
    """Owns the current GOAT goal and GoalGraph bridge."""

    def __init__(self, agent: "GOATAgent"):
        self.agent = agent

    def current_goal(self) -> Optional[GoalNode]:
        if self.agent.instruction_graph is None:
            return None
        goal = self.agent.instruction_graph.get_current_goal()
        return goal if isinstance(goal, GoalNode) else None

    def set_goal(self, goal: GoalNode) -> None:
        if self.agent.instruction_graph is None:
            self.agent.instruction_graph = InstructionGraph(
                goal_type="object_goal",
                goal_nodes=[goal],
            )
        else:
            self.agent.instruction_graph.set_current_goal(goal)


class PerceptionModule:
    """Light CLIP every step, VLM only when useful."""

    def __init__(self, config: ConfTopoConfig):
        self.config = config
        self.clip = LightPerceiver(room_labels=config.perception.room_labels)
        self.vlm: Optional[VLMPerceiver] = None
        if config.perception.backend == "vlm":
            self.vlm = VLMPerceiver(
                Qwen3VLBackend(
                    api_base=config.perception.vlm_api_base,
                    model=config.perception.vlm_model,
                    timeout=config.perception.vlm_timeout,
                )
            )
        self.last_vlm_step: int = -1
        self.last_vlm_reason: str = ""

    def configure_goal(self, goal: GoalNode) -> None:
        self.clip.set_goal_labels(
            labels=[goal.target_object],
            embeddings=(
                goal.target_embedding[np.newaxis, :]
                if goal.target_embedding is not None
                else None
            ),
        )
        if goal.landmarks:
            self.clip.set_landmark_labels(
                labels=goal.landmarks,
                embeddings=goal.landmark_embeddings,
            )

    def run_clip(self, rgb_embed: Optional[np.ndarray], goal: Optional[GoalNode]) -> ClipHint:
        if rgb_embed is None:
            return ClipHint()
        report = self.clip.perceive(rgb_embed)
        return ClipHint(
            room_label=str(report.get("room_label", "unknown")),
            room_confidence=float(report.get("room_confidence", 0.0)),
            room_scores=list(report.get("room_scores", [])),
            goal_scores=list(report.get("goal_scores", [])),
            landmark_scores=list(report.get("landmark_scores", [])),
            best_goal_sim=float(report.get("best_goal_sim", 0.0)),
            best_landmark_sim=float(report.get("best_landmark_sim", 0.0)),
            raw=report,
        )

    def should_trigger_vlm(
        self,
        *,
        clip_hint: ClipHint,
        goal: Optional[GoalNode],
        state: AgentState,
        step_id: int,
    ) -> bool:
        if self.vlm is None or goal is None:
            self.last_vlm_reason = "no_vlm"
            return False
        if state.mode in (GOATState.TRACK, GOATState.APPROACH, GOATState.VERIFY_STOP):
            self.last_vlm_reason = "state_refresh"
            return step_id - self.last_vlm_step >= 2
        if clip_hint.goal_candidate:
            self.last_vlm_reason = "clip_goal_candidate"
            return True

        if state.mode == GOATState.SEARCH:
            if self.last_vlm_step < 0:
                self.last_vlm_reason = "initial_search_scan"
                return True

            if step_id <= 30 and step_id - self.last_vlm_step >= 5:
                self.last_vlm_reason = "early_search_scan"
                return True

            if step_id - self.last_vlm_step >= 8:
                self.last_vlm_reason = "periodic_search"
                return True

        self.last_vlm_reason = "skip"
        return False

    def run_vlm(
        self,
        rgb: Optional[np.ndarray],
        goal: Optional[GoalNode],
        *,
        step_id: int,
        state: AgentState,
        clip_hint: ClipHint,
    ) -> Optional[Dict[str, Any]]:
        if self.vlm is None or rgb is None or goal is None:
            return None
        goal_text = _goal_text(goal)
        mode = "confirm" if state.mode != GOATState.SEARCH else "explore"
        context = (
            f"state={state.mode.value}; "
            f"clip_room={clip_hint.room_label}; "
            f"clip_goal_sim={clip_hint.best_goal_sim:.3f}"
        )
        try:
            report = self.vlm.perceive(
                np.asarray(rgb),
                goal_text,
                step_id=step_id,
                context=context,
                mode=mode,
            )
        except Exception as exc:
            return {
                "fresh": False,
                "step": step_id,
                "mode": mode,
                "trigger_reason": f"vlm_error:{type(exc).__name__}",
                "goal_visible": False,
                "stop_candidate": False,
                "objects": [],
                "error": str(exc),
            }

        self.last_vlm_step = step_id
        return _vlm_report_to_dict(report, reason=self.last_vlm_reason, mode=mode)


class HypothesisPool:
    """Short-term weak evidence from CLIP or non-confirmed VLM observations."""

    def __init__(
        self,
        *,
        max_size: int = 30,
        ttl_steps: int = 12,
        merge_radius: float = 2.0,
        decay: float = 0.85,
        min_confidence: float = 0.12,
    ):
        self.max_size = max_size
        self.ttl_steps = ttl_steps
        self.merge_radius = merge_radius
        self.decay = decay
        self.min_confidence = min_confidence
        self._items: Dict[str, Hypothesis] = {}
        self._counter = 0

    def clear(self) -> None:
        self._items.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[Hypothesis]:
        return list(self._items.values())

    def add_or_update(
        self,
        *,
        label: str,
        kind: str,
        confidence: float,
        pose: AgentPose,
        anchor_waypoint_id: Optional[str],
        source: str,
        step_id: int,
        metadata: Optional[Dict[str, Any]] = None,
        evidence_position: Optional[np.ndarray] = None,
    ) -> Optional[Hypothesis]:
        label = str(label).strip()
        if not label or confidence < self.min_confidence:
            return None

        observed_position = (
            np.asarray(evidence_position, dtype=np.float32).copy()
            if evidence_position is not None
            else pose.position.copy()
        )
        metadata = dict(metadata or {})
        metadata_record: Dict[str, Any] = {}
        self._merge_metadata(metadata_record, metadata)
        matched = self._find_match(label, kind, observed_position)
        if matched is not None:
            old_weight = max(1, matched.seen_count)
            new_conf = max(float(confidence), matched.decayed_confidence(step_id, self.decay))
            matched.position = (
                matched.position * old_weight + observed_position
            ) / float(old_weight + 1)
            repeat_bonus = min(0.2, 0.03 * matched.seen_count)
            matched.confidence = min(1.0, new_conf + repeat_bonus)
            matched.anchor_waypoint_id = anchor_waypoint_id or matched.anchor_waypoint_id
            matched.source = source
            matched.last_seen_step = step_id
            matched.seen_count += 1
            self._merge_metadata(matched.metadata, metadata)
            return matched

        self._counter += 1
        hypothesis = Hypothesis(
            hypothesis_id=f"hyp_{self._counter}",
            label=label,
            kind=kind,
            confidence=float(confidence),
            position=observed_position,
            anchor_waypoint_id=anchor_waypoint_id,
            source=source,
            first_seen_step=step_id,
            last_seen_step=step_id,
            metadata=metadata_record,
        )
        self._items[hypothesis.hypothesis_id] = hypothesis
        self._trim(step_id)
        return hypothesis

    def decay_and_prune(self, current_step: int) -> None:
        for hyp in self._items.values():
            if current_step > hyp.last_seen_step:
                hyp.confidence = float(hyp.confidence * self.decay)
        self._trim(current_step)

    def active_for_goal(
        self,
        goal: Optional[GoalNode],
        current_step: int,
    ) -> List[Hypothesis]:
        if goal is None:
            return []
        goal_label = str(goal.target_object or "").lower().strip()
        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        landmarks = {str(label).lower().strip() for label in goal.landmarks or []}
        out = []
        for hyp in self._items.values():
            if current_step - hyp.last_seen_step > self.ttl_steps:
                continue
            if hyp.confidence < self.min_confidence:
                continue
            label = hyp.label.lower().strip()
            if hyp.kind == "object" and label == goal_label:
                out.append(hyp)
            elif hyp.kind == "room" and label in room_priors:
                out.append(hyp)
            elif hyp.kind == "landmark" and label in landmarks:
                out.append(hyp)
        out.sort(key=lambda hyp: (hyp.confidence, hyp.seen_count), reverse=True)
        return out

    def suppress_confirmed(self, *, label: str, position: np.ndarray, step_id: int) -> None:
        label_key = str(label).lower().strip()
        for hyp_id, hyp in list(self._items.items()):
            if hyp.kind != "object" or hyp.label.lower().strip() != label_key:
                continue
            if _distance(hyp.position, position) <= self.merge_radius * 1.5:
                self._items.pop(hyp_id, None)
        self._trim(step_id)

    def to_debug(self, current_step: int) -> List[Dict[str, Any]]:
        return [
            {
                "id": hyp.hypothesis_id,
                "label": hyp.label,
                "kind": hyp.kind,
                "confidence": round(float(hyp.confidence), 4),
                "age": int(current_step - hyp.last_seen_step),
                "seen_count": hyp.seen_count,
                "source": hyp.source,
                "anchor_waypoint_id": hyp.anchor_waypoint_id,
            }
            for hyp in sorted(
                self._items.values(),
                key=lambda item: item.confidence,
                reverse=True,
            )[:8]
        ]

    def _find_match(
        self,
        label: str,
        kind: str,
        position: np.ndarray,
    ) -> Optional[Hypothesis]:
        label_key = label.lower().strip()
        best: Optional[Hypothesis] = None
        best_dist = float("inf")
        for hyp in self._items.values():
            if hyp.label.lower().strip() != label_key or hyp.kind != kind:
                continue
            dist = _distance(hyp.position, position)
            if dist < self.merge_radius and dist < best_dist:
                best = hyp
                best_dist = dist
        return best

    @staticmethod
    def _merge_metadata(target: Dict[str, Any], update: Dict[str, Any]) -> None:
        old_room_probs = target.get("room_probs") or {}
        target.update(update)
        waypoint_id = update.get("observed_from_waypoint_id")
        if waypoint_id:
            ids = list(target.get("observed_from_waypoint_ids") or [])
            if waypoint_id not in ids:
                ids.append(waypoint_id)
            target["observed_from_waypoint_ids"] = ids[-8:]

        for key, hist_key in (
            ("bbox", "bbox_history"),
            ("range_bin", "range_bin_history"),
            ("bearing", "bearing_history"),
        ):
            value = update.get(key)
            if value in (None, ""):
                continue
            history = list(target.get(hist_key) or [])
            history.append(value)
            target[hist_key] = history[-8:]

        new_room_probs = update.get("room_probs") or {}
        if old_room_probs or new_room_probs:
            target["room_probs"] = _merge_score_dicts(
                _scores_to_dict(old_room_probs),
                _scores_to_dict(new_room_probs),
            )

        target["ttl"] = update.get("ttl", target.get("ttl"))

    def _trim(self, current_step: int) -> None:
        stale_ids = [
            hyp_id
            for hyp_id, hyp in self._items.items()
            if (
                current_step - hyp.last_seen_step > self.ttl_steps
                or hyp.confidence < self.min_confidence
            )
        ]
        for hyp_id in stale_ids:
            self._items.pop(hyp_id, None)

        if len(self._items) <= self.max_size:
            return
        keep = sorted(
            self._items.values(),
            key=lambda hyp: (hyp.confidence, hyp.last_seen_step, hyp.seen_count),
            reverse=True,
        )[:self.max_size]
        self._items = {hyp.hypothesis_id: hyp for hyp in keep}


class ObjectMemoryPool:
    """Confirmed object memory backed by OBJECT nodes in the TopoMap."""

    def __init__(
        self,
        topo_map: DynamicTopoMap,
        *,
        merge_radius: float = 2.0,
        max_bbox_history: int = 8,
        max_observed_from: int = 8,
    ):
        self.topo_map = topo_map
        self.merge_radius = merge_radius
        self.max_bbox_history = max_bbox_history
        self.max_observed_from = max_observed_from
        self._entries: Dict[str, ObjectMemoryEntry] = {}

    def clear_runtime(self) -> None:
        """Keep long-term object entries across GOAT goals."""
        return

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> List[ObjectMemoryEntry]:
        return list(self._entries.values())

    def get(self, object_id: str) -> Optional[ObjectMemoryEntry]:
        return self._entries.get(object_id)

    def goal_entries(self, goal: Optional[GoalNode]) -> List[ObjectMemoryEntry]:
        if goal is None:
            return []
        goal_label = str(goal.target_object or "").lower().strip()
        out = [
            entry
            for entry in self._entries.values()
            if entry.label.lower().strip() == goal_label
        ]
        out.sort(key=lambda entry: (entry.confidence, entry.seen_count), reverse=True)
        return out

    def matching_entry(
        self,
        *,
        label: str,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        room_probs: Optional[Dict[str, float]] = None,
    ) -> Optional[ObjectMemoryEntry]:
        node = self._find_matching_node(
            label,
            _estimate_object_position(pose, obj),
            waypoint_id=waypoint_id,
            obj=obj,
            room_probs=dict(room_probs or {}),
        )
        if node is None:
            return None
        return self._entries.get(node.node_id)

    def upsert_confirmed(
        self,
        *,
        label: str,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        room_probs: Optional[Dict[str, float]] = None,
    ) -> ObjectMemoryEntry:
        estimated_position = _estimate_object_position(pose, obj)
        room_probs = dict(room_probs or {})
        node = self._find_matching_node(
            label,
            estimated_position,
            waypoint_id=waypoint_id,
            obj=obj,
            room_probs=room_probs,
        )
        if node is None:
            node_id = self._create_object_node(
                label,
                confidence,
                obj,
                pose,
                waypoint_id,
                estimated_position,
                room_probs,
            )
        else:
            node_id = node.node_id
            self._update_object_node(
                node,
                confidence,
                obj,
                pose,
                waypoint_id,
                estimated_position,
                room_probs,
            )

        node = self.topo_map.get_node(node_id)
        if node is None:
            raise RuntimeError(f"missing object node after upsert: {node_id}")

        entry = self._entries.get(node_id)
        bbox = obj.get("bbox")
        attrs = dict(obj)
        if entry is None:
            entry = ObjectMemoryEntry(
                object_id=node_id,
                node_id=node_id,
                label=label,
                confidence=float(node.confidence),
                attributes=attrs,
                bbox_history=[bbox] if bbox is not None else [],
                anchor_waypoint_id=waypoint_id,
                anchor_waypoint_position=pose.position.copy(),
                estimated_object_position=node.position.copy(),
                observed_from=[pose.position.copy()],
                first_seen_step=self.topo_map.current_step,
                last_seen_step=self.topo_map.current_step,
                observed_from_waypoint_ids=[waypoint_id],
                best_anchor_waypoint_id=waypoint_id,
                range_bin_history=[str(obj.get("range_bin", "unknown"))],
                bearing_history=[str(obj.get("bearing", ""))],
                room_probs=room_probs,
                position_uncertainty=_object_position_uncertainty(obj, seen_count=1),
            )
            self._entries[node_id] = entry
        else:
            entry.confidence = float(node.confidence)
            entry.attributes.update(attrs)
            if bbox is not None:
                entry.bbox_history.append(bbox)
                entry.bbox_history = entry.bbox_history[-self.max_bbox_history:]
            entry.anchor_waypoint_id = waypoint_id
            entry.anchor_waypoint_position = pose.position.copy()
            entry.best_anchor_waypoint_id = waypoint_id
            entry.estimated_object_position = node.position.copy()
            entry.observed_from.append(pose.position.copy())
            entry.observed_from = entry.observed_from[-self.max_observed_from:]
            if waypoint_id not in entry.observed_from_waypoint_ids:
                entry.observed_from_waypoint_ids.append(waypoint_id)
            entry.observed_from_waypoint_ids = (
                entry.observed_from_waypoint_ids[-self.max_observed_from:]
            )
            entry.range_bin_history.append(str(obj.get("range_bin", "unknown")))
            entry.range_bin_history = entry.range_bin_history[-self.max_bbox_history:]
            bearing = str(obj.get("bearing", ""))
            if bearing:
                entry.bearing_history.append(bearing)
                entry.bearing_history = entry.bearing_history[-self.max_bbox_history:]
            entry.room_probs = _merge_score_dicts(entry.room_probs, room_probs)
            entry.last_seen_step = self.topo_map.current_step
            entry.seen_count += 1
            entry.position_uncertainty = _object_position_uncertainty(
                obj,
                seen_count=entry.seen_count,
            )

        self._sync_node_from_entry(node, entry)
        return entry

    def mark_navigation_failed(
        self,
        object_id: str,
        *,
        unreachable: bool = False,
    ) -> None:
        entry = self._entries.get(object_id)
        if entry is None:
            return

        entry.failed_count += 1
        if unreachable:
            entry.unreachable_count += 1
        entry.attributes["failed_count"] = entry.failed_count
        entry.attributes["unreachable_count"] = entry.unreachable_count

        node = self.topo_map.get_node(entry.node_id)
        if node is not None:
            node.attributes["failed_count"] = entry.failed_count
            node.attributes["unreachable_count"] = entry.unreachable_count
            node.attributes["last_failed_step"] = self.topo_map.current_step
            node.attributes["memory_state"] = (
                "unreachable" if unreachable else "navigation_failed"
            )

    def to_debug(self) -> List[Dict[str, Any]]:
        return [
            {
                "object_id": entry.object_id,
                "label": entry.label,
                "confidence": round(float(entry.confidence), 4),
                "seen_count": entry.seen_count,
                "anchor_waypoint_id": entry.anchor_waypoint_id,
                "best_anchor_waypoint_id": entry.best_anchor_waypoint_id,
                "observed_from_waypoint_ids": list(entry.observed_from_waypoint_ids),
                "estimated_object_position": entry.estimated_object_position.tolist(),
                "position_uncertainty": round(float(entry.position_uncertainty), 4),
                "last_seen_step": entry.last_seen_step,
                "latest_bbox": entry.latest_bbox,
                "failed_count": entry.failed_count,
                "unreachable_count": entry.unreachable_count,
            }
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (item.confidence, item.seen_count),
                reverse=True,
            )[:8]
        ]

    def _find_matching_node(
        self,
        label: str,
        position: np.ndarray,
        *,
        waypoint_id: str,
        obj: Dict[str, Any],
        room_probs: Dict[str, float],
    ) -> Optional[SemanticNode]:
        label_key = label.lower().strip()
        best = None
        best_score = -float("inf")
        waypoint = self.topo_map.get_node(waypoint_id)
        for node in self.topo_map.get_nodes_by_type(NodeType.OBJECT):
            if node.attributes.get("semantic_role") != "object":
                continue
            if node.label.lower().strip() != label_key:
                continue
            score = self._merge_score(
                node,
                position,
                waypoint_position=waypoint.position if waypoint is not None else None,
                obj=obj,
                room_probs=room_probs,
            )
            if score >= 0.75 and score > best_score:
                best = node
                best_score = score
        return best

    def _merge_score(
        self,
        node: SemanticNode,
        position: np.ndarray,
        *,
        waypoint_position: Optional[np.ndarray],
        obj: Dict[str, Any],
        room_probs: Dict[str, float],
    ) -> float:
        attrs = node.attributes
        estimated_dist = _distance(node.position, position)
        if estimated_dist > self.merge_radius * 2.0:
            return -float("inf")

        score = 0.0
        if estimated_dist <= self.merge_radius:
            score += 0.35
        elif estimated_dist <= self.merge_radius * 2.0:
            score += 0.15

        anchor_dist: Optional[float] = None
        anchor_position = attrs.get("anchor_waypoint_position")
        if waypoint_position is not None and isinstance(anchor_position, (list, tuple)):
            anchor_dist = _distance(np.asarray(anchor_position, dtype=np.float32), waypoint_position)
            if anchor_dist > self.merge_radius * 4.0 and estimated_dist > self.merge_radius:
                return -float("inf")
            if anchor_dist <= self.merge_radius * 2.0:
                score += 0.25
            elif anchor_dist <= self.merge_radius * 4.0:
                score += 0.10

        old_room = _dominant_label(attrs.get("room_probs", {}))
        new_room = _dominant_label(room_probs)
        if old_room and new_room:
            if old_room == new_room:
                score += 0.15
            else:
                if estimated_dist > self.merge_radius * 0.75:
                    return -float("inf")
                score -= 0.25

        range_bin = str(obj.get("range_bin", "")).lower()
        old_ranges = [str(item).lower() for item in attrs.get("range_bin_history", [])]
        if range_bin and range_bin in old_ranges[-3:]:
            score += 0.10

        bearing = str(obj.get("bearing", "")).lower()
        old_bearings = [str(item).lower() for item in attrs.get("bearing_history", [])]
        if bearing and bearing in old_bearings[-3:]:
            score += 0.10
        elif bearing in {"left", "right"} and _last_directional_bearing(old_bearings) in {"left", "right"}:
            if bearing != _last_directional_bearing(old_bearings) and estimated_dist > self.merge_radius * 0.75:
                score -= 0.15

        seen_count = int(attrs.get("seen_count", 1) or 1)
        score += min(0.10, 0.02 * seen_count)
        return score

    def _create_object_node(
        self,
        label: str,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
        room_probs: Dict[str, float],
    ) -> str:
        attrs = self._node_attrs(
            confidence,
            obj,
            pose,
            waypoint_id,
            estimated_position,
            room_probs,
            seen_count=1,
        )
        node_id = self.topo_map.add_or_update_object_memory_node(
            label=label,
            confidence=confidence,
            estimated_position=estimated_position.copy(),
            anchor_waypoint_id=waypoint_id,
            obj_attrs=attrs,
        )
        return node_id

    def _update_object_node(
        self,
        node: SemanticNode,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
        room_probs: Dict[str, float],
    ) -> None:
        seen_count = int(node.attributes.get("seen_count", 1)) + 1
        repeat_bonus = min(0.15, 0.02 * seen_count)
        old_weight = max(1, seen_count - 1)
        node.position = (
            node.position * old_weight + estimated_position
        ) / float(old_weight + 1)
        node.confidence = min(1.0, max(float(node.confidence), confidence) + repeat_bonus)
        node.step_id = self.topo_map.current_step
        node.attributes.update(
            self._node_attrs(
                confidence,
                obj,
                pose,
                waypoint_id,
                node.position,
                room_probs,
                seen_count=seen_count,
            )
        )
        self.topo_map.add_or_update_object_memory_node(
            label=node.label,
            confidence=node.confidence,
            estimated_position=node.position.copy(),
            anchor_waypoint_id=waypoint_id,
            obj_attrs=node.attributes,
            node_id=node.node_id,
        )

    def _node_attrs(
        self,
        confidence: float,
        obj: Dict[str, Any],
        pose: AgentPose,
        waypoint_id: str,
        estimated_position: np.ndarray,
        room_probs: Dict[str, float],
        *,
        seen_count: int,
    ) -> Dict[str, Any]:
        bbox = obj.get("bbox")
        range_bin = str(obj.get("range_bin", "unknown"))
        bearing = str(obj.get("bearing", ""))
        return {
            "layer": "structure",
            "semantic_role": "object",
            "memory_state": "confirmed",
            "source": "vlm",
            "confidence": float(confidence),
            "bbox": bbox,
            "bbox_history": [bbox] if bbox is not None else [],
            "range_bin": range_bin,
            "range_bin_history": [range_bin],
            "bearing": bearing,
            "bearing_history": [bearing] if bearing else [],
            "visibility": obj.get("visibility", "unknown"),
            "attributes": dict(obj),
            "anchor_waypoint_id": waypoint_id,
            "best_anchor_waypoint_id": waypoint_id,
            "anchor_waypoint_position": pose.position.copy().tolist(),
            "estimated_object_position": estimated_position.copy().tolist(),
            "position_uncertainty": _object_position_uncertainty(obj, seen_count=seen_count),
            "observed_from": pose.position.copy().tolist(),
            "observed_from_history": [pose.position.copy().tolist()],
            "observed_from_waypoint_ids": [waypoint_id],
            "room_probs": dict(room_probs),
            "last_seen_step": self.topo_map.current_step,
            "seen_count": seen_count,
            "failed_count": int(obj.get("failed_count", 0) or 0),
            "unreachable_count": int(obj.get("unreachable_count", 0) or 0),
        }

    def _sync_node_from_entry(
        self,
        node: SemanticNode,
        entry: ObjectMemoryEntry,
    ) -> None:
        node.attributes["bbox_history"] = list(entry.bbox_history)
        node.attributes["observed_from_history"] = [
            pos.tolist() for pos in entry.observed_from
        ]
        node.attributes["anchor_waypoint_id"] = entry.anchor_waypoint_id
        node.attributes["best_anchor_waypoint_id"] = (
            entry.best_anchor_waypoint_id or entry.anchor_waypoint_id
        )
        node.attributes["anchor_waypoint_position"] = (
            entry.anchor_waypoint_position.tolist()
        )
        node.attributes["estimated_object_position"] = (
            entry.estimated_object_position.tolist()
        )
        node.attributes["observed_from_waypoint_ids"] = list(
            entry.observed_from_waypoint_ids
        )
        node.attributes["range_bin_history"] = list(entry.range_bin_history)
        node.attributes["bearing_history"] = list(entry.bearing_history)
        node.attributes["room_probs"] = dict(entry.room_probs)
        node.attributes["position_uncertainty"] = float(entry.position_uncertainty)
        node.attributes["layer"] = "structure"
        node.attributes["semantic_role"] = "object"
        node.attributes["seen_count"] = entry.seen_count
        node.attributes["last_seen_step"] = entry.last_seen_step
        node.attributes["failed_count"] = entry.failed_count
        node.attributes["unreachable_count"] = entry.unreachable_count
        node.attributes["memory_state"] = "confirmed"
        node.position = entry.estimated_object_position.copy()
        node.confidence = max(float(node.confidence), float(entry.confidence))
        anchor_id = entry.best_anchor_waypoint_id or entry.anchor_waypoint_id
        self.topo_map.bind_object_to_anchor(node.node_id, anchor_id)
        for waypoint_id in entry.observed_from_waypoint_ids:
            self.topo_map.bind_object_observation(node.node_id, waypoint_id)


class MemoryManager:
    """Owns short-term hypotheses and the shared DynamicTopoMap."""

    def __init__(self, topo_map: DynamicTopoMap, config: ConfTopoConfig):
        self.topo_map = topo_map
        self.config = config
        self.hypothesis_pool = HypothesisPool()
        self.object_memory_pool = ObjectMemoryPool(topo_map)
        self.structure_layer: Dict[str, Any] = {}
        self.navigation_layer: Dict[str, Any] = {}
        self.current_waypoint_id: Optional[str] = None

    def reset_runtime(self) -> None:
        self.current_waypoint_id = None
        self.hypothesis_pool.clear()
        self.object_memory_pool.clear_runtime()

    def mark_object_navigation_failed(
        self,
        object_id: str,
        *,
        unreachable: bool = False,
    ) -> None:
        self.object_memory_pool.mark_navigation_failed(
            object_id,
            unreachable=unreachable,
        )

    def mark_navigation_target_failed(
        self,
        node_id: Optional[str],
        *,
        target_type: str,
        unreachable: bool = False,
        reason: str = "navigation_failed",
    ) -> bool:
        if node_id is None:
            return False
        ok = self.topo_map.mark_navigation_target_failed(
            node_id,
            target_type=target_type,
            unreachable=unreachable,
            reason=reason,
        )
        node = self.topo_map.get_node(node_id)
        self.navigation_layer["last_failed_target"] = {
            "node_id": node_id,
            "target_type": target_type,
            "failed_count": node.attributes.get("failed_count") if node else None,
            "unreachable_count": node.attributes.get("unreachable_count") if node else None,
            "reason": reason,
            "step": self.topo_map.current_step,
        }
        return ok

    def mark_navigation_target_reached(
        self,
        node_id: Optional[str],
        *,
        target_type: str,
        reason: str = "target_reached",
    ) -> bool:
        if node_id is None:
            return False
        node = self.topo_map.get_node(node_id)
        if node is None:
            return False
        if target_type in {"frontier", "candidate_waypoint"}:
            self.topo_map.consume_node(node_id, reason=reason)
        else:
            node.attributes["visit_count"] = int(node.attributes.get("visit_count", 0) or 0) + 1
            node.attributes["last_reached_step"] = self.topo_map.current_step
        self.navigation_layer["last_reached_target"] = {
            "node_id": node_id,
            "target_type": target_type,
            "reason": reason,
            "step": self.topo_map.current_step,
        }
        return True

    def begin_navigation_step(
        self,
        *,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
    ) -> str:
        self.topo_map.step()
        return self._add_or_update_waypoint(pose, rgb_embed)

    def update_structure_from_clip(
        self,
        *,
        clip_hint: ClipHint,
        goal: Optional[GoalNode],
        waypoint_id: str,
        rgb_embed: Optional[np.ndarray],
        pose: AgentPose,
    ) -> None:
        self._update_room(clip_hint, pose, rgb_embed, waypoint_id, goal)
        self.update_hypothesis(clip_hint, pose, waypoint_id)

    def update_structure_from_vlm(
        self,
        *,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
        pose: AgentPose,
        waypoint_id: str,
    ) -> None:
        if vlm_report is None:
            return
        self._update_room_from_vlm(vlm_report, waypoint_id, goal)
        self.update_weak_vlm_hypotheses(vlm_report, pose, goal, waypoint_id)
        self.update_confirmed_memory(vlm_report, pose, goal, waypoint_id)
        self.update_landmarks_and_portals(vlm_report, pose, goal, waypoint_id)

    def update_navigation_frontiers(
        self,
        *,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        waypoint_id: str,
    ) -> None:
        self._generate_frontiers(pose, rgb_embed, waypoint_id)

    def finalize_step_memory(self) -> None:
        self.hypothesis_pool.decay_and_prune(self.topo_map.current_step)
        self.topo_map.decay_all_confidences()

    def prune_far_frontiers(
        self,
        agent_pos: np.ndarray,
        keep_radius: float = 6.0,
    ) -> None:
        """Remove frontier nodes that are too far from agent position.
        
        Keeps frontiers within keep_radius to avoid unbounded accumulation
        while preserving exploration history in current area.
        """
        removed_count = 0
        frontiers_to_check = [
            node for node in self.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER)
        ]
        
        for frontier in frontiers_to_check:
            dist = float(np.linalg.norm(frontier.position - agent_pos))
            if dist > keep_radius:
                try:
                    self.topo_map.remove_node(frontier.node_id)
                    removed_count += 1
                except Exception:
                    pass  # Node might already be removed
        
        if removed_count > 0:
            self.navigation_layer["last_frontier_prune"] = {
                "step": self.topo_map.current_step,
                "removed_count": removed_count,
                "keep_radius": keep_radius,
            }

    def update_step_memory(
        self,
        *,
        clip_hint: ClipHint,
        vlm_report: Optional[Dict[str, Any]],
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        goal: Optional[GoalNode],
    ) -> None:
        waypoint_id = self.begin_navigation_step(pose=pose, rgb_embed=rgb_embed)
        self.update_structure_from_clip(
            clip_hint=clip_hint,
            goal=goal,
            waypoint_id=waypoint_id,
            rgb_embed=rgb_embed,
            pose=pose,
        )
        self.update_structure_from_vlm(
            vlm_report=vlm_report,
            goal=goal,
            pose=pose,
            waypoint_id=waypoint_id,
        )
        self.update_navigation_frontiers(
            pose=pose,
            rgb_embed=rgb_embed,
            waypoint_id=waypoint_id,
        )
        self.finalize_step_memory()
        
        # Prune far frontiers every 10 steps to avoid unbounded accumulation
        if self.topo_map.current_step % 10 == 0:
            self.prune_far_frontiers(pose.position, keep_radius=6.0)

    def update_hypothesis(
        self,
        clip_hint: ClipHint,
        pose: AgentPose,
        waypoint_id: Optional[str],
    ) -> None:
        label = _best_label(clip_hint.goal_scores)
        if label is not None and clip_hint.best_goal_sim >= 0.25:
            self.hypothesis_pool.add_or_update(
                label=label,
                kind="object",
                confidence=clip_hint.best_goal_sim,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_goal",
                step_id=self.topo_map.current_step,
                metadata={"scores": list(clip_hint.goal_scores[:3])},
            )

        room_label = str(clip_hint.room_label or "").strip()
        if room_label and room_label != "unknown" and clip_hint.room_confidence >= 0.35:
            self.hypothesis_pool.add_or_update(
                label=room_label,
                kind="room",
                confidence=clip_hint.room_confidence,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_room",
                step_id=self.topo_map.current_step,
            )

        landmark = _best_label(clip_hint.landmark_scores)
        if landmark is not None and clip_hint.best_landmark_sim >= 0.25:
            self.hypothesis_pool.add_or_update(
                label=landmark,
                kind="landmark",
                confidence=clip_hint.best_landmark_sim,
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="clip_landmark",
                step_id=self.topo_map.current_step,
                metadata={"scores": list(clip_hint.landmark_scores[:3])},
            )

    def update_weak_vlm_hypotheses(
        self,
        vlm_report: Dict[str, Any],
        pose: AgentPose,
        goal: Optional[GoalNode],
        waypoint_id: str,
    ) -> None:
        goal_label = str(getattr(goal, "target_object", "") or "").lower().strip()
        room_probs = self._waypoint_room_probs(waypoint_id)
        room_hint = _dominant_label(room_probs)
        for obj in vlm_report.get("objects", []) or []:
            label = str(obj.get("label", "")).strip()
            if not label:
                continue
            label_key = label.lower().strip()
            estimated_position = _estimate_object_position(pose, obj)
            goal_relevant = bool(goal_label and label_key == goal_label)
            self.hypothesis_pool.add_or_update(
                label=label,
                kind="object",
                confidence=float(obj.get("confidence", 0.0) or 0.0),
                pose=pose,
                anchor_waypoint_id=waypoint_id,
                source="goal_vlm" if goal_relevant else "weak_vlm",
                step_id=self.topo_map.current_step,
                metadata={
                    "bbox": obj.get("bbox"),
                    "range_bin": obj.get("range_bin", "unknown"),
                    "visibility": obj.get("visibility", "unknown"),
                    "bearing": obj.get("bearing", ""),
                    "bbox_confidence": obj.get("bbox_confidence", "unknown"),
                    "observed_from_waypoint_id": waypoint_id,
                    "observed_from_position": pose.position.copy().tolist(),
                    "estimated_object_position": estimated_position.copy().tolist(),
                    "room_hint": room_hint,
                    "room_probs": room_probs,
                    "goal_relevant": goal_relevant,
                    "goal_visible": bool(vlm_report.get("goal_visible", False)) and goal_relevant,
                    "goal_match_confidence": float(
                        vlm_report.get("goal_match_confidence", 0.0) or 0.0
                    ),
                    "ttl": self.hypothesis_pool.ttl_steps,
                },
                evidence_position=estimated_position,
            )

    def update_confirmed_memory(
        self,
        vlm_report: Dict[str, Any],
        pose: AgentPose,
        goal: Optional[GoalNode],
        waypoint_id: str,
    ) -> None:
        objects = vlm_report.get("objects", []) or []
        goal_label = str(getattr(goal, "target_object", "") or "").lower().strip()
        candidates: List[Dict[str, Any]] = []
        for obj in objects:
            label = str(obj.get("label", "")).lower().strip()
            if not label:
                continue
            confidence = float(obj.get("confidence", 0.0) or 0.0)
            if label == goal_label or confidence >= 0.72:
                candidates.append(dict(obj))
        if not candidates:
            return

        promoted_count = 0
        candidates.sort(
            key=lambda obj: (
                str(obj.get("label", "")).lower().strip() == goal_label,
                float(obj.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for candidate in candidates:
            if promoted_count >= 3:
                break
            if not self._should_promote_object_candidate(
                candidate,
                vlm_report=vlm_report,
                goal=goal,
                pose=pose,
                waypoint_id=waypoint_id,
            ):
                continue
            label = str(candidate.get("label") or "object")
            label_key = label.lower().strip()
            goal_relevant = bool(goal_label and label_key == goal_label)
            confidence = max(
                float(candidate.get("confidence", 0.0) or 0.0),
                float(vlm_report.get("goal_match_confidence", 0.0) or 0.0)
                if goal_relevant
                else 0.0,
                0.55,
            )
            entry = self.object_memory_pool.upsert_confirmed(
                label=label,
                confidence=confidence,
                obj=candidate,
                pose=pose,
                waypoint_id=waypoint_id,
                room_probs=self._waypoint_room_probs(waypoint_id),
            )
            self.hypothesis_pool.suppress_confirmed(
                label=label,
                position=_estimate_object_position(pose, candidate),
                step_id=self.topo_map.current_step,
            )
            self.structure_layer["last_confirmed_object_id"] = entry.object_id
            if goal_relevant:
                self.structure_layer["last_confirmed_goal_object_id"] = entry.object_id
            self._sync_object_viewing_candidate(entry)
            promoted_count += 1

    def _should_promote_object_candidate(
        self,
        obj: Dict[str, Any],
        *,
        vlm_report: Dict[str, Any],
        goal: Optional[GoalNode],
        pose: AgentPose,
        waypoint_id: str,
    ) -> bool:
        if not bool(vlm_report.get("fresh", False)):
            return False
        label = str(obj.get("label", "")).lower().strip()
        if not label:
            return False
        goal_label = str(getattr(goal, "target_object", "") or "").lower().strip()
        goal_relevant = bool(goal_label and label == goal_label)

        visibility = str(obj.get("visibility", "")).lower()
        range_bin = str(obj.get("range_bin", "")).lower()
        bbox = obj.get("bbox")
        bbox_area = _bbox_area(bbox)
        bbox_valid = bbox_area > 0.01
        center = _bbox_center_x(bbox)
        bbox_confidence = str(obj.get("bbox_confidence", "")).lower()
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        goal_match_confidence = float(vlm_report.get("goal_match_confidence", 0.0) or 0.0)
        visible = visibility not in {"occluded", "not_visible"}
        close_or_valid = range_bin in {"very_near", "close", "near"} or bbox_valid
        if not visible or not close_or_valid:
            return False

        if bbox_confidence == "low":
            return False
        if not bbox_valid:
            return False
        if _bbox_edge_clipped(bbox):
            return False
        if center is not None and not 0.12 <= center <= 0.88:
            return False

        room_probs = self._waypoint_room_probs(waypoint_id)
        matching_confirmed = self.object_memory_pool.matching_entry(
            label=label,
            obj=obj,
            pose=pose,
            waypoint_id=waypoint_id,
            room_probs=room_probs,
        )
        if matching_confirmed is not None:
            if goal_relevant and bool(vlm_report.get("goal_visible", False)):
                return True
            return confidence >= 0.65

        hypothesis = self._matching_object_hypothesis(
            label=label,
            position=_estimate_object_position(pose, obj),
        )
        if hypothesis is None:
            return False

        waypoint_ids = list(hypothesis.metadata.get("observed_from_waypoint_ids") or [])
        multi_frame = hypothesis.seen_count >= 2
        multi_view = len(waypoint_ids) >= 2
        consistent_signal = self._hypothesis_consistent_with_observation(
            hypothesis,
            obj,
            room_probs=room_probs,
        )
        strong_goal_match = bool(vlm_report.get("goal_visible", False)) and (
            confidence >= 0.60 or goal_match_confidence >= 0.50
        )
        high_conf_context_object = (
            not goal_relevant
            and confidence >= 0.78
            and bbox_area >= 0.03
            and bbox_area <= 0.55
            and visibility in {"visible", "clear", "prominent", "partially_visible", ""}
        )
        strong_single_goal_anchor = (
            goal_relevant
            and bool(vlm_report.get("goal_visible", False))
            and (confidence >= 0.90 or goal_match_confidence >= 0.50)
            and 0.04 <= bbox_area <= 0.18
            and range_bin in {"very_near", "close", "near", "mid", "medium"}
        )
        evidence_ready = multi_frame or multi_view or consistent_signal
        if goal_relevant:
            return strong_goal_match and (evidence_ready or strong_single_goal_anchor)
        return high_conf_context_object and evidence_ready

    def _matching_object_hypothesis(
        self,
        *,
        label: str,
        position: np.ndarray,
    ) -> Optional[Hypothesis]:
        best: Optional[Hypothesis] = None
        best_dist = float("inf")
        for hyp in self.hypothesis_pool.items():
            if hyp.kind != "object" or hyp.label.lower().strip() != label:
                continue
            dist = _distance(hyp.position, position)
            if dist <= self.hypothesis_pool.merge_radius * 1.5 and dist < best_dist:
                best = hyp
                best_dist = dist
        return best

    def _hypothesis_consistent_with_observation(
        self,
        hypothesis: Hypothesis,
        obj: Dict[str, Any],
        *,
        room_probs: Dict[str, float],
    ) -> bool:
        metadata = hypothesis.metadata
        range_bin = str(obj.get("range_bin", "")).lower().strip()
        bearing = str(obj.get("bearing", "")).lower().strip()
        range_hits = [
            str(item).lower().strip()
            for item in metadata.get("range_bin_history", []) or []
            if str(item).strip()
        ]
        bearing_hits = [
            str(item).lower().strip()
            for item in metadata.get("bearing_history", []) or []
            if str(item).strip()
        ]
        range_consistent = bool(range_bin and range_hits.count(range_bin) >= 2)
        bearing_consistent = bool(bearing and bearing_hits.count(bearing) >= 2)
        room_consistent = False
        old_room = _dominant_label(metadata.get("room_probs", {}))
        new_room = _dominant_label(room_probs)
        if old_room and new_room:
            room_consistent = old_room == new_room
        return (
            hypothesis.seen_count >= 2
            and (range_consistent or bearing_consistent or room_consistent)
        )

    def update_landmarks_and_portals(
        self,
        vlm_report: Dict[str, Any],
        pose: AgentPose,
        goal: Optional[GoalNode],
        waypoint_id: str,
    ) -> None:
        labels: List[str] = []
        if goal is not None:
            labels.extend(str(label) for label in goal.landmarks or [])
        labels.extend(str(label) for label in vlm_report.get("landmarks", []) or [])
        for obj in vlm_report.get("objects", []) or []:
            label = str(obj.get("label", "")).strip()
            if _is_landmark_or_portal_label(label):
                labels.append(label)

        for label in _dedupe_labels(labels)[:8]:
            semantic_role = "portal" if _is_portal_label(label) else "landmark"
            existing = self._find_nearby_structure_label(label, pose.position, semantic_role)
            goal_landmarks = list(goal.landmarks or []) if goal else []
            instruction_relevance = 1.0 if label in goal_landmarks else 0.4
            if semantic_role == "portal":
                self.topo_map.add_or_update_portal_node(
                    label=label,
                    confidence=0.55,
                    estimated_position=pose.position.copy(),
                    observed_from_waypoint_id=waypoint_id,
                    node_id=existing.node_id if existing is not None else None,
                    attributes={
                        "source": "language_or_vlm",
                        "instruction_relevance": instruction_relevance,
                    },
                )
            else:
                self.topo_map.add_or_update_landmark_node(
                    label=label,
                    confidence=0.55,
                    estimated_position=pose.position.copy(),
                    observed_from_waypoint_id=waypoint_id,
                    source="language_or_vlm",
                    node_id=existing.node_id if existing is not None else None,
                    attributes={
                        "stability": 0.35,
                        "instruction_relevance": instruction_relevance,
                    },
                )

    def _find_nearby_structure_label(
        self,
        label: str,
        position: np.ndarray,
        semantic_role: str,
    ) -> Optional[SemanticNode]:
        label_key = label.lower().strip()
        node_type = NodeType.PORTAL if semantic_role == "portal" else NodeType.LANDMARK
        for node in self.topo_map.find_nodes_within_radius(
            position,
            radius=3.0,
            node_type=node_type,
        ):
            if node.label.lower().strip() != label_key:
                continue
            if node.attributes.get("semantic_role") == semantic_role:
                return node
        return None

    def _sync_object_viewing_candidate(self, entry: ObjectMemoryEntry) -> None:
        anchor_id = entry.best_anchor_waypoint_id or entry.anchor_waypoint_id
        anchor = self.topo_map.get_node(anchor_id)
        if anchor is None:
            return
        node = self.topo_map.get_node(entry.node_id)
        if node is not None:
            node.attributes["best_approach_position"] = anchor.position.copy().tolist()
            try:
                self.topo_map._sync_approach_point(entry.node_id, anchor_id, anchor.position.copy())
            except AttributeError:
                pass

    def _add_or_update_waypoint(
        self,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
    ) -> str:
        node_id = self.topo_map.add_or_update_waypoint(
            pose.position,
            heading=pose.heading,
            embedding=rgb_embed,
        )
        if self.current_waypoint_id is not None and self.current_waypoint_id != node_id:
            self.topo_map.add_edge(self.current_waypoint_id, node_id, EdgeType.NAVIGABLE)
        self.current_waypoint_id = node_id
        return node_id

    def _update_room(
        self,
        clip_hint: ClipHint,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        waypoint_id: str,
        goal: Optional[GoalNode] = None,
    ) -> None:
        room_scores = _scores_to_dict(clip_hint.room_scores)
        if not room_scores and clip_hint.room_label != "unknown":
            room_scores = {clip_hint.room_label: clip_hint.room_confidence}
        if not room_scores:
            return
        goal_room_prior = list(getattr(goal, "room_prior", []) or [])
        room_id = self.topo_map.add_or_update_room_region(
            waypoint_id=waypoint_id,
            room_scores=room_scores,
            embedding=rgb_embed,
            goal_room_prior=goal_room_prior,
        )
        if room_id is not None:
            self.structure_layer["last_room_id"] = room_id

    def _update_room_from_vlm(
        self,
        vlm_report: Dict[str, Any],
        waypoint_id: str,
        goal: Optional[GoalNode] = None,
    ) -> None:
        room_label = str(vlm_report.get("room", "")).strip()
        room_confidence = float(vlm_report.get("room_confidence", 0.0) or 0.0)
        if not room_label or room_label == "unknown" or room_confidence < 0.2:
            return

        room_scores: Dict[str, float] = {room_label: room_confidence}
        waypoint = self.topo_map.get_node(waypoint_id)
        if waypoint is not None:
            prev_scores = _scores_to_dict(
                waypoint.attributes.get("room_scores_snapshot")
                or waypoint.attributes.get("room_scores")
                or {}
            )
            if prev_scores:
                room_scores = _merge_score_dicts(prev_scores, room_scores, old_weight=0.6)

        goal_room_prior = list(getattr(goal, "room_prior", []) or [])
        room_id = self.topo_map.add_or_update_room_region(
            waypoint_id=waypoint_id,
            room_scores=room_scores,
            embedding=None,
            goal_room_prior=goal_room_prior,
        )
        if room_id is not None:
            self.structure_layer["last_room_id"] = room_id
            self.structure_layer["last_room_source"] = "vlm"

    def _generate_frontiers(
        self,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        waypoint_id: str,
    ) -> None:
        for delta, distance in (
            (-1.25, 2.8),
            (-0.65, 2.6),
            (0.0, 3.0),
            (0.65, 2.6),
            (1.25, 2.8),
        ):
            heading = pose.heading + delta
            position = pose.position + np.array(
                [-np.sin(heading) * distance, 0.0, -np.cos(heading) * distance],
                dtype=np.float32,
            )
            nearest = self.topo_map.find_nearest_node(
                position,
                node_type=NodeType.WAYPOINT_FRONTIER,
            )
            if nearest is not None and np.linalg.norm(nearest.position - position) < 1.2:
                continue
            self.topo_map.add_or_update_frontier(
                position,
                created_from_waypoint=waypoint_id,
                embedding=rgb_embed,
                attributes={
                    "source": "frontier",
                    "expected_room_probs": self._waypoint_room_probs(waypoint_id),
                },
            )

    def _waypoint_room_probs(self, waypoint_id: str) -> Dict[str, float]:
        waypoint = self.topo_map.get_node(waypoint_id)
        if waypoint is None:
            return {}
        scores = (
            waypoint.attributes.get("room_scores_snapshot")
            or waypoint.attributes.get("room_scores")
            or {}
        )
        out = _scores_to_dict(scores)
        if out:
            return out

        # Fallback 1: resolve room region bound to this waypoint.
        room_id = waypoint.attributes.get("room_region_id") or waypoint.attributes.get("room_id")
        if room_id:
            room = self.topo_map.get_node(str(room_id))
            if room is not None:
                out = _scores_to_dict(room.attributes.get("room_probs", {}))
                if out:
                    return out

        # Fallback 2: infer from graph relation (waypoint --contains--> room).
        for nbr_id in self.topo_map.graph.neighbors(waypoint_id):
            edge = self.topo_map.graph.edges[waypoint_id, nbr_id]
            edge_types = list(edge.get("edge_types", []))
            if not edge_types and edge.get("edge_type"):
                edge_types = [edge.get("edge_type")]
            if EdgeType.CONTAINS.value not in edge_types:
                continue
            room = self.topo_map.get_node(str(nbr_id))
            if room is None or room.node_type != NodeType.ROOM:
                continue
            out = _scores_to_dict(room.attributes.get("room_probs", {}))
            if out:
                return out

        # Fallback 3: reuse last structure-layer room when waypoint has no room observation yet.
        last_room_id = self.structure_layer.get("last_room_id")
        if last_room_id:
            room = self.topo_map.get_node(str(last_room_id))
            if room is not None:
                out = _scores_to_dict(room.attributes.get("room_probs", {}))
                if out:
                    return out
        return {}


class StructurePlanner:
    """Decide where to search from structure-layer memory."""

    def build_search_intents(
        self,
        *,
        memory: MemoryManager,
        goal: Optional[GoalNode],
        pose: AgentPose,
    ) -> List[SearchIntent]:
        if goal is None:
            return [self._explore_intent(memory)]

        intents: List[SearchIntent] = []
        intents.extend(self._object_intents(memory, goal, pose))
        intents.extend(self._room_intents(memory, goal, pose))
        intents.extend(self._landmark_intents(memory, goal, pose))
        intents.extend(self._portal_intents(memory, goal, pose))
        intents.extend(self._hypothesis_intents(memory, goal, pose))
        intents.append(self._explore_intent(memory))
        intents.sort(key=lambda intent: intent.score, reverse=True)
        return intents

    def select_intent(
        self,
        *,
        memory: MemoryManager,
        goal: Optional[GoalNode],
        pose: AgentPose,
    ) -> SearchIntent:
        intents = self.build_search_intents(memory=memory, goal=goal, pose=pose)
        return intents[0] if intents else self._explore_intent(memory)

    def _object_intents(
        self,
        memory: MemoryManager,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        goal_label = str(goal.target_object or "").lower().strip()
        for entry in memory.object_memory_pool.entries():
            if entry.label.lower().strip() != goal_label:
                continue
            node = memory.topo_map.get_node(entry.node_id)
            if node is None:
                continue
            anchor_id = entry.best_anchor_waypoint_id or entry.anchor_waypoint_id
            anchor = memory.topo_map.get_node(anchor_id)
            if anchor is None:
                anchor = memory.topo_map.navigation_target_for_structure_node(entry.node_id)
            if anchor is None:
                continue
            failed = max(
                int(entry.failed_count),
                int(node.attributes.get("failed_count", 0) or 0),
            )
            unreachable = max(
                int(entry.unreachable_count),
                int(node.attributes.get("unreachable_count", 0) or 0),
            )
            risk = 0.25 * failed + 0.75 * unreachable
            bad_anchor_ids = {
                str(item)
                for item in (node.attributes.get("bad_anchor_waypoint_ids") or [])
                if str(item).strip()
            }
            anchor_failed_count = int(node.attributes.get("anchor_failed_count", 0) or 0)
            bad_anchor_penalty = 0.35 * anchor_failed_count
            if anchor.node_id in bad_anchor_ids:
                bad_anchor_penalty += 0.45
            dist = _distance(pose.position, anchor.position)
            score = (
                1.0
                + 0.45 * float(entry.confidence)
                + min(0.25, 0.04 * entry.seen_count)
                - 0.04 * dist
                - risk
                - bad_anchor_penalty
            )
            intents.append(SearchIntent(
                intent_type="object",
                target_label=entry.label,
                structure_node_id=entry.node_id,
                preferred_anchor_waypoint_id=anchor.node_id,
                target_position=anchor.position.copy(),
                score=score,
                reason=(
                    f"confirmed_object:{entry.label}; conf={entry.confidence:.2f}; "
                    f"seen={entry.seen_count}; failed={failed}; unreachable={unreachable}; "
                    f"anchor_failed={anchor_failed_count}"
                ),
                preferred_room=node.attributes.get("room_label"),
                evidence_refs=[entry.object_id],
            ))
        return intents

    def _room_intents(
        self,
        memory: MemoryManager,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        if not room_priors:
            return intents
        for node in memory.topo_map.get_nodes_by_type(NodeType.ROOM):
            probs = _scores_to_dict(node.attributes.get("room_probs", {}))
            best_prior_prob = max(
                (float(probs.get(room, 0.0)) for room in room_priors),
                default=0.0,
            )
            label_key = str(node.label or "").lower().strip()
            label_bonus = 0.35 if label_key in room_priors else 0.0
            if best_prior_prob <= 0.0 and label_bonus <= 0.0:
                continue
            nav_node = memory.topo_map.navigation_target_for_structure_node(node.node_id)
            dist = _distance(pose.position, nav_node.position) if nav_node is not None else 0.0
            evidence_count = len(node.attributes.get("evidence_views", []))
            object_support = len(node.attributes.get("observed_objects", {})) * 0.03
            uncertainty = 0.15 if node.attributes.get("room_state") != "confirmed" else 0.0
            score = (
                0.45
                + label_bonus
                + 0.75 * best_prior_prob
                + min(0.20, 0.03 * evidence_count)
                + min(0.15, object_support)
                - 0.03 * dist
                - uncertainty
            )
            intents.append(SearchIntent(
                intent_type="room",
                target_label=str(node.label or ""),
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav_node.node_id if nav_node is not None else None,
                target_position=nav_node.position.copy() if nav_node is not None else node.position.copy(),
                score=score,
                reason=(
                    f"room_prior:{','.join(sorted(room_priors))}; "
                    f"prob={best_prior_prob:.2f}; state={node.attributes.get('room_state', 'unknown')}"
                ),
                preferred_room=str(node.label or ""),
                evidence_refs=[node.node_id],
            ))
        return intents

    def _landmark_intents(
        self,
        memory: MemoryManager,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        landmark_labels = {str(label).lower().strip() for label in goal.landmarks or []}
        if not landmark_labels:
            return intents
        for node in memory.topo_map.get_nodes_by_type(NodeType.LANDMARK):
            label = str(node.label or "").lower().strip()
            if label not in landmark_labels:
                continue
            nav_node = memory.topo_map.navigation_target_for_structure_node(node.node_id)
            if nav_node is None:
                continue
            dist = _distance(pose.position, nav_node.position)
            confidence = float(node.confidence)
            score = 0.55 + 0.35 * confidence - 0.035 * dist
            intents.append(SearchIntent(
                intent_type="landmark",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav_node.node_id,
                target_position=nav_node.position.copy(),
                score=score,
                reason=f"instruction_landmark:{node.label}; conf={confidence:.2f}",
                related_landmarks=[node.label],
                evidence_refs=[node.node_id],
            ))
        return intents

    def _portal_intents(
        self,
        memory: MemoryManager,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        room_priors = {str(room).lower().strip() for room in goal.room_prior or []}
        for node in memory.topo_map.get_nodes_by_type(NodeType.PORTAL):
            if node.attributes.get("blocked", False):
                continue
            nav_node = memory.topo_map.navigation_target_for_structure_node(node.node_id)
            if nav_node is None:
                continue
            connected_rooms = {
                str(room).lower().strip()
                for room in node.attributes.get("connected_rooms", []) or []
            }
            room_match = 0.25 if room_priors and connected_rooms.intersection(room_priors) else 0.0
            dist = _distance(pose.position, nav_node.position)
            portal_failed = int(node.attributes.get("failed_count", 0) or 0)
            repeated_visits = int(node.attributes.get("visit_count", 0) or 0)
            score = (
                0.35
                + room_match
                + 0.25 * float(node.confidence)
                - 0.03 * dist
                - 0.30 * portal_failed
                - 0.20 * min(3, max(0, repeated_visits - 1))
            )
            intents.append(SearchIntent(
                intent_type="portal",
                target_label=node.label,
                structure_node_id=node.node_id,
                preferred_anchor_waypoint_id=nav_node.node_id,
                target_position=nav_node.position.copy(),
                score=score,
                reason=(
                    f"portal_candidate:{node.label}; room_match={room_match:.2f}; "
                    f"failed={portal_failed}; visits={repeated_visits}"
                ),
                preferred_room=next(iter(connected_rooms), None),
                evidence_refs=[node.node_id],
            ))
        return intents

    def _hypothesis_intents(
        self,
        memory: MemoryManager,
        goal: GoalNode,
        pose: AgentPose,
    ) -> List[SearchIntent]:
        intents: List[SearchIntent] = []
        for hyp in memory.hypothesis_pool.active_for_goal(goal, memory.topo_map.current_step):
            if hyp.anchor_waypoint_id is None:
                continue
            anchor = memory.topo_map.get_node(hyp.anchor_waypoint_id)
            if anchor is None:
                continue
            dist = _distance(pose.position, anchor.position)
            score = 0.30 + 0.45 * hyp.confidence + min(0.15, 0.04 * hyp.seen_count) - 0.03 * dist
            intents.append(SearchIntent(
                intent_type="hypothesis",
                target_label=hyp.label,
                structure_node_id=None,
                preferred_anchor_waypoint_id=anchor.node_id,
                target_position=anchor.position.copy(),
                score=score,
                reason=f"weak_{hyp.kind}:{hyp.label}; conf={hyp.confidence:.2f}; seen={hyp.seen_count}",
                evidence_refs=[hyp.hypothesis_id],
            ))
        return intents

    def _explore_intent(self, memory: MemoryManager) -> SearchIntent:
        return SearchIntent(
            intent_type="explore",
            target_label="frontier",
            structure_node_id=None,
            preferred_anchor_waypoint_id=None,
            target_position=None,
            score=0.0,
            reason="no_stronger_structure_intent",
            evidence_refs=[],
        )


class NavigationPlanner:
    """Resolve search intents to executable navigation-layer targets."""

    def __init__(self):
        self.last_debug: Dict[str, Any] = {}
        self._memory: Optional[MemoryManager] = None

    def select_navigation_target(
        self,
        *,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
        state: AgentState,
    ) -> NavigationTarget:
        self._memory = memory
        proposals = self.build_navigation_proposals(intent=intent, memory=memory, pose=pose)
        proposals.sort(key=lambda proposal: proposal.score, reverse=True)

        # Hard exclusion after stuck/failure feedback: recently failed targets
        # should not be immediately re-selected.
        current_step = int(memory.topo_map.current_step)
        failed_info = memory.navigation_layer.get("last_failed_target", {})
        failed_id = str(failed_info.get("node_id", "") or "").strip()
        failed_until = int(failed_info.get("blocked_until_step", -1) or -1)
        if failed_id and current_step <= failed_until:
            proposals = [p for p in proposals if p.target_node_id != failed_id]

        self.last_debug = {
            "intent": self._intent_debug(intent),
            "proposal_count": len(proposals),
            "top_proposals": [self._navigation_proposal_debug(p) for p in proposals[:6]],
        }
        if not proposals:
            return NavigationTarget(
                node_id=None,
                position=None,
                target_type="idle",
                reason=f"no_navigation_target_for:{intent.intent_type}",
                source_intent=intent,
                structure_node_id=intent.structure_node_id,
                evidence_refs=list(intent.evidence_refs),
            )
        best = proposals[0]
        return NavigationTarget(
            node_id=best.target_node_id,
            position=best.target_position,
            target_type=best.target_type,
            reason=best.reason,
            source_intent=intent,
            structure_node_id=best.structure_node_id,
            evidence_refs=list(best.evidence_refs),
            score=best.score,
        )

    def build_navigation_proposals(
        self,
        *,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        proposals: List[NavigationProposal] = []
        if intent.intent_type == "object":
            proposals.extend(self._anchor_proposals(intent, memory, pose, "anchor_waypoint"))
            proposals.extend(self._candidate_proposals(intent, memory, pose))
        elif intent.intent_type == "room":
            proposals.extend(self._room_member_proposals(intent, memory, pose))
            proposals.extend(self._portal_or_frontier_proposals(intent, memory, pose))
        elif intent.intent_type == "landmark":
            proposals.extend(self._anchor_proposals(intent, memory, pose, "landmark_waypoint"))
            proposals.extend(self._nearby_frontier_proposals(intent, memory, pose))
        elif intent.intent_type == "portal":
            proposals.extend(self._portal_connection_proposals(intent, memory, pose))
        elif intent.intent_type == "hypothesis":
            proposals.extend(self._anchor_proposals(intent, memory, pose, "hypothesis_waypoint"))
        proposals.extend(self._explore_frontier_proposals(intent, memory, pose))
        return proposals

    def _anchor_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
        target_type: str,
    ) -> List[NavigationProposal]:
        current_step = int(memory.topo_map.current_step)
        candidates: List[SemanticNode] = []
        if intent.preferred_anchor_waypoint_id:
            node = memory.topo_map.get_node(intent.preferred_anchor_waypoint_id)
            if node is not None:
                candidates.append(node)

        if intent.structure_node_id:
            fallback = memory.topo_map.navigation_target_for_structure_node(intent.structure_node_id)
            if fallback is not None:
                candidates.append(fallback)

            # For object intents, try alternate observed anchors when the best
            # anchor is marked bad or cooling down.
            if intent.intent_type == "object":
                structure_node = memory.topo_map.get_node(intent.structure_node_id)
                if structure_node is not None:
                    bad_ids = {
                        str(item)
                        for item in (structure_node.attributes.get("bad_anchor_waypoint_ids") or [])
                        if str(item).strip()
                    }
                    for waypoint_id in structure_node.attributes.get("observed_from_waypoint_ids", []) or []:
                        waypoint_id = str(waypoint_id)
                        if not waypoint_id or waypoint_id in bad_ids:
                            continue
                        alt = memory.topo_map.get_node(waypoint_id)
                        if alt is not None:
                            candidates.append(alt)

        dedup: Dict[str, SemanticNode] = {}
        for item in candidates:
            if item.node_id not in dedup:
                dedup[item.node_id] = item

        node = None
        for candidate in dedup.values():
            if self._is_navigation_node_available(candidate, current_step=current_step):
                node = candidate
                break

        if node is None:
            return []
        return [self._proposal_for_node(
            node=node,
            intent=intent,
            pose=pose,
            target_type=target_type,
            semantic_alignment=1.0,
            reason=f"{intent.intent_type}_resolved_to:{node.node_id}",
        )]

    def _candidate_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        if intent.structure_node_id is None:
            return []
        current_step = int(memory.topo_map.current_step)
        out = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_APPROACH):
            if node.attributes.get("source_object_id") != intent.structure_node_id:
                continue
            if not self._is_navigation_node_available(node, current_step=current_step):
                continue
            out.append(self._proposal_for_node(
                node=node,
                intent=intent,
                pose=pose,
                target_type="candidate_waypoint",
                semantic_alignment=0.95,
                reason=f"object_approach_candidate:{node.node_id}",
            ))
        return out

    def _room_member_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        room = memory.topo_map.get_node(intent.structure_node_id) if intent.structure_node_id else None
        if room is None:
            return []
        current_step = int(memory.topo_map.current_step)
        out = []
        for waypoint_id in room.attributes.get("member_waypoints", []) or []:
            node = memory.topo_map.get_node(str(waypoint_id))
            if node is None:
                continue
            if not self._is_navigation_node_available(node, current_step=current_step):
                continue
            out.append(self._proposal_for_node(
                node=node,
                intent=intent,
                pose=pose,
                target_type="room_waypoint",
                semantic_alignment=0.75,
                reason=f"room_member:{room.label}->{node.node_id}",
            ))
        return out

    def _portal_or_frontier_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        current_step = int(memory.topo_map.current_step)
        out = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.PORTAL):
            connected_rooms = {str(item) for item in node.attributes.get("connected_rooms", []) or []}
            if intent.structure_node_id and intent.structure_node_id not in connected_rooms:
                continue
            nav_node = memory.topo_map.navigation_target_for_structure_node(node.node_id)
            if nav_node is None:
                continue
            if not self._is_navigation_node_available(nav_node, current_step=current_step):
                continue
            out.append(self._proposal_for_node(
                node=nav_node,
                intent=intent,
                pose=pose,
                target_type="portal_waypoint",
                semantic_alignment=0.65,
                reason=f"room_portal:{node.label}",
            ))
        out.extend(self._expected_room_frontiers(intent, memory, pose))
        return out

    def _portal_connection_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        node = memory.topo_map.get_node(intent.structure_node_id) if intent.structure_node_id else None
        if node is None:
            return []
        current_step = int(memory.topo_map.current_step)
        out = []
        for candidate_id in node.attributes.get("connected_waypoints", []) or []:
            nav_node = memory.topo_map.get_node(str(candidate_id))
            if nav_node is None:
                continue
            if not self._is_navigation_node_available(nav_node, current_step=current_step):
                continue
            out.append(self._proposal_for_node(
                node=nav_node,
                intent=intent,
                pose=pose,
                target_type="portal_waypoint",
                semantic_alignment=0.85,
                reason=f"portal_connection:{node.label}",
            ))
        if not out:
            out.extend(self._anchor_proposals(intent, memory, pose, "portal_waypoint"))
        return out

    def _nearby_frontier_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        center_node = memory.topo_map.get_node(intent.preferred_anchor_waypoint_id) if intent.preferred_anchor_waypoint_id else None
        if center_node is None:
            return []
        current_step = int(memory.topo_map.current_step)
        out = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if not self._is_navigation_node_available(node, current_step=current_step):
                continue
            if _distance(center_node.position, node.position) > 5.0:
                continue
            out.append(self._proposal_for_node(
                node=node,
                intent=intent,
                pose=pose,
                target_type="frontier",
                semantic_alignment=0.45,
                reason=f"near_landmark_frontier:{node.node_id}",
            ))
        return out

    def _expected_room_frontiers(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        preferred = str(intent.preferred_room or intent.target_label or "").lower().strip()
        current_step = int(memory.topo_map.current_step)
        out = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if not self._is_navigation_node_available(node, current_step=current_step):
                continue
            expected = _scores_to_dict(node.attributes.get("expected_room_probs", {}))
            alignment = float(expected.get(preferred, 0.0)) if preferred else 0.0
            if alignment <= 0.0:
                continue
            out.append(self._proposal_for_node(
                node=node,
                intent=intent,
                pose=pose,
                target_type="frontier",
                semantic_alignment=alignment,
                reason=f"expected_room_frontier:{preferred}:{alignment:.2f}",
            ))
        return out

    def _explore_frontier_proposals(
        self,
        intent: SearchIntent,
        memory: MemoryManager,
        pose: AgentPose,
    ) -> List[NavigationProposal]:
        current_step = int(memory.topo_map.current_step)
        out = []
        for node in memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_FRONTIER):
            if not self._is_navigation_node_available(node, current_step=current_step):
                continue
            out.append(self._proposal_for_node(
                node=node,
                intent=intent,
                pose=pose,
                target_type="frontier",
                semantic_alignment=0.20 if intent.intent_type == "explore" else 0.05,
                reason=f"explore_frontier:{node.node_id}",
            ))
        return out

    def _proposal_for_node(
        self,
        *,
        node: SemanticNode,
        intent: SearchIntent,
        pose: AgentPose,
        target_type: str,
        semantic_alignment: float,
        reason: str,
    ) -> NavigationProposal:
        dist = _distance(pose.position, node.position)
        failed = float(node.attributes.get("failed_count", 0) or 0)
        unreachable = float(node.attributes.get("unreachable_count", 0) or 0)
        blocked = bool(node.attributes.get("blocked", False))
        frontier_value = float(node.confidence) if node.node_type == NodeType.WAYPOINT_FRONTIER else 0.0
        visit_count = float(node.attributes.get("visit_count", 0) or 0)
        blocked_penalty = 1.0 if blocked else 0.0
        failed_penalty = 0.20 * failed + 0.45 * unreachable
        distance_cost = 0.035 * dist
        novelty = self._frontier_novelty(node, pose) if node.node_type == NodeType.WAYPOINT_FRONTIER else 0.0
        revisit_penalty = 0.12 * min(4.0, visit_count)
        reachability = max(0.0, 1.0 - blocked_penalty - 0.2 * unreachable)
        score = (
            0.40 * reachability
            + 0.35 * semantic_alignment
            + 0.20 * frontier_value
            + 0.18 * novelty
            + 0.20 * float(intent.score)
            - distance_cost
            - failed_penalty
            - revisit_penalty
            - blocked_penalty
        )
        return NavigationProposal(
            target_node_id=node.node_id,
            target_position=node.position.copy(),
            target_type=target_type,
            score=score,
            reason=(
                f"{reason}; nav_score={score:.2f}; dist={dist:.2f}; "
                f"novelty={novelty:.2f}; visits={visit_count:.0f}; "
                f"failed={failed:.0f}; unreachable={unreachable:.0f}"
            ),
            source_intent=intent,
            structure_node_id=intent.structure_node_id,
            evidence_refs=list(intent.evidence_refs),
        )

    def _frontier_novelty(self, node: SemanticNode, pose: AgentPose) -> float:
        visited = self._nearest_visited_distance(node.position)
        from_agent = _distance(pose.position, node.position)
        distance_outward = max(0.0, min(1.0, (from_agent - 1.5) / 4.0))
        visited_outward = max(0.0, min(1.0, visited / 3.5))
        return float(max(distance_outward, visited_outward))

    def _nearest_visited_distance(self, position: np.ndarray) -> float:
        if self._memory is None:
            return 4.0
        best = float("inf")
        for node in self._memory.topo_map.get_nodes_by_type(NodeType.WAYPOINT_VISITED):
            best = min(best, _distance(position, node.position))
        return 4.0 if best == float("inf") else best

    @staticmethod
    def _is_navigation_node_available(node: SemanticNode, *, current_step: int) -> bool:
        attrs = node.attributes
        if bool(attrs.get("consumed", False)):
            return False

        blocked_until = attrs.get("blocked_until_step")
        if blocked_until is not None:
            try:
                until_step = int(blocked_until)
            except (TypeError, ValueError):
                until_step = -1
            if current_step <= until_step:
                return False
            # Cooldown passed: release temporary blocked mark.
            if bool(attrs.get("blocked", False)):
                attrs["blocked"] = False

        if bool(attrs.get("blocked", False)):
            return False

        if int(attrs.get("unreachable_count", 0) or 0) >= 2:
            return False
        return True

    @staticmethod
    def _intent_debug(intent: SearchIntent) -> Dict[str, Any]:
        return {
            "intent_type": intent.intent_type,
            "target_label": intent.target_label,
            "structure_node_id": intent.structure_node_id,
            "preferred_anchor_waypoint_id": intent.preferred_anchor_waypoint_id,
            "preferred_room": intent.preferred_room,
            "score": round(float(intent.score), 4),
            "reason": intent.reason,
            "evidence_refs": list(intent.evidence_refs),
        }

    @staticmethod
    def _navigation_proposal_debug(proposal: NavigationProposal) -> Dict[str, Any]:
        return {
            "target_node_id": proposal.target_node_id,
            "target_type": proposal.target_type,
            "score": round(float(proposal.score), 4),
            "reason": proposal.reason,
            "structure_node_id": proposal.structure_node_id,
            "evidence_refs": list(proposal.evidence_refs),
        }


class Controller:
    """Converts selected targets and state decisions to environment actions."""

    def search_action(self, nav_target: NavigationTarget) -> Dict[str, Any]:
        if nav_target.position is None:
            return {"action": "turn_left", "reason": "search_no_target"}
        return {
            "action": "navigate",
            "target_position": nav_target.position,
            "target_node_id": nav_target.node_id,
            "target_type": nav_target.target_type,
        }

    def approach_action(
        self,
        vlm_report: Optional[Dict[str, Any]],
        nav_target: NavigationTarget,
        goal: Optional[GoalNode] = None,
    ) -> Dict[str, Any]:
        bbox = _target_bbox(vlm_report, goal)
        if bbox is not None:
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
            if center < 0.35:
                return {"action": "turn_left", "reason": "approach_center_target"}
            if center > 0.65:
                return {"action": "turn_right", "reason": "approach_center_target"}
            return {"action": "move_forward", "reason": "approach_visible_target"}
        if vlm_report and vlm_report.get("goal_visible", False):
            direction = str(vlm_report.get("target_direction") or "").lower()
            if "right" in direction:
                return {"action": "turn_right", "reason": "approach_scan_visible_no_bbox"}
            return {"action": "turn_left", "reason": "approach_scan_visible_no_bbox"}
        return self.search_action(nav_target)

    def verify_action(
        self,
        vlm_report: Optional[Dict[str, Any]] = None,
        goal: Optional[GoalNode] = None,
    ) -> Dict[str, Any]:
        fresh = bool(vlm_report and vlm_report.get("fresh", False))
        bbox = _target_bbox(vlm_report, goal)
        if bbox is None:
            return {"action": "turn_left", "reason": "verify_scan_no_target"}
        if not fresh:
            return {"action": "turn_left", "reason": "verify_wait_for_fresh"}
        center = _bbox_center_x(bbox)
        if center is None:
            return {"action": "turn_left", "reason": "verify_scan_no_bbox"}
        if center < 0.35:
            return {"action": "turn_left", "reason": "verify_center_target"}
        if center > 0.65:
            return {"action": "turn_right", "reason": "verify_center_target"}
        return {"action": "move_forward", "reason": "verify_hold_centered"}

    def stop_action(self) -> Dict[str, Any]:
        return {"action": "stop", "reason": "verified_goal"}


class StopVerifier:
    """Multi-frame stop gate.  SEARCH/TRACK cannot stop directly."""

    def observe(
        self,
        state: AgentState,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        if not vlm_report or not bool(vlm_report.get("fresh", False)):
            return False
        state.verify_attempts += 1
        best = _target_object(vlm_report, goal)
        if best is not None:
            state.verify_area_window.append(_bbox_area(best.get("bbox")))
            state.verify_area_window = state.verify_area_window[-4:]
        hit = self._single_frame_stop_hit(vlm_report, goal, state)
        state.verify_window.append(hit)
        state.verify_window = state.verify_window[-4:]
        state.verify_hits = sum(1 for item in state.verify_window if item)
        return state.verify_hits >= 2

    def should_enter_verify(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        best = _target_object(vlm_report, goal)
        if best is None:
            return False
        bbox_area = _bbox_area(best.get("bbox"))
        range_bin = str(best.get("range_bin", "")).lower()
        return _range_area_close_enough_for_verify(range_bin, bbox_area)

    def _single_frame_stop_hit(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
        state: AgentState,
    ) -> bool:
        if not vlm_report or not bool(vlm_report.get("goal_visible", False)):
            return False
        best = _target_object(vlm_report, goal)
        if best is None:
            return False
        bbox = best.get("bbox")
        bbox_area = _bbox_area(bbox)
        range_bin = str(best.get("range_bin", "")).lower()
        bbox_confidence = str(best.get("bbox_confidence", "")).lower()
        visibility = str(best.get("visibility", "")).lower()
        center = _bbox_center_x(bbox)
        centered = center is None or 0.15 <= center <= 0.90
        area_growing = _area_trend(state.verify_area_window) >= 0.015
        area_holding_close = (
            len(state.verify_area_window) >= 2
            and min(state.verify_area_window[-2:]) > 0.0
            and state.verify_area_window[-1] >= 0.9 * state.verify_area_window[-2]
        )
        close_enough = _range_area_close_enough_for_stop(range_bin, bbox_area)
        strong_visual_stop = (
            close_enough
            and bbox_confidence != "low"
            and visibility not in {"occluded", "not_visible"}
            and centered
            and not _bbox_edge_clipped(bbox)
            and (area_growing or area_holding_close)
        )
        vlm_stop_candidate = (
            bool(vlm_report.get("stop_candidate", False))
            and close_enough
            and not _bbox_edge_clipped(bbox)
            and (area_growing or area_holding_close)
        )
        return (
            vlm_stop_candidate
            or strong_visual_stop
        )


class GOATAgent(ConfTopoBaseAgent):
    """Monolithic GOAT agent with modular internals."""

    def __init__(self, config: Optional[ConfTopoConfig] = None):
        super().__init__(config)
        self.goal_manager = GoalManager(self)
        self.perception = PerceptionModule(self.config)
        self.memory = MemoryManager(self.topo_map, self.config)
        self.structure_planner = StructurePlanner()
        self.navigation_planner = NavigationPlanner()
        self.controller = Controller()
        self.stop_verifier = StopVerifier()
        self.state = AgentState()

        self.step_id: int = 0
        self._origin_position: Optional[np.ndarray] = None
        self._pose: Optional[AgentPose] = None
        self._rgb: Optional[np.ndarray] = None
        self._rgb_embed: Optional[np.ndarray] = None
        self._last_clip_hint: ClipHint = ClipHint()
        self._fresh_vlm_report: Optional[Dict[str, Any]] = None
        self._cached_vlm_report: Optional[Dict[str, Any]] = None
        self._cached_vlm_step: int = -1
        self._vlm_for_control: Optional[Dict[str, Any]] = None
        self._last_proposals: List[Any] = []
        self._last_intents: List[SearchIntent] = []
        self._last_intent: Optional[SearchIntent] = None
        self._last_nav_target: NavigationTarget = NavigationTarget(None, None)
        self._last_pose_for_stuck: Optional[np.ndarray] = None
        self._stuck_monitor_target_id: Optional[str] = None
        self._stuck_marked_target_id: Optional[str] = None

    def reset(self) -> None:
        super().reset()
        self.memory = MemoryManager(self.topo_map, self.config)
        self.state = AgentState(transition_reason="reset")
        self.step_id = 0
        self._origin_position = None
        self._pose = None
        self._rgb = None
        self._rgb_embed = None
        self._last_clip_hint = ClipHint()
        self._fresh_vlm_report = None
        self._cached_vlm_report = None
        self._cached_vlm_step = -1
        self._vlm_for_control = None
        self._last_proposals = []
        self._last_intents = []
        self._last_intent = None
        self._last_nav_target = NavigationTarget(None, None)
        self._last_pose_for_stuck = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None

    def reset_keep_memory(self) -> None:
        super().reset_keep_memory()
        self.state = AgentState(transition_reason="new_goal")
        self.memory.reset_runtime()
        self._fresh_vlm_report = None
        self._cached_vlm_report = None
        self._cached_vlm_step = -1
        self._vlm_for_control = None
        self._last_proposals = []
        self._last_intents = []
        self._last_intent = None
        self._last_nav_target = NavigationTarget(None, None)
        self._last_pose_for_stuck = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None

    def set_new_goal(self, goal: GoalNode) -> None:
        self.reset_keep_memory()
        self.goal_manager.set_goal(goal)
        self.perception.configure_goal(goal)

    def observe(self, obs: Dict[str, Any]) -> None:
        raw_position = np.asarray(obs["position"], dtype=np.float32)
        if self._origin_position is None:
            self._origin_position = raw_position.copy()
        position = raw_position - self._origin_position
        position[1] = 0.0
        self._pose = AgentPose(
            position=position,
            heading=float(obs.get("heading", 0.0)),
        )
        self._rgb = obs.get("rgb")
        if "rgb_embed" in obs:
            self._rgb_embed = np.asarray(obs["rgb_embed"], dtype=np.float32)
        elif "pano_rgb_embed" in obs:
            self._rgb_embed = np.asarray(obs["pano_rgb_embed"], dtype=np.float32)
        else:
            self._rgb_embed = None

    def update_memory(self) -> None:
        if self._pose is None:
            return
        self.update_memory_from_perception(
            goal=self.goal_manager.current_goal(),
            vlm_report=self._fresh_vlm_report,
        )

    def update_memory_from_perception(
        self,
        *,
        goal: Optional[GoalNode],
        vlm_report: Optional[Dict[str, Any]],
    ) -> None:
        if self._pose is None:
            return
        waypoint_id = self.memory.begin_navigation_step(
            pose=self._pose,
            rgb_embed=self._rgb_embed,
        )
        self.memory.update_structure_from_clip(
            clip_hint=self._last_clip_hint,
            goal=goal,
            waypoint_id=waypoint_id,
            rgb_embed=self._rgb_embed,
            pose=self._pose,
        )
        self.memory.update_structure_from_vlm(
            vlm_report=vlm_report,
            goal=goal,
            pose=self._pose,
            waypoint_id=waypoint_id,
        )
        self.memory.update_navigation_frontiers(
            pose=self._pose,
            rgb_embed=self._rgb_embed,
            waypoint_id=waypoint_id,
        )
        self.memory.finalize_step_memory()

    def plan(self) -> NavigationTarget:
        if self._pose is None:
            return NavigationTarget(None, None, "idle")
        goal = self.goal_manager.current_goal()
        self._last_intents = self.structure_planner.build_search_intents(
            memory=self.memory,
            goal=goal,
            pose=self._pose,
        )
        self._last_intent = (
            self._last_intents[0]
            if self._last_intents
            else SearchIntent(intent_type="explore", reason="no_intents")
        )
        self._last_nav_target = self.navigation_planner.select_navigation_target(
            intent=self._last_intent,
            memory=self.memory,
            pose=self._pose,
            state=self.state,
        )
        self._last_proposals = []
        return self._last_nav_target

    def act(self, plan_output: NavigationTarget) -> Dict[str, Any]:
        return self.act_with_state_machine(
            nav_target=plan_output,
            vlm_report=self._vlm_for_control,
        )

    def report_navigation_failed(
        self,
        target_node_id: Optional[str] = None,
        *,
        unreachable: bool = False,
    ) -> bool:
        """Let the external executor report that the selected navigation target failed."""
        nav_target = self._last_nav_target
        if nav_target.node_id is None:
            return False
        if target_node_id is not None and target_node_id != nav_target.node_id:
            return False
        return self.memory.mark_navigation_target_failed(
            nav_target.node_id,
            target_type=nav_target.target_type,
            unreachable=unreachable,
            reason="external_navigation_failed",
        )

    def on_navigation_event(
        self,
        target_node_id: Optional[str],
        event: str,
    ) -> bool:
        nav_target = self._last_nav_target
        if target_node_id is not None and nav_target.node_id != target_node_id:
            return False
        if event == "target_reached":
            return self.memory.mark_navigation_target_reached(
                nav_target.node_id,
                target_type=nav_target.target_type,
                reason="external_target_reached",
            )
        if event in {"navigation_failed", "unreachable"}:
            return self.report_navigation_failed(
                target_node_id,
                unreachable=(event == "unreachable"),
            )
        return False

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_id += 1
        self._step_count += 1

        self.observe(obs)
        goal = self.goal_manager.current_goal()

        self._last_clip_hint = self.perception.run_clip(self._rgb_embed, goal)
        need_vlm = self.perception.should_trigger_vlm(
            clip_hint=self._last_clip_hint,
            goal=goal,
            state=self.state,
            step_id=self.step_id,
        )
        self._fresh_vlm_report = None
        if need_vlm:
            self._fresh_vlm_report = self.perception.run_vlm(
                self._rgb,
                goal,
                step_id=self.step_id,
                state=self.state,
                clip_hint=self._last_clip_hint,
            )
            if self._fresh_vlm_report and self._fresh_vlm_report.get("fresh", False):
                self._cached_vlm_report = self._fresh_vlm_report
                self._cached_vlm_step = self.step_id

        self._vlm_for_control = self._fresh_vlm_report
        if (
            self._vlm_for_control is None
            and self._cached_vlm_report is not None
            and self.step_id - self._cached_vlm_step <= 3
        ):
            self._vlm_for_control = dict(self._cached_vlm_report)
            self._vlm_for_control["fresh"] = False
            self._vlm_for_control["cached_from_step"] = self._cached_vlm_step

        self.update_memory_from_perception(
            goal=goal,
            vlm_report=self._fresh_vlm_report,
        )
        nav_target = self.plan()
        action = self.act_with_state_machine(
            nav_target=nav_target,
            vlm_report=self._vlm_for_control,
        )
        self._update_stuck_monitor(nav_target, action)
        action["debug"] = self._debug_snapshot(need_vlm)
        return action

    def act_with_state_machine(
        self,
        *,
        nav_target: NavigationTarget,
        vlm_report: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        goal = self.goal_manager.current_goal()
        target = _target_object(vlm_report, goal)
        target_visible = target is not None
        self._observe_target_visibility(target, nav_target)

        if self.state.mode == GOATState.SEARCH:
            if target_visible:
                self._enter_state(GOATState.TRACK, "target_object_visible")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            return self.controller.search_action(nav_target)

        elif self.state.mode == GOATState.TRACK:
            if self._target_lost(max_lost_steps=4):
                self._enter_state(GOATState.SEARCH, "target_lost_in_track")
                return self.controller.search_action(nav_target)
            if (
                target_visible
                and bool(vlm_report and vlm_report.get("fresh", False))
                and self.step_id >= self.state.verify_cooldown_until_step
                and self._should_enter_verify(vlm_report, goal)
            ):
                self._enter_state(GOATState.VERIFY_STOP, "area_trend_verify_candidate")
                return self.controller.verify_action(vlm_report, goal)
            if self._tracking_stable(vlm_report) and self.state.track_hits >= 2:
                self._enter_state(GOATState.APPROACH, "bbox_or_bearing_stable")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            return self.controller.approach_action(vlm_report, nav_target, goal)

        elif self.state.mode == GOATState.APPROACH:
            if self._target_lost(max_lost_steps=3):
                self._enter_state(GOATState.TRACK, "target_lost_during_approach")
                return self.controller.search_action(nav_target)
            if (
                target_visible
                and bool(vlm_report and vlm_report.get("fresh", False))
                and self.step_id >= self.state.verify_cooldown_until_step
                and self._should_enter_verify(vlm_report, goal)
            ):
                self._enter_state(GOATState.VERIFY_STOP, "near_and_large_bbox")
                return self.controller.verify_action(vlm_report, goal)
            return self.controller.approach_action(vlm_report, nav_target, goal)

        elif self.state.mode == GOATState.VERIFY_STOP:
            if not self._goal_evidence_ready_for_stop(vlm_report, goal):
                self.state.verify_cooldown_until_step = self.step_id + 4
                self._enter_state(GOATState.TRACK, "verify_rejected_unconfirmed_goal")
                return self.controller.search_action(nav_target)
            if self.stop_verifier.observe(self.state, vlm_report, goal):
                self._enter_state(GOATState.STOP, "multi_frame_verified")
                return self.controller.stop_action()
            if self.state.verify_attempts >= 6 and self.state.verify_hits == 0:
                self.state.verify_cooldown_until_step = self.step_id + 3
                self._enter_state(GOATState.APPROACH, "verify_failed_no_hits")
                return self.controller.approach_action(vlm_report, nav_target, goal)
            if self._target_lost(max_lost_steps=3):
                self.state.verify_cooldown_until_step = self.step_id + 3
                self._enter_state(GOATState.TRACK, "target_lost_during_verify")
                return self.controller.search_action(nav_target)
            return self.controller.verify_action(vlm_report, goal)

        return self.controller.stop_action()

    def _enter_state(self, mode: GOATState, reason: str) -> None:
        if self.state.mode == mode:
            return
        self.state.mode = mode
        self.state.transition_reason = reason
        self.state.state_steps = 0
        if mode == GOATState.SEARCH:
            self.state.track_hits = 0
            self.state.target_visible_window.clear()
            self.state.active_object_node_id = None
        if mode == GOATState.TRACK:
            self.state.track_hits = 1 if self.state.target_visible_window[-1:] == [True] else 0
        if mode == GOATState.VERIFY_STOP:
            self.state.verify_attempts = 0
            self.state.verify_area_window.clear()
        else:
            self.state.verify_window.clear()
            self.state.verify_hits = 0
            self.state.verify_attempts = 0

    def _observe_target_visibility(
        self,
        target: Optional[Dict[str, Any]],
        nav_target: NavigationTarget,
    ) -> None:
        target_visible = target is not None
        self.state.state_steps += 1
        self.state.target_visible_window.append(target_visible)
        self.state.target_visible_window = self.state.target_visible_window[-5:]
        if target_visible:
            area = _bbox_area(target.get("bbox"))
            if area > 0.0:
                self.state.target_area_window.append(area)
                self.state.target_area_window = self.state.target_area_window[-5:]
                self.state.target_area_trend = _area_trend(
                    self.state.target_area_window
                )
            self.state.last_goal_seen_step = self.step_id
            self.state.target_lost_steps = 0
            self.state.track_hits += 1
            if nav_target.target_type in {"anchor_waypoint", "candidate_waypoint"}:
                self.state.active_object_node_id = (
                    nav_target.structure_node_id or nav_target.node_id
                )
        else:
            self.state.target_lost_steps += 1
            self.state.track_hits = 0

    def _target_lost(self, *, max_lost_steps: int) -> bool:
        if self.state.last_goal_seen_step < 0:
            return False
        return self.state.target_lost_steps >= max_lost_steps

    def _tracking_stable(self, vlm_report: Optional[Dict[str, Any]]) -> bool:
        best = _target_object(vlm_report, self.goal_manager.current_goal())
        if best is None:
            return False
        bbox = best.get("bbox")
        center = _bbox_center_x(bbox)
        if center is None:
            return bool(vlm_report and vlm_report.get("goal_visible", False))
        return 0.25 <= center <= 0.75

    def _should_enter_verify(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        best = _target_object(vlm_report, goal)
        if best is None or goal is None:
            return False

        bbox = best.get("bbox")
        bbox_area = _bbox_area(bbox)
        center = _bbox_center_x(bbox)
        range_bin = str(best.get("range_bin", "")).lower()
        visibility = str(best.get("visibility", "")).lower()
        bbox_confidence = str(best.get("bbox_confidence", "")).lower()
        area_growing = self.state.target_area_trend >= 0.015
        area_grew_over_window = (
            len(self.state.target_area_window) >= 3
            and self.state.target_area_window[-1]
            >= 1.20 * max(0.01, self.state.target_area_window[0])
        )
        evidence_ready = self._goal_evidence_ready_for_verify(vlm_report, goal)
        if (
            not _range_area_close_enough_for_verify(range_bin, bbox_area)
            or center is None
            or not 0.15 <= center <= 0.90
            or _bbox_edge_clipped(bbox)
            or visibility in {"occluded", "not_visible"}
            or bbox_confidence == "low"
            or not (area_growing or area_grew_over_window or evidence_ready)
        ):
            return False
        return True

    def _goal_evidence_ready_for_verify(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        if goal is None:
            return False
        goal_label = str(goal.target_object or "").lower().strip()
        if not goal_label:
            return False
        best = _target_object(vlm_report, goal)
        bbox_area = _bbox_area(best.get("bbox")) if best is not None else 0.0
        for entry in self.memory.object_memory_pool.entries():
            if entry.label.lower().strip() != goal_label:
                continue
            if entry.seen_count >= 1 and bbox_area <= 0.22:
                return True
        for hyp in self.memory.hypothesis_pool.items():
            if hyp.kind != "object" or hyp.label.lower().strip() != goal_label:
                continue
            if hyp.seen_count >= 2:
                return True
            if (
                hyp.source == "goal_vlm"
                and float(hyp.confidence) >= 0.85
                and 0.04 <= bbox_area <= 0.18
            ):
                return True
        return False

    def _goal_evidence_ready_for_stop(
        self,
        vlm_report: Optional[Dict[str, Any]],
        goal: Optional[GoalNode],
    ) -> bool:
        if goal is None:
            return False
        goal_label = str(goal.target_object or "").lower().strip()
        if not goal_label:
            return False
        best = _target_object(vlm_report, goal)
        bbox_area = _bbox_area(best.get("bbox")) if best is not None else 0.0
        if bbox_area <= 0.0 or bbox_area > 0.30:
            return False
        for entry in self.memory.object_memory_pool.entries():
            if entry.label.lower().strip() == goal_label and entry.seen_count >= 2:
                return True
        for hyp in self.memory.hypothesis_pool.items():
            if (
                hyp.kind == "object"
                and hyp.label.lower().strip() == goal_label
                and hyp.seen_count >= 2
            ):
                return True
        return False

    def _update_stuck_monitor(
        self,
        nav_target: NavigationTarget,
        action: Dict[str, Any],
    ) -> None:
        if self._pose is None:
            return

        current_position = self._pose.position.copy()
        if self._last_pose_for_stuck is None:
            self._last_pose_for_stuck = current_position
            return

        motion_delta = _distance(current_position, self._last_pose_for_stuck)
        self.state.last_motion_delta = motion_delta
        self._last_pose_for_stuck = current_position

        action_name = str(action.get("action", ""))
        target_id = nav_target.node_id
        target_type = nav_target.target_type
        can_be_stuck = (
            self.state.mode in (GOATState.SEARCH, GOATState.TRACK, GOATState.APPROACH)
            and action_name in {"navigate", "move_forward"}
            and target_id is not None
            and target_type in {
                "frontier",
                "room_waypoint",
                "hypothesis_waypoint",
                "anchor_waypoint",
                "candidate_waypoint",
                "landmark_waypoint",
                "portal_waypoint",
            }
            and not bool(self._vlm_for_control and self._vlm_for_control.get("goal_visible", False))
        )

        if not can_be_stuck:
            self._reset_stuck_monitor(clear_reason=False)
            return

        if target_id != self._stuck_monitor_target_id:
            self._stuck_monitor_target_id = target_id
            self._stuck_marked_target_id = None
            self.state.stuck_steps = 0
            self.state.stuck_target_node_id = target_id

        if motion_delta < 0.03:
            self.state.stuck_steps += 1
        else:
            self.state.stuck_steps = 0
            self.state.last_stuck_reason = ""

        if self.state.stuck_steps < 8:
            return

        self.state.stuck_recoveries += 1
        self.state.last_stuck_reason = (
            f"stuck_on_{target_type}:{target_id}; "
            f"delta={motion_delta:.3f}; steps={self.state.stuck_steps}"
        )
        cooldown_steps = 30
        if target_type == "room_waypoint":
            cooldown_steps = 20
        elif target_type == "frontier":
            cooldown_steps = 10**9
        blocked_until_step = int(self.topo_map.current_step + cooldown_steps)

        if target_type in {"anchor_waypoint", "candidate_waypoint"}:
            # Anchor failure should not delete semantic object memory; only cool down
            # the problematic anchor waypoint and remember it as a bad anchor.
            node = self.topo_map.get_node(target_id)
            if node is not None:
                node.attributes["blocked"] = True
                node.attributes["blocked_until_step"] = blocked_until_step
                node.attributes["failure_reason"] = "stuck_no_progress"
                node.attributes["failed_count"] = int(node.attributes.get("failed_count", 0) or 0) + 1

            structure_id = nav_target.structure_node_id
            structure_node = self.topo_map.get_node(structure_id) if structure_id else None
            if structure_node is not None and structure_node.node_type == NodeType.OBJECT:
                bad_ids = [
                    str(item)
                    for item in (structure_node.attributes.get("bad_anchor_waypoint_ids") or [])
                    if str(item).strip()
                ]
                if target_id not in bad_ids:
                    bad_ids.append(target_id)
                structure_node.attributes["bad_anchor_waypoint_ids"] = bad_ids[-8:]
                structure_node.attributes["anchor_failed_count"] = int(
                    structure_node.attributes.get("anchor_failed_count", 0) or 0
                ) + 1

            self.memory.navigation_layer["last_failed_target"] = {
                "node_id": target_id,
                "target_type": target_type,
                "failed_count": None,
                "unreachable_count": None,
                "reason": "anchor_waypoint_scan_no_progress",
                "step": self.topo_map.current_step,
                "blocked_until_step": blocked_until_step,
            }
        else:
            # Record failure on navigation node and then enforce explicit cooldown.
            self.memory.mark_navigation_target_failed(
                target_id,
                target_type=target_type,
                unreachable=False,
                reason="stuck_no_progress",
            )
            node = self.topo_map.get_node(target_id)
            if node is not None:
                node.attributes["blocked"] = True
                node.attributes["blocked_until_step"] = blocked_until_step
                node.attributes["failure_reason"] = "stuck_no_progress"
                if target_type == "frontier":
                    node.attributes["consumed"] = True

            if target_type == "portal_waypoint":
                structure_id = nav_target.structure_node_id
                structure_node = self.topo_map.get_node(structure_id) if structure_id else None
                if structure_node is not None and structure_node.node_type == NodeType.PORTAL:
                    structure_node.attributes["failed_count"] = int(
                        structure_node.attributes.get("failed_count", 0) or 0
                    ) + 1
                    structure_node.attributes["blocked_until_step"] = blocked_until_step

            last_failed = dict(self.memory.navigation_layer.get("last_failed_target", {}))
            if last_failed:
                last_failed["blocked_until_step"] = blocked_until_step
                self.memory.navigation_layer["last_failed_target"] = last_failed
        self.state.stuck_steps = 0

        action.clear()
        reason = (
            "anchor_waypoint_scan_no_progress"
            if target_type in {"anchor_waypoint", "candidate_waypoint"}
            else "stuck_recovery"
        )
        action.update(
            {
                "action": "turn_left",
                "reason": reason,
                "blocked_target_node_id": target_id,
                "blocked_target_type": target_type,
            }
        )

    def _reset_stuck_monitor(self, *, clear_reason: bool = True) -> None:
        self.state.stuck_steps = 0
        self.state.stuck_target_node_id = None
        self._stuck_monitor_target_id = None
        self._stuck_marked_target_id = None
        if clear_reason:
            self.state.last_stuck_reason = ""

    def _debug_snapshot(self, need_vlm: bool) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "state": self.state.mode.value,
            "transition_reason": self.state.transition_reason,
            "state_steps": self.state.state_steps,
            "clip_room": self._last_clip_hint.room_label,
            "clip_goal_sim": self._last_clip_hint.best_goal_sim,
            "need_vlm": need_vlm,
            "vlm_reason": self.perception.last_vlm_reason,
            "fresh_vlm": bool(self._fresh_vlm_report),
            "fresh_vlm_room": (
                None if self._fresh_vlm_report is None else self._fresh_vlm_report.get("room")
            ),
            "fresh_vlm_room_confidence": (
                None
                if self._fresh_vlm_report is None
                else float(self._fresh_vlm_report.get("room_confidence", 0.0) or 0.0)
            ),
            "cached_vlm_step": self._cached_vlm_step,
            "using_cached_vlm": bool(
                self._vlm_for_control is not None
                and self._fresh_vlm_report is None
            ),
            "vlm_goal_visible": bool(
                self._vlm_for_control and self._vlm_for_control.get("goal_visible", False)
            ),
            "target_visible_window": list(self.state.target_visible_window),
            "target_area_window": [
                round(float(area), 4) for area in self.state.target_area_window
            ],
            "target_area_trend": round(float(self.state.target_area_trend), 4),
            "verify_area_window": [
                round(float(area), 4) for area in self.state.verify_area_window
            ],
            "track_hits": self.state.track_hits,
            "target_lost_steps": self.state.target_lost_steps,
            "last_goal_seen_step": self.state.last_goal_seen_step,
            "active_object_node_id": self.state.active_object_node_id,
            "hypothesis_count": len(self.memory.hypothesis_pool),
            "active_hypotheses": self.memory.hypothesis_pool.to_debug(
                self.topo_map.current_step
            ),
            "object_memory_count": len(self.memory.object_memory_pool),
            "object_memory": self.memory.object_memory_pool.to_debug(),
            "search_intent_count": len(self._last_intents),
            "selected_intent": (
                self.navigation_planner._intent_debug(self._last_intent)
                if self._last_intent is not None
                else None
            ),
            "navigation_debug": dict(self.navigation_planner.last_debug),
            "target_node_id": self._last_nav_target.node_id,
            "target_type": self._last_nav_target.target_type,
            "target_reason": self._last_nav_target.reason,
            "target_score": round(float(self._last_nav_target.score), 4),
            "target_structure_node_id": self._last_nav_target.structure_node_id,
            "target_evidence_refs": list(self._last_nav_target.evidence_refs),
            "verify_hits": self.state.verify_hits,
            "verify_attempts": self.state.verify_attempts,
            "verify_cooldown_until_step": self.state.verify_cooldown_until_step,
            "stuck_steps": self.state.stuck_steps,
            "stuck_recoveries": self.state.stuck_recoveries,
            "stuck_target_node_id": self.state.stuck_target_node_id,
            "last_motion_delta": self.state.last_motion_delta,
            "last_stuck_reason": self.state.last_stuck_reason,
            "last_failed_target": dict(
                self.memory.navigation_layer.get("last_failed_target", {})
            ),
            "topomap_layers": self._layer_debug_snapshot(),
        }

    def _layer_debug_snapshot(self) -> Dict[str, Any]:
        structure = self.topo_map.get_structure_view()
        navigation = self.topo_map.get_navigation_view()
        cross = self.topo_map.get_cross_layer_view()
        return {
            "structure_nodes": len(structure.get("nodes", [])),
            "structure_edges": len(structure.get("edges", [])),
            "navigation_nodes": len(navigation.get("nodes", [])),
            "navigation_edges": len(navigation.get("edges", [])),
            "cross_edges": len(cross.get("edges", [])),
            "waypoint_room_bindings": len(cross.get("waypoint_room_bindings", [])),
        }


def _scores_to_dict(scores: Any) -> Dict[str, float]:
    if isinstance(scores, dict):
        items = scores.items()
    elif isinstance(scores, (list, tuple)):
        items = scores
    else:
        return {}
    out: Dict[str, float] = {}
    for item in items:
        try:
            label, score = item
            text = str(label).strip()
            if not text:
                continue
            out[text] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def _merge_score_dicts(
    old: Dict[str, float],
    new: Dict[str, float],
    *,
    old_weight: float = 0.7,
) -> Dict[str, float]:
    if not old:
        return dict(new)
    if not new:
        return dict(old)
    labels = set(old) | set(new)
    return {
        label: float(old_weight * old.get(label, 0.0) + (1.0 - old_weight) * new.get(label, 0.0))
        for label in labels
    }


def _dominant_label(scores: Any) -> Optional[str]:
    score_dict = _scores_to_dict(scores)
    if not score_dict:
        return None
    label, score = max(score_dict.items(), key=lambda item: item[1])
    return label.lower().strip() if score > 0.0 else None


def _last_directional_bearing(values: List[str]) -> Optional[str]:
    for value in reversed(values):
        bearing = str(value).lower().strip()
        if bearing in {"left", "right", "center"}:
            return bearing
    return None


def _object_position_uncertainty(obj: Dict[str, Any], *, seen_count: int) -> float:
    range_bin = str(obj.get("range_bin", "unknown")).lower().strip()
    bbox_area = _bbox_area(obj.get("bbox"))
    base = {
        "very_near": 0.25,
        "close": 0.30,
        "near": 0.45,
        "mid": 0.70,
        "medium": 0.70,
        "far": 0.90,
    }.get(range_bin, 0.85)
    if bbox_area >= 0.20:
        base -= 0.15
    elif bbox_area >= 0.08:
        base -= 0.08
    base -= min(0.25, 0.04 * max(0, seen_count - 1))
    return float(np.clip(base, 0.1, 1.0))


def _dedupe_labels(labels: List[str]) -> List[str]:
    seen = set()
    out = []
    for label in labels:
        text = str(label).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _is_portal_label(label: str) -> bool:
    key = str(label or "").lower()
    return any(
        token in key
        for token in (
            "door", "doorway", "corridor", "hallway", "entrance",
            "passage", "stairs", "staircase", "opening",
        )
    )


def _is_landmark_or_portal_label(label: str) -> bool:
    key = str(label or "").lower().strip()
    if not key:
        return False
    if _is_portal_label(key):
        return True
    strong_landmarks = (
        "sofa", "couch", "bed", "tv", "television", "refrigerator",
        "fridge", "sink", "table", "bookshelf", "counter", "wardrobe",
        "cabinet", "bathtub", "toilet", "stairs",
    )
    weak_small = (
        "cup", "book", "remote", "plant", "decoration", "bottle",
    )
    if any(token in key for token in weak_small):
        return False
    return any(token in key for token in strong_landmarks)


def _goal_text(goal: GoalNode) -> str:
    parts = [
        str(goal.description or "").strip(),
        " ".join(str(attr) for attr in goal.attributes or []),
        str(goal.target_object or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def _goal_id(goal: GoalNode) -> str:
    return str(goal.image_cache_key or goal.goal_image_id or goal.target_object)


def _best_label(scores: List[Any]) -> Optional[str]:
    if not scores:
        return None
    return str(scores[0][0])


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _estimate_object_position(pose: AgentPose, obj: Dict[str, Any]) -> np.ndarray:
    range_bin = str(obj.get("range_bin", "unknown")).lower().strip()
    forward_distance = _range_bin_distance(range_bin)

    bbox_center = _bbox_center_x(obj.get("bbox"))
    if bbox_center is None:
        bearing = str(obj.get("bearing", "")).lower().strip()
        if bearing == "left":
            lateral_fraction = -0.45
        elif bearing == "right":
            lateral_fraction = 0.45
        else:
            lateral_fraction = 0.0
    else:
        lateral_fraction = float(np.clip((bbox_center - 0.5) * 1.4, -0.75, 0.75))

    heading = float(pose.heading)
    forward = np.array(
        [-np.sin(heading), 0.0, -np.cos(heading)],
        dtype=np.float32,
    )
    right = np.array(
        [np.cos(heading), 0.0, -np.sin(heading)],
        dtype=np.float32,
    )
    lateral_distance = lateral_fraction * max(0.4, forward_distance * 0.8)
    estimate = (
        pose.position
        + forward * forward_distance
        + right * lateral_distance
    ).astype(np.float32)
    estimate[1] = 0.0
    return estimate


def _range_bin_distance(range_bin: str) -> float:
    if range_bin in {"very_near", "close"}:
        return 0.6
    if range_bin == "near":
        return 1.35
    if range_bin in {"mid", "medium"}:
        return 2.2
    if range_bin == "far":
        return 3.2
    return 1.5


def _range_area_close_enough_for_verify(range_bin: str, bbox_area: float) -> bool:
    return range_bin in {"very_near", "close", "near"}


def _range_area_close_enough_for_stop(range_bin: str, bbox_area: float) -> bool:
    return range_bin in {"very_near", "close", "near"}


def _vlm_report_to_dict(report: PerceptionReport, *, reason: str, mode: str) -> Dict[str, Any]:
    return {
        "fresh": True,
        "step": int(report.step_id),
        "mode": mode,
        "trigger_reason": reason,
        "room": report.room_label,
        "room_confidence": float(report.room_confidence),
        "goal_visible": bool(report.goal_visible),
        "goal_match_confidence": float(report.goal_match_confidence),
        "target_direction": report.target_direction,
        "target_visibility": report.target_visibility,
        "apparent_scale": report.apparent_scale,
        "stop_candidate": bool(report.stop_candidate),
        "recommended_action": report.recommended_action,
        "goal_reason": report.goal_reason,
        "objects": [obj.to_dict() for obj in report.objects],
        "landmarks": list(report.portals),
        "raw": dict(report.raw),
    }


def _target_object(
    vlm_report: Optional[Dict[str, Any]],
    goal: Optional[GoalNode],
) -> Optional[Dict[str, Any]]:
    if not vlm_report or goal is None:
        return None
    goal_label = str(goal.target_object or "").lower().strip()
    objects = [dict(obj) for obj in vlm_report.get("objects", []) if obj]
    matched = [
        obj
        for obj in objects
        if str(obj.get("label", "")).lower().strip() == goal_label
    ]
    if matched:
        return max(matched, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))

    if bool(vlm_report.get("goal_visible", False)) and objects:
        goal_match_confidence = float(vlm_report.get("goal_match_confidence", 0.0) or 0.0)
        visible_objects = [
            obj
            for obj in objects
            if str(obj.get("visibility", "")).lower() not in {"occluded", "not_visible"}
        ]
        candidates = visible_objects or objects
        if len(candidates) == 1 or goal_match_confidence >= 0.5:
            return max(candidates, key=lambda obj: float(obj.get("confidence", 0.0) or 0.0))
    return None


def _target_bbox(
    vlm_report: Optional[Dict[str, Any]],
    goal: Optional[GoalNode],
) -> Optional[Any]:
    target = _target_object(vlm_report, goal)
    return None if target is None else target.get("bbox")


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return 0.0
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return max(0.0, min(1.0, (x2 - x1) * (y2 - y1)))


def _bbox_center_x(bbox: Any) -> Optional[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, _y1, x2, _y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return 0.5 * (x1 + x2)


def _bbox_edge_clipped(bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, _y1, x2, _y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    width = max(0.0, x2 - x1)
    left_strip = x1 <= 0.01 and x2 <= 0.40
    right_strip = x2 >= 0.99 and x1 >= 0.60
    return width <= 0.40 and (left_strip or right_strip)


def _area_trend(areas: List[float]) -> float:
    valid = [float(area) for area in areas if area > 0.0]
    if len(valid) < 2:
        return 0.0
    return float(valid[-1] - valid[0])


ConfTopoGOATAgent = GOATAgent


__all__ = [
    "AgentPose",
    "AgentState",
    "ClipHint",
    "Controller",
    "GOATAgent",
    "GOATState",
    "GoalManager",
    "Hypothesis",
    "HypothesisPool",
    "MemoryManager",
    "NavigationTarget",
    "ObjectMemoryEntry",
    "ObjectMemoryPool",
    "PerceptionModule",
    "SearchIntent",
    "StructureProposal",
    "NavigationProposal",
    "StructurePlanner",
    "NavigationPlanner",
    "StopVerifier",
    "ConfTopoGOATAgent",
]
