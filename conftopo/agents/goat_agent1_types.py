"""Shared types for the clean Agent1 GOAT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import numpy as np


class GOATState(str, Enum):
    SEARCH = "SEARCH"
    CONFIRM = "CONFIRM"
    STOP = "STOP"


class ActivityState(str, Enum):
    """Task-conditioned lifecycle state of a node within an ActiveTopoView.

    Lives only in the view (task state), never on the persistent node.
      active     - participates in planning candidates now
      verify     - relevant but low map-confidence, needs re-observation
      folded     - hidden for now but recoverable (low relevance region)
      suppressed - excluded from candidates (blacklisted / dead)
    """

    ACTIVE = "active"
    VERIFY = "verify"
    FOLDED = "folded"
    SUPPRESSED = "suppressed"


@dataclass
class ActiveNodeInfo:
    """Per-node task-conditioned annotation kept inside an ActiveTopoView.

    ``relevance`` (R) and ``activity_state`` are task state (goal-scoped);
    ``map_confidence`` (C) and ``reachability`` (A) mirror environment facts
    read from the persistent node for convenient, single-pass planning.
    """

    node_id: str
    node_kind: str = ""  # "structure" | "navigation"
    relevance: float = 0.0
    map_confidence: float = 0.0
    reachability: float = 0.0
    activity_state: ActivityState = ActivityState.ACTIVE
    utility: float = 0.0


@dataclass
class ProjectionConfig:
    """Thresholds controlling how the projector classifies nodes.

    Phase B uses the degenerate defaults (everything ACTIVE); later phases
    tighten these to actually fold / suppress / route.
    """

    relevance_active_threshold: float = 0.0
    relevance_verify_threshold: float = 0.45
    # Rooms scoring below this task-relevance are folded away (their internal
    # frontiers/candidates + low-relevance objects leave the candidate set).
    # room_task_relevance floors at 0.35; goal-prior rooms score >= 0.70.
    relevance_fold_threshold: float = 0.45
    # VERIFY trigger: a goal-relevant object corroborated by fewer than this many
    # views (single-glimpse) must be re-observed before it is trusted to STOP.
    verify_min_view_count: int = 2
    verify_confidence_ceiling: float = 0.55  # reserved for future fold/verify tuning
    fold_enabled: bool = False
    suppress_enabled: bool = False
    verify_enabled: bool = False
    connectivity_enabled: bool = False


@dataclass
class RecoveryState:
    """Per-goal search budget / recovery ladder (agent-side, never on nodes)."""

    level: int = 0
    max_level: int = 2
    exhausted_regions: Set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.level = 0
        self.exhausted_regions = set()

    def maybe_escalate(
        self,
        *,
        no_progress_steps: int,
        budget: int,
        current_region: Optional[str] = None,
    ) -> bool:
        """Escalate one recovery level when the active region looks exhausted.

        Exhaustion here is 'no search progress for `budget` steps'. On escalation
        the current region is remembered as exhausted and the projector will
        relax folding (re-exposing next-relevant / folded regions). Levels are
        monotonic within a goal (reset on goal switch)."""
        if self.level >= self.max_level:
            return False
        if no_progress_steps < budget:
            return False
        self.level += 1
        if current_region:
            self.exhausted_regions.add(str(current_region))
        return True


@dataclass
class ActiveTopoView:
    """A thin, goal-scoped projection over the persistent DynamicTopoMap.

    Rebuilt on events (goal switch, promotion, verify result, ...) rather than
    every step. In Phase B this is a pure observability layer: every node is
    ACTIVE, so the planner (still reading the full graph) is unaffected.
    """

    goal_id: Optional[str] = None
    built_step: int = -1
    rebuild_count: int = 0
    recovery_level: int = 0
    node_infos: Dict[str, ActiveNodeInfo] = field(default_factory=dict)
    active_structure: Set[str] = field(default_factory=set)
    active_navigation: Set[str] = field(default_factory=set)
    active_edges: Set[str] = field(default_factory=set)
    folded_regions: Set[str] = field(default_factory=set)
    verification_nodes: Set[str] = field(default_factory=set)
    recovery_regions: Set[str] = field(default_factory=set)
    entry_waypoint: Optional[str] = None
    persistent_node_count: int = 0

    def is_candidate(self, node_id: Optional[str]) -> bool:
        """Whether a node may enter planning candidates under this view.

        Phase B: no node is suppressed, so this is always True for known nodes.
        """
        if node_id is None:
            return True
        info = self.node_infos.get(str(node_id))
        if info is None:
            return True
        return info.activity_state not in (
            ActivityState.SUPPRESSED,
            ActivityState.FOLDED,
        )

    def is_verify(self, node_id: Optional[str]) -> bool:
        """True when the node is a high-relevance / low-confidence target that
        should be re-observed before it is trusted enough to STOP on."""
        if node_id is None:
            return False
        info = self.node_infos.get(str(node_id))
        return info is not None and info.activity_state == ActivityState.VERIFY

    def activity_distribution(self) -> Dict[str, int]:
        dist = {state.value: 0 for state in ActivityState}
        for info in self.node_infos.values():
            dist[info.activity_state.value] += 1
        return dist

    def state_buckets(self) -> Dict[str, List[str]]:
        """Compact per-node states for visualization: only the non-ACTIVE nodes
        (folded / suppressed / verify), keyed by state."""
        buckets: Dict[str, List[str]] = {
            ActivityState.FOLDED.value: [],
            ActivityState.SUPPRESSED.value: [],
            ActivityState.VERIFY.value: [],
        }
        for info in self.node_infos.values():
            if info.activity_state.value in buckets:
                buckets[info.activity_state.value].append(info.node_id)
        return buckets

    def stats(self) -> Dict[str, Any]:
        active_count = sum(
            1 for i in self.node_infos.values() if i.activity_state == ActivityState.ACTIVE
        )
        return {
            "goal_id": self.goal_id,
            "built_step": int(self.built_step),
            "rebuild_count": int(self.rebuild_count),
            "recovery_level": int(self.recovery_level),
            "persistent_node_count": int(self.persistent_node_count),
            "active_node_count": int(active_count),
            "verification_node_count": len(self.verification_nodes),
            "folded_node_count": len(self.folded_regions),
            "entry_waypoint": self.entry_waypoint,
            "activity_distribution": self.activity_distribution(),
        }


@dataclass
class AgentPose:
    position: np.ndarray
    heading: float = 0.0


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
class NavigationTarget:
    node_id: Optional[str]
    position: Optional[np.ndarray]
    target_type: str = "idle"
    reason: str = ""
    source_intent: Optional[SearchIntent] = None
    structure_node_id: Optional[str] = None
    evidence_refs: List[Any] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ScanContext:
    scan_type: str = ""
    anchor_node_id: Optional[str] = None
    structure_node_id: Optional[str] = None
    remaining_steps: int = 0
    fresh_vlm_required: bool = True
    started_step: int = 0
    failure_recorded: bool = False

    def active(self) -> bool:
        return self.remaining_steps > 0 and bool(self.scan_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_type": self.scan_type,
            "anchor_node_id": self.anchor_node_id,
            "structure_node_id": self.structure_node_id,
            "remaining_steps": int(self.remaining_steps),
            "fresh_vlm_required": bool(self.fresh_vlm_required),
            "started_step": int(self.started_step),
            "failure_recorded": bool(self.failure_recorded),
            "active": self.active(),
        }


@dataclass
class NavigationHistoryEntry:
    step_id: int
    node_id: Optional[str]
    target_type: str = ""
    structure_node_id: Optional[str] = None
    distance: Optional[float] = None
    progress_delta: float = 0.0
    event: str = "selected"


@dataclass
class PlanningContext:
    recent_target_counts: Dict[str, int] = field(default_factory=dict)
    stale_target_ids: Dict[str, int] = field(default_factory=dict)
    current_waypoint_id: Optional[str] = None
    trajectory: List[str] = field(default_factory=list)
    recent_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MapAuditContext:
    trajectory: List[str] = field(default_factory=list)
    map_edges: List[Dict[str, Any]] = field(default_factory=list)
    supplementary_candidates: List[Dict[str, Any]] = field(default_factory=list)
    current_options: List[Dict[str, Any]] = field(default_factory=list)
    recent_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentDebugSnapshot:
    step_id: int
    state: str
    perception: Dict[str, Any]
    memory: Dict[str, Any]
    structure_intents: List[Dict[str, Any]]
    selected_intent: Dict[str, Any]
    navigation_proposals: List[Dict[str, Any]]
    selected_navigation_target: Dict[str, Any]
    stop_verifier: Dict[str, Any]
