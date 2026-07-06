"""Memory writing policy for the clean Agent1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from conftopo.agents.goat_agent1_types import AgentPose
from conftopo.core.confidence import update_memory_state, update_on_observation
from conftopo.core.dynamic_topo_map import DynamicTopoMap, EdgeType, NodeType, SemanticNode
from conftopo.core.hypothesis_pool import Hypothesis, HypothesisPool
from conftopo.core.instruction_graph import GoalNode
from conftopo.perception.perception_report import PerceptionReport


def bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center_x(bbox: Any) -> Optional[float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, _y1, x2, _y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    return 0.5 * (x1 + x2)


def bbox_edge_clipped(bbox: Any) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x1, _y1, x2, _y2 = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    width = max(0.0, x2 - x1)
    return width <= 0.40 and ((x1 <= 0.01 and x2 <= 0.40) or (x2 >= 0.99 and x1 >= 0.60))


def range_bin_distance(range_bin: str) -> float:
    key = str(range_bin or "").lower().strip()
    if key in {"very_near", "close"}:
        return 0.7
    if key == "near":
        return 1.35
    if key in {"mid", "medium"}:
        return 2.2
    if key == "far":
        return 3.2
    return 1.5


def estimate_object_position(pose: AgentPose, obj: Dict[str, Any]) -> np.ndarray:
    forward_distance = range_bin_distance(str(obj.get("range_bin", "unknown")))
    center = bbox_center_x(obj.get("bbox"))
    if center is None:
        bearing = str(obj.get("bearing", "")).lower().strip()
        lateral_fraction = -0.45 if bearing == "left" else 0.45 if bearing == "right" else 0.0
    else:
        lateral_fraction = float(np.clip((center - 0.5) * 1.4, -0.75, 0.75))
    heading = float(pose.heading)
    forward = np.array([-np.sin(heading), 0.0, -np.cos(heading)], dtype=np.float32)
    right = np.array([np.cos(heading), 0.0, -np.sin(heading)], dtype=np.float32)
    estimate = pose.position + forward * forward_distance + right * lateral_fraction * max(0.4, forward_distance * 0.8)
    return estimate.astype(np.float32)


def _embedding_from_value(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if arr.size <= 0:
        return None
    arr = arr.reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _object_description(attrs: Dict[str, Any]) -> str:
    if not isinstance(attrs, dict):
        return ""
    desc = attrs.get("description") or attrs.get("desc") or attrs.get("appearance")
    if isinstance(desc, str):
        return desc.strip()
    words: List[str] = []
    for key in ("color", "material", "shape", "size", "state"):
        value = attrs.get(key)
        if isinstance(value, str):
            words.append(value.strip())
        elif isinstance(value, dict):
            label = value.get("value") or value.get("label") or value.get("name")
            if isinstance(label, str):
                words.append(label.strip())
    return " ".join(word for word in words if word)


_FRONTIER_DIRECTIONS = (
    (0.0, "front"),
    (0.8, "front_left"),
    (-0.8, "front_right"),
    (1.57, "left"),
    (-1.57, "right"),
)


def _direction_alignment(direction_label: str, bearing: Any) -> float:
    direction = str(direction_label or "").lower().strip().replace("-", "_")
    key = str(bearing or "").lower().strip().replace("-", "_")
    if not key or key == "unknown":
        return 0.0
    if key in {"center", "front", "forward"}:
        return 1.0 if direction == "front" else 0.45 if "front" in direction else 0.0
    if "left" in key:
        if "left" in direction:
            return 1.0
        return 0.35 if direction == "front" else 0.0
    if "right" in key:
        if "right" in direction:
            return 1.0
        return 0.35 if direction == "front" else 0.0
    return 0.0


def _directional_room_probs(
    base_probs: Dict[str, float],
    goal: Optional[GoalNode],
    report: Optional[PerceptionReport],
    direction_label: str,
) -> Dict[str, float]:
    probs = {str(k).lower().strip(): float(v) for k, v in (base_probs or {}).items()}
    target_bbox_ok = _report_has_goal_bbox(report, goal)
    if goal is not None and report is not None and report.is_full and target_bbox_ok:
        target_align = _direction_alignment(direction_label, report.target_direction)
        target_conf = float(report.goal_match_confidence or 0.0)
        for room in goal.room_prior or []:
            key = str(room).lower().strip()
            if key:
                probs[key] = float(probs.get(key, 0.0)) + target_align * min(0.08, 0.02 + 0.06 * target_conf)
    if report is not None and report.is_full and report.portals:
        portal_align = 1.0 if direction_label == "front" else 0.45 if "front" in direction_label else 0.15
        for key in ("hallway", "corridor", "entry", "room"):
            probs[key] = float(probs.get(key, 0.0)) + 0.02 * portal_align
    return probs


def _report_has_goal_bbox(report: Optional[PerceptionReport], goal: Optional[GoalNode]) -> bool:
    if report is None or goal is None or not report.is_full or not report.goal_visible:
        return False
    for obj in report.objects:
        data = obj.to_dict()
        if GoalRelevanceScorer.label_relevance(str(data.get("label", "")), goal) < 0.65:
            continue
        if bbox_area(data.get("bbox")) <= 0.0:
            continue
        if str(data.get("bbox_confidence", "")).lower() == "low":
            continue
        if str(data.get("visibility", "")).lower() in {"not_visible", "occluded"}:
            continue
        return True
    return False


def _portal_label(portal: Dict[str, Any]) -> str:
    portal_type = str(portal.get("type", "portal") or "portal")
    state = str(portal.get("state", "uncertain") or "uncertain")
    bearing = str(portal.get("bearing", "unknown") or "unknown")
    parts = [
        state if state != "uncertain" else "",
        portal_type if portal_type != "unknown" else "portal",
        bearing if bearing != "unknown" else "",
    ]
    return " ".join(part for part in parts if part).strip() or "portal"


def _directional_semantic_value(
    room_probs: Dict[str, float],
    goal: Optional[GoalNode],
    report: Optional[PerceptionReport],
    direction_label: str,
) -> float:
    value = frontier_semantic_value(room_probs, goal)
    if report is not None and report.is_full:
        target_align = _direction_alignment(direction_label, report.target_direction)
        if _report_has_goal_bbox(report, goal):
            value = max(value, target_align * min(0.10, 0.03 + 0.07 * float(report.goal_match_confidence or 0.0)))
        for obj in report.objects:
            data = obj.to_dict()
            relevance = GoalRelevanceScorer.label_relevance(str(data.get("label", "")), goal)
            if relevance <= 0.0:
                continue
            align = _direction_alignment(direction_label, data.get("bearing", "unknown"))
            if bbox_area(data.get("bbox")) > 0.0 and str(data.get("bbox_confidence", "")).lower() != "low":
                value = max(value, min(0.10, align * relevance * float(data.get("confidence", 0.0) or 0.0) * 0.10))
        if report.portals:
            portal_align = 1.0 if direction_label == "front" else 0.45 if "front" in direction_label else 0.15
            value = max(value, 0.05 * portal_align)
    return float(max(0.0, min(1.0, value)))


class GoalRelevanceScorer:
    """Conservative task relevance for querying generic memory."""

    _ALIASES = {
        "rack": {
            "rack", "clothes rack", "coat rack", "towel rack", "shelf",
            "shelving", "storage shelf", "wooden shelf", "shelf rack",
            "bookcase",
        },
        "bed": {"bed", "mattress", "bunk bed"},
        "heater": {"heater", "radiator", "baseboard heater", "space heater"},
        "sink": {"sink", "basin", "washbasin"},
        "sofa": {"sofa", "couch", "loveseat"},
        "tv": {"tv", "television", "monitor", "screen"},
    }

    @classmethod
    def label_relevance(cls, label: str, goal: Optional[GoalNode]) -> float:
        if goal is None:
            return 0.0
        label_key = cls._clean(label)
        goal_key = cls._clean(goal.target_object)
        if not label_key or not goal_key:
            return 0.0
        if label_key == goal_key:
            return 1.0
        if goal_key in label_key or label_key in goal_key:
            return 0.85
        label_tokens = set(label_key.replace("-", " ").split())
        goal_tokens = set(goal_key.replace("-", " ").split())
        if label_tokens and goal_tokens:
            overlap = len(label_tokens & goal_tokens) / float(len(goal_tokens))
            if overlap >= 0.5:
                return 0.65
        for canonical, names in cls._ALIASES.items():
            group = {canonical, *names}
            if goal_key in group and label_key in group:
                return 0.80
            if goal_key in group:
                for alias in group:
                    if alias and (alias in label_key or label_key in alias):
                        return 0.78
        return 0.0

    @classmethod
    def object_score(cls, node: SemanticNode, goal: Optional[GoalNode]) -> float:
        score = 0.75 * cls.label_relevance(node.label, goal)
        attrs = str(node.attributes).lower()
        if goal is not None:
            for attr in goal.attributes or []:
                if str(attr).lower().strip() and str(attr).lower().strip() in attrs:
                    score += 0.10
            for room in goal.room_prior or []:
                if str(room).lower().strip() and str(room).lower().strip() in attrs:
                    score += 0.08
            for landmark in goal.landmarks or []:
                if str(landmark).lower().strip() and str(landmark).lower().strip() in attrs:
                    score += 0.05
        return min(1.0, float(score))

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").lower().strip().replace("_", " ")


class GoalEvidenceFilter:
    """Single source of truth for 'is this detection a trustworthy goal instance'.

    Both the memory writer (object promotion / hypothesis) and the controller
    (target selection / STOP) route their semantic + geometric checks through
    here, so bed/rack/doorway rules live in exactly one place.
    """

    DOORWAY_PHRASES = (
        "through doorway", "behind doorway", "behind door",
        "through door", "through opening", "in doorway",
    )
    BED_FLOOR_WORDS = {"rug", "carpet", "floor", "mat", "stairs", "stair"}
    BED_WORDS = {
        "bed", "mattress", "sheet", "sheets", "bedding",
        "pillow", "pillows", "blanket", "headboard",
    }
    RACK_IMPOSSIBLE = ("above doorway", "on ceiling", "on wall")

    @staticmethod
    def _relation_text(obj: Dict[str, Any]) -> str:
        return " ".join(str(x).lower() for x in obj.get("spatial_relation", []) or [])

    @staticmethod
    def _description_text(obj: Dict[str, Any]) -> str:
        attrs = obj.get("attributes") or {}
        desc = attrs.get("description", "") if isinstance(attrs, dict) else ""
        if isinstance(desc, dict):
            desc = desc.get("value", "")
        return str(desc or "").lower()

    @classmethod
    def doorway_view(cls, obj: Dict[str, Any]) -> bool:
        relation_text = cls._relation_text(obj)
        return any(phrase in relation_text for phrase in cls.DOORWAY_PHRASES)

    @classmethod
    def label_conflict(cls, obj: Dict[str, Any], goal: Optional[GoalNode]) -> bool:
        """True when label/description contradicts the goal (e.g. bed-on-floor)."""
        label = GoalRelevanceScorer._clean(obj.get("label"))
        goal_key = GoalRelevanceScorer._clean(getattr(goal, "target_object", ""))
        relation_text = cls._relation_text(obj)
        text = " ".join([label, cls._description_text(obj), relation_text])
        if goal_key == "bed" or label == "bed":
            if any(word in relation_text for word in cls.BED_FLOOR_WORDS):
                return True
            if any(word in text for word in cls.BED_FLOOR_WORDS) and not any(
                word in text for word in cls.BED_WORDS
            ):
                return True
            if any(word in label for word in cls.BED_FLOOR_WORDS):
                return True
        if goal_key == "rack" or label == "rack":
            if any(phrase in text for phrase in cls.RACK_IMPOSSIBLE):
                return True
        return False

    @classmethod
    def is_usable_detection(cls, obj: Dict[str, Any], goal: Optional[GoalNode]) -> bool:
        """Full geometric + semantic gate for treating a detection as the goal."""
        bbox_conf = str(obj.get("bbox_confidence", "")).lower()
        bearing = str(obj.get("bearing", "")).lower()
        visibility = str(obj.get("visibility", "")).lower()
        center = bbox_center_x(obj.get("bbox"))
        area = bbox_area(obj.get("bbox"))
        side_bearing = bearing in {"left", "right"}

        if area <= 0.0 or bbox_conf == "low":
            return False
        if visibility not in {"visible", "clear"}:
            return False
        if bbox_edge_clipped(obj.get("bbox")) and side_bearing:
            return False
        if center is not None and not 0.12 <= center <= 0.88:
            return False
        if cls.label_conflict(obj, goal):
            return False

        goal_key = GoalRelevanceScorer._clean(getattr(goal, "target_object", ""))
        relation_text = cls._relation_text(obj)
        if goal_key == "rack" and "doorway" in relation_text and (
            bbox_conf != "high" or side_bearing
        ):
            return False
        if cls.doorway_view(obj) and (bbox_conf != "high" or side_bearing):
            return False
        return True


@dataclass
class ObjectPromotionPolicy:
    min_candidate_confidence: float = 0.70
    high_confidence: float = 0.90
    repeated_confidence: float = 0.72
    min_bbox_area: float = 0.01
    max_bbox_area: float = 0.45
    blocked_labels: Set[str] = field(default_factory=lambda: {
        "wall", "floor", "ceiling", "rug", "carpet", "doorway", "door frame",
        "door", "wooden door", "wooden door frame", "hallway corridor",
        "corridor", "staircase", "stairs", "handrail", "wooden handrail",
        "floor rug", "floor mat", "tile floor", "floor tile",
        "ceiling light", "ceiling light fixture", "light fixture",
        "light switch",
        "window", "window blind", "window blinds", "blinds", "curtain", "curtains",
        "picture frame", "photo frame", "painting", "artwork", "wall art", "poster",
        "framed artwork", "framed picture", "framed pictures",
        "sticker", "magnet", "deer head mount", "animal head mount",
    })

    def is_scene_surface_or_portal(self, label: str) -> bool:
        key = GoalRelevanceScorer._clean(label)
        if key in self.blocked_labels:
            return True
        return any(
            term in key
            for term in (
                "wall", "floor", "ceiling", "doorway", "door frame",
                "door", "window blind", "curtain", "rug", "carpet",
                "floor mat", "light switch", "ceiling light", "corridor",
            )
        )

    def label_description_conflict(self, obj: Dict[str, Any], *, goal: Optional[GoalNode]) -> bool:
        return GoalEvidenceFilter.label_conflict(obj, goal)

    def should_promote_object(self, obj: Dict[str, Any], *, seen_count: int) -> bool:
        label = GoalRelevanceScorer._clean(obj.get("label"))
        if not label or self.is_scene_surface_or_portal(label):
            return False
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        visibility = str(obj.get("visibility", "")).lower()
        bbox_confidence = str(obj.get("bbox_confidence", "")).lower()
        area = bbox_area(obj.get("bbox"))
        center = bbox_center_x(obj.get("bbox"))
        if visibility in {"occluded", "not_visible"}:
            return False
        has_valid_bbox = area > 0.0
        if not has_valid_bbox:
            return seen_count >= 2 and bbox_confidence != "low" and confidence >= self.repeated_confidence
        if bbox_confidence == "low" or area < self.min_bbox_area or area > self.max_bbox_area:
            return False
        if bbox_edge_clipped(obj.get("bbox")):
            return False
        if center is not None and not 0.10 <= center <= 0.90:
            return False
        if confidence >= self.min_candidate_confidence:
            return True
        if seen_count >= 2 and confidence >= self.repeated_confidence:
            return True
        return False


class Agent1MemoryWriter:
    """Writes perception into the two-layer DynamicTopoMap."""

    def __init__(self, topo_map: DynamicTopoMap):
        self.topo_map = topo_map
        self.hypothesis_pool = HypothesisPool(default_ttl=16, verify_seen_count=2, verify_confidence=0.45)
        self.promotion = ObjectPromotionPolicy()
        self.current_waypoint_id: Optional[str] = None
        self.previous_waypoint_id: Optional[str] = None
        self.frontier_probe: Optional[Callable[[np.ndarray, Optional[str]], Dict[str, Any]]] = None
        self.last_frontier_rejected: List[Dict[str, Any]] = []
        self.last_frontier_probes: List[Dict[str, Any]] = []
        self.object_candidate_counts: Dict[str, int] = {}

    def reset_runtime(self) -> None:
        self.current_waypoint_id = None
        self.previous_waypoint_id = None
        self.last_frontier_rejected = []
        self.last_frontier_probes = []
        self.object_candidate_counts = {}

    def set_frontier_probe(
        self,
        probe_fn: Optional[Callable[[np.ndarray, Optional[str]], Dict[str, Any]]],
    ) -> None:
        self.frontier_probe = probe_fn

    def record_object_verification(
        self,
        node_id: Optional[str],
        *,
        success: bool,
        detection_score: float = 0.0,
        strong_negative_step: float = 0.5,
    ) -> Optional[float]:
        """Commit the outcome of an ActiveTopo VERIFY cycle onto an object node.

        Uses ``confidence.py`` (no new confidence model):
          success -> consistent re-observation, confidence rises toward the new
                     detection, multi_view_count increments.
          failure -> inconsistent re-observation, confidence drops and
                     strong_negative_evidence accumulates; if it collapses the
                     node is consumed so planning stops chasing it.
        Returns the new confidence, or None when the node is missing / non-object.
        """
        if node_id is None:
            return None
        node = self.topo_map.get_node(str(node_id))
        if node is None or node.node_type != NodeType.OBJECT:
            return None
        attrs = node.attributes
        if success:
            attrs["multi_view_count"] = int(attrs.get("multi_view_count", node.visit_count or 1) or 1) + 1
            attrs["verify_success_count"] = int(attrs.get("verify_success_count", 0) or 0) + 1
            target = max(float(detection_score), float(node.confidence))
            new_conf = update_on_observation(float(node.confidence), target, is_consistent=True)
        else:
            attrs["verify_fail_count"] = int(attrs.get("verify_fail_count", 0) or 0) + 1
            prev_neg = float(attrs.get("strong_negative_evidence", 0.0) or 0.0)
            attrs["strong_negative_evidence"] = min(1.0, prev_neg + float(strong_negative_step))
            new_conf = update_on_observation(float(node.confidence), float(detection_score), is_consistent=False)
        node.confidence = float(new_conf)
        attrs["last_verify_step"] = int(self.topo_map.current_step)
        state = update_memory_state(
            confidence=float(new_conf),
            current_state=str(attrs.get("memory_state", "candidate")),
            strong_negative_evidence=float(attrs.get("strong_negative_evidence", 0.0) or 0.0),
            multi_view_count=int(attrs.get("multi_view_count", 1) or 1),
            target_relevance=float(attrs.get("target_relevance", 0.0) or 0.0),
        )
        attrs["memory_state"] = state
        if state in {"rejected", "expired"}:
            self.topo_map.consume_node(str(node_id), reason="verify_failed")
        return float(new_conf)

    def update(
        self,
        *,
        pose: AgentPose,
        rgb_embed: Optional[np.ndarray],
        clip_report: PerceptionReport,
        vlm_report: Optional[PerceptionReport],
        goal: Optional[GoalNode],
        rgb: Optional[np.ndarray] = None,
        object_crop_encoder: Optional[Callable[[Optional[np.ndarray], Any], Optional[np.ndarray]]] = None,
        object_text_encoder: Optional[Callable[[str, str], Optional[np.ndarray]]] = None,
        allow_frontier_generation: bool = True,
    ) -> str:
        self.last_frontier_rejected = []
        self.last_frontier_probes = []
        waypoint_id = self._write_waypoint(pose, rgb_embed, clip_report)
        self._write_room(waypoint_id, clip_report, goal)
        self._write_clip_hypotheses(waypoint_id, pose, clip_report, goal)
        if vlm_report is not None and vlm_report.is_full:
            self._write_vlm_report(
                waypoint_id,
                pose,
                vlm_report,
                goal,
                rgb=rgb,
                object_crop_encoder=object_crop_encoder,
                object_text_encoder=object_text_encoder,
            )
        if allow_frontier_generation:
            self._write_frontiers(waypoint_id, pose, goal, vlm_report=vlm_report)
        self.hypothesis_pool.decay(self.topo_map.current_step)
        return waypoint_id

    def _write_waypoint(self, pose: AgentPose, rgb_embed: Optional[np.ndarray], report: PerceptionReport) -> str:
        attrs = {
            "room_label": str(report.room_label or "unknown"),
            "room_confidence": float(report.room_confidence),
        }
        waypoint_id = self.topo_map.navigation_layer.add_or_update_waypoint(
            pose.position,
            heading=pose.heading,
            embedding=rgb_embed,
            attributes=attrs,
        )
        if self.current_waypoint_id and self.current_waypoint_id != waypoint_id:
            self.topo_map.add_edge(self.current_waypoint_id, waypoint_id, EdgeType.NAVIGABLE)
        self.previous_waypoint_id = self.current_waypoint_id
        self.current_waypoint_id = waypoint_id
        return waypoint_id

    def _write_room(self, waypoint_id: str, report: PerceptionReport, goal: Optional[GoalNode]) -> None:
        scores = dict(report.room_scores or [])
        if report.room_label and report.room_label != "unknown":
            scores[report.room_label] = max(float(scores.get(report.room_label, 0.0)), float(report.room_confidence))
        room_id = self.topo_map.structure_layer.add_or_update_room_region(
            waypoint_id=waypoint_id,
            room_scores=scores,
            goal_room_prior=list(goal.room_prior or []) if goal is not None else None,
        )
        if room_id:
            waypoint = self.topo_map.get_node(waypoint_id)
            room = self.topo_map.get_node(room_id)
            if waypoint is not None and room is not None:
                waypoint.attributes["room_region_id"] = room_id

    def _write_clip_hypotheses(
        self,
        waypoint_id: str,
        pose: AgentPose,
        report: PerceptionReport,
        goal: Optional[GoalNode],
    ) -> None:
        goal_id = _goal_id(goal)
        if report.goal_scores:
            label, score = report.goal_scores[0]
            if float(score) >= 0.20:
                self.hypothesis_pool.add_or_update(Hypothesis(
                    id="",
                    goal_id=goal_id,
                    kind="goal_view",
                    label=str(label),
                    source="clip_goal",
                    anchor_node_id=waypoint_id,
                    position=pose.position.copy(),
                    score=float(score),
                    confidence=float(score),
                    first_seen_step=self.topo_map.current_step,
                    last_seen_step=self.topo_map.current_step,
                ))
        if report.landmark_scores:
            label, score = report.landmark_scores[0]
            if float(score) >= 0.20:
                self.hypothesis_pool.add_or_update(Hypothesis(
                    id="",
                    goal_id=goal_id,
                    kind="landmark",
                    label=str(label),
                    source="clip_landmark",
                    anchor_node_id=waypoint_id,
                    position=pose.position.copy(),
                    score=float(score),
                    confidence=float(score),
                    first_seen_step=self.topo_map.current_step,
                    last_seen_step=self.topo_map.current_step,
                ))

    def _write_vlm_report(
        self,
        waypoint_id: str,
        pose: AgentPose,
        report: PerceptionReport,
        goal: Optional[GoalNode],
        *,
        rgb: Optional[np.ndarray] = None,
        object_crop_encoder: Optional[Callable[[Optional[np.ndarray], Any], Optional[np.ndarray]]] = None,
        object_text_encoder: Optional[Callable[[str, str], Optional[np.ndarray]]] = None,
    ) -> None:
        if report.room_label and report.room_label != "unknown":
            self._write_room(waypoint_id, report, goal)
        report_mode = str((report.raw or {}).get("_mode", "explore")).lower() if isinstance(report.raw, dict) else "explore"
        for obj_obs in report.objects:
            obj = obj_obs.to_dict()
            label = str(obj.get("label", "")).strip()
            if not label:
                continue
            seen_count = self._record_object_candidate(label)
            self._add_weak_object_hypothesis(waypoint_id, pose, obj, goal)
            if not self.promotion.should_promote_object(obj, seen_count=seen_count):
                continue
            estimated_position = estimate_object_position(pose, obj)
            position = pose.position.copy()
            visual_embedding = _embedding_from_value(obj.get("embedding"))
            if visual_embedding is None and object_crop_encoder is not None:
                visual_embedding = object_crop_encoder(rgb, obj.get("bbox"))
            raw_attrs = dict(obj.get("attributes") or {})
            description = _object_description(raw_attrs)
            text_embedding = None
            if object_text_encoder is not None:
                text_embedding = object_text_encoder(label, description)
            object_attrs = dict(raw_attrs)
            object_attrs.update({
                "raw_label": label,
                "description": description,
                "visual_embedding_source": "clip_crop" if visual_embedding is not None else "none",
                "text_embedding_source": "clip_text" if text_embedding is not None else "none",
            })
            if text_embedding is not None:
                object_attrs["text_embedding"] = text_embedding.tolist()
            node_id, _merged = self.topo_map.object_memory.upsert(
                label=label,
                bbox=obj.get("bbox"),
                confidence=float(obj.get("confidence", 0.0) or 0.0),
                position=position,
                embedding=visual_embedding,
                viewpoint_id=waypoint_id,
                view_heading=pose.heading,
                room_context=obj.get("room_context") or report.room_label,
                target_relevance=GoalRelevanceScorer.label_relevance(label, goal),
                source="vlm",
                spatial_attrs={
                    "range_bin": obj.get("range_bin", "unknown"),
                    "bearing": obj.get("bearing", "unknown"),
                    "visibility": obj.get("visibility", "unknown"),
                    "bbox_confidence": obj.get("bbox_confidence", "unknown"),
                    "observed_from_waypoint_id": waypoint_id,
                    "view_heading": pose.heading,
                    # Forward the VLM spatial phrase so the object node carries it
                    # for Phase G description/relation relevance scoring.
                    "spatial_relation": obj.get("spatial_relation", []),
                },
                object_attrs=object_attrs,
            )
            node = self.topo_map.get_node(node_id)
            if node is not None:
                node.attributes.setdefault("semantic_role", "object")
                node.attributes["layer"] = "structure"
                node.attributes["semantic_confirmed"] = True
                node.attributes["position_confirmed"] = False
                node.attributes["position_source"] = "observation_anchor"
                node.attributes["memory_state"] = "semantic_confirmed"
                node.attributes["estimated_object_position"] = estimated_position.tolist()
                node.attributes["last_seen_step"] = self.topo_map.current_step
                node.attributes["bbox"] = obj.get("bbox")
                node.attributes["view_heading"] = pose.heading
                node.attributes["bearing"] = obj.get("bearing", "unknown")
                node.attributes["range_bin"] = obj.get("range_bin", "unknown")
                node.attributes["visibility"] = obj.get("visibility", "unknown")
                node.attributes["seen_count"] = int(node.attributes.get("multi_view_count", 1) or 1)
                node.attributes["visual_embedding_source"] = object_attrs["visual_embedding_source"]
                node.attributes["text_embedding_source"] = object_attrs["text_embedding_source"]
            self.topo_map.cross_layer.bind_object_to_anchor(node_id, waypoint_id)
            self.topo_map.cross_layer.bind_object_observation(node_id, waypoint_id)
            room_id = self._room_id_for_waypoint(waypoint_id)
            if room_id:
                self.topo_map.cross_layer.bind_room_contains_object(room_id, node_id)
            self.hypothesis_pool.promote_matching(
                goal_id=_goal_id(goal),
                label=label,
                anchor_node_id=waypoint_id,
                object_node_id=node_id,
            )
        if report_mode == "explore":
            for portal in report.portals[:2]:
                if self._valid_portal(portal):
                    self._write_portal(waypoint_id, pose, portal, report.room_label)

    def _add_weak_object_hypothesis(
        self,
        waypoint_id: str,
        pose: AgentPose,
        obj: Dict[str, Any],
        goal: Optional[GoalNode],
    ) -> None:
        confidence = float(obj.get("confidence", 0.0) or 0.0)
        if confidence <= 0.0:
            return
        label = str(obj.get("label", ""))
        relevance = GoalRelevanceScorer.label_relevance(label, goal)
        if relevance < 0.65:
            return
        if self.promotion.label_description_conflict(obj, goal=goal):
            return
        self.hypothesis_pool.add_or_update(Hypothesis(
            id="",
            goal_id=_goal_id(goal),
            kind="object",
            label=label,
            source="vlm_object",
            anchor_node_id=waypoint_id,
            position=pose.position.copy(),
            score=confidence,
            confidence=confidence,
            weak_bbox=obj.get("bbox"),
            attributes={
                "range_bin": obj.get("range_bin", "unknown"),
                "visibility": obj.get("visibility", "unknown"),
                "goal_relevance": relevance,
            },
            first_seen_step=self.topo_map.current_step,
            last_seen_step=self.topo_map.current_step,
        ))

    def _valid_portal(self, portal: Any) -> bool:
        if isinstance(portal, dict):
            portal_type = GoalRelevanceScorer._clean(portal.get("type", portal.get("label", "")))
            label = GoalRelevanceScorer._clean(portal.get("label", ""))
            key = " ".join(part for part in (portal_type, label) if part)
        else:
            key = GoalRelevanceScorer._clean(portal)
        if not key or len(key) > 80:
            return False
        positive = ("doorway", "opening", "open door", "closed door", "door ahead", "door", "passage", "hallway", "corridor", "stair")
        negative = ("fridge", "refrigerator", "cabinet", "window", "picture")
        return any(term in key for term in positive) and not any(term in key for term in negative)

    def _write_portal(self, waypoint_id: str, pose: AgentPose, portal_data: Any, room_label: str) -> None:
        portal = dict(portal_data) if isinstance(portal_data, dict) else {"label": str(portal_data)}
        label = str(portal.get("label") or _portal_label(portal))
        bearing = str(portal.get("bearing", "unknown")).lower()
        state = str(portal.get("state", "uncertain")).lower()
        portal_type = str(portal.get("type", "unknown")).lower()
        traversable = bool(portal.get("traversable", state == "open"))
        pos = pose.position.copy()
        attrs = {
            "room_hint": room_label,
            "portal_requires_beyond_waypoint": False,
            "position_source": "observation_anchor",
            "position_confirmed": False,
            "portal_type": portal_type,
            "portal_state": state,
            "bearing": bearing,
            "traversable": traversable,
        }
        portal_id = self.topo_map.structure_layer.add_or_update_portal_node(
            label=label or "portal",
            confidence=0.55,
            estimated_position=pos.astype(np.float32),
            observed_from_waypoint_id=waypoint_id,
            connected_waypoint_id=waypoint_id,
            attributes=attrs,
        )
        portal = self.topo_map.get_node(portal_id)
        if portal is not None:
            portal.attributes.pop("beyond_frontier_id", None)
            connected = [str(x) for x in portal.attributes.get("connected_waypoints", []) or []]
            if waypoint_id not in connected:
                connected.append(waypoint_id)
            portal.attributes["connected_waypoints"] = connected[-8:]
        self.topo_map.cross_layer.bind_portal_connection(portal_id, waypoint_id)

    def _add_frontier_candidate(
        self,
        *,
        waypoint_id: str,
        position: np.ndarray,
        confidence: float,
        attrs: Dict[str, Any],
        reject_radius: float = 0.75,
    ) -> Optional[str]:
        pos = np.asarray(position, dtype=np.float32)
        if self.topo_map.has_nearby_visited(pos, radius=reject_radius):
            self.last_frontier_rejected.append({
                "direction_index": attrs.get("direction_index"),
                "direction_label": attrs.get("direction_label", ""),
                "candidate_position": pos.astype(np.float32).round(4).tolist(),
                "reason": "too_near_visited",
            })
            return None
        if self.frontier_probe is not None:
            probe = dict(self.frontier_probe(pos.astype(np.float32), None) or {})
            probe["direction_index"] = attrs.get("direction_index")
            probe["direction_label"] = attrs.get("direction_label", "")
            self.last_frontier_probes.append(probe)
            if probe.get("reason") == "reachability_unavailable":
                attrs.update({
                    "reachability": "unknown",
                    "reachability_reason": "reachability_unavailable",
                    "frontier_source": attrs.get("source", "geometric_fallback"),
                })
                return self.topo_map.navigation_layer.add_or_update_frontier(
                    pos.astype(np.float32),
                    created_from_waypoint=waypoint_id,
                    confidence=confidence,
                    attributes=attrs,
                )
            if not bool(probe.get("reachable", False)):
                row = dict(probe)
                row.setdefault("reason", "unreachable")
                row.setdefault("candidate_position", pos.astype(np.float32).round(4).tolist())
                self.last_frontier_rejected.append(row)
                return None
            target_pos = probe.get("target_position") or probe.get("target_position_relative")
            if target_pos is not None:
                pos = np.asarray(target_pos, dtype=np.float32)
            if self.topo_map.has_nearby_visited(pos, radius=reject_radius):
                row = dict(probe)
                row["reason"] = "too_near_visited"
                row["candidate_position"] = pos.astype(np.float32).round(4).tolist()
                self.last_frontier_rejected.append(row)
                return None
            attrs.update({
                "reachability": "reachable",
                "geodesic_distance": float(probe.get("geodesic_distance", 0.0) or 0.0),
                "path_points_relative": list(probe.get("path_points_relative", []) or []),
                "snapped_from": probe.get("snapped_from", None),
                "frontier_source": attrs.get("source", "navmesh_probe"),
            })
        else:
            attrs.update({
                "reachability": "unknown",
                "reachability_reason": "reachability_unavailable",
                "frontier_source": attrs.get("source", "geometric_fallback"),
            })
        return self.topo_map.navigation_layer.add_or_update_frontier(
            pos.astype(np.float32),
            created_from_waypoint=waypoint_id,
            confidence=confidence,
            attributes=attrs,
        )

    def _write_frontiers(
        self,
        waypoint_id: str,
        pose: AgentPose,
        goal: Optional[GoalNode],
        *,
        vlm_report: Optional[PerceptionReport] = None,
    ) -> None:
        room_probs = self._current_room_probs(waypoint_id)
        report = vlm_report if vlm_report is not None and vlm_report.is_full else None
        for idx, (offset, direction_label) in enumerate(_FRONTIER_DIRECTIONS):
            heading = pose.heading + offset
            forward = np.array([-np.sin(heading), 0.0, -np.cos(heading)], dtype=np.float32)
            pos = pose.position + forward * 2.3
            direction_room_probs = _directional_room_probs(room_probs, goal, report, direction_label)
            semantic_value = _directional_semantic_value(direction_room_probs, goal, report, direction_label)
            attrs = {
                "direction_index": idx,
                "direction_label": direction_label,
                "expected_room_probs": direction_room_probs,
                "semantic_value": semantic_value,
                "visual_semantic_reason": (
                    f"target_direction={getattr(report, 'target_direction', 'none')}; "
                    f"goal_visible={bool(getattr(report, 'goal_visible', False))}; "
                    f"portals={len(getattr(report, 'portals', []) or []) if report is not None else 0}"
                ),
                "source": "directional_frontier",
            }
            self._add_frontier_candidate(
                waypoint_id=waypoint_id,
                position=pos.astype(np.float32),
                confidence=0.42 + 0.10 * semantic_value,
                attrs=attrs,
            )

    def _current_room_probs(self, waypoint_id: str) -> Dict[str, float]:
        waypoint = self.topo_map.get_node(waypoint_id)
        if waypoint is None:
            return {}
        room_id = waypoint.attributes.get("room_region_id") or waypoint.attributes.get("room_id")
        room = self.topo_map.get_node(str(room_id)) if room_id else None
        return dict(room.attributes.get("room_probs", {})) if room is not None else {}

    def _room_id_for_waypoint(self, waypoint_id: str) -> Optional[str]:
        waypoint = self.topo_map.get_node(waypoint_id)
        if waypoint is None:
            return None
        room_id = waypoint.attributes.get("room_region_id") or waypoint.attributes.get("room_id")
        return str(room_id) if room_id else None

    def _record_object_candidate(self, label: str) -> int:
        key = GoalRelevanceScorer._clean(label)
        if not key:
            return 0
        self.object_candidate_counts[key] = int(self.object_candidate_counts.get(key, 0) or 0) + 1
        return self.object_candidate_counts[key]

    def to_debug(self) -> Dict[str, Any]:
        return {
            "hypotheses": self.hypothesis_pool.to_debug_list(limit=12),
            "object_memory": [
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "confidence": round(float(node.confidence), 4),
                    "anchor_waypoint_id": node.attributes.get("best_anchor_waypoint_id") or node.attributes.get("anchor_waypoint_id"),
                    "observed_from_waypoint_ids": list(node.attributes.get("observed_from_waypoint_ids", [])),
                    "failed_count": int(node.attributes.get("failed_count", 0) or 0),
                    "unreachable_count": int(node.attributes.get("unreachable_count", 0) or 0),
                }
                for node in self.topo_map.object_memory.get_all()
            ],
            "failed_targets": list(self.topo_map.failed_target_history[-8:]),
        }


def frontier_semantic_value(room_probs: Dict[str, float], goal: Optional[GoalNode]) -> float:
    if goal is None:
        return 0.0
    value = 0.0
    for room in goal.room_prior or []:
        value = max(value, float(room_probs.get(str(room), 0.0)))
    return float(min(1.0, value))


def _goal_id(goal: Optional[GoalNode]) -> Optional[str]:
    if goal is None:
        return None
    return str(goal.image_cache_key or goal.goal_image_id or goal.target_object)
