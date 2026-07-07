"""ObjectMemoryPool: manages object entity memory for ConfTopo.

Separated from DynamicTopoMap so object merge / bbox history / confidence /
anchor logic lives in a single responsibility boundary. DynamicTopoMap retains
graph-edge ownership (OBSERVED_AT, BELONGS_TO, etc.) and delegates object
entity operations here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from conftopo.core.confidence import (
    ConfidenceFactors,
    compute_semantic_confidence,
    update_memory_state,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conftopo.core.dynamic_topo_map import DynamicTopoMap, NodeType, SemanticNode


def _node_type():
    from conftopo.core.dynamic_topo_map import NodeType
    return NodeType

# Desired stand-off distance (metres) between the agent's approach point and the
# object. The approach point is placed this far back from the object along the
# navigable viewing side; the pathfinder then snaps it onto the navmesh. Kept
# small so the agent actually closes in on the target instead of stopping at a
# comfortable viewing distance.
DESIRED_STANDOFF = 0.7

# Local copies of utility functions to avoid circular imports
def _bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
    except (TypeError, ValueError):
        return 0.0
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else float(inter / denom)


def _embedding_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a)
    b = np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _angle_delta(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def _range_bin_rank(range_bin: Optional[str]) -> int:
    if not range_bin:
        return 0
    key = str(range_bin).strip().lower()
    return {"close": 5, "very_near": 4, "near": 3, "medium": 2, "mid": 2, "far": 1}.get(key, 0)


def _bbox_area_from_list(bbox: Optional[List[float]]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    w = max(0.0, float(bbox[2]) - float(bbox[0]))
    h = max(0.0, float(bbox[3]) - float(bbox[1]))
    return w * h


def _normalize_angle(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


class ObjectMemoryPool:
    """Entity-level object memory, separate from the topological graph.

    The pool manages:
    - Object creation, update, merge, and removal.
    - Bbox history, multi-view tracking, and detection scores.
    - Confidence computation (via ConfidenceFactors).
    - Anchor-waypoint binding and promotion.
    - Approach-position computation.

    DynamicTopoMap owns graph operations; this pool owns object semantics.
    """

    def __init__(self, topo_map):  # type: ignore (DynamicTopoMap)
        self._map = topo_map
        self.object_history_keep_recent = 2

    # ── core upsert ─────────────────────────────────────────────────

    def upsert(
        self,
        *,
        label: str,
        bbox: Optional[List[float]],
        confidence: float,
        position: np.ndarray,
        embedding: Optional[np.ndarray] = None,
        viewpoint_id: Optional[str] = None,
        view_heading: float = 0.0,
        room_context: Optional[str] = None,
        target_relevance: float = 0.0,
        room_prior_score: float = 0.0,
        source: str = "heavy",
        spatial_attrs: Optional[Dict[str, Any]] = None,
        object_attrs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """Create or update an object node.

        Returns (node_id, merged_existing).
        """
        label = str(label)
        pos = np.array(position, dtype=np.float32)
        emb = np.array(embedding, dtype=np.float32) if embedding is not None else None

        matches = self._map.find_nodes_within_radius(
            pos, self._map.merge_radius * 2.0, _node_type().OBJECT,
        )
        match = self._best_match(matches, label, bbox, emb, view_heading, pos, room_context)

        if match is None:
            attrs = self._new_attributes(
                bbox=bbox, confidence=confidence,
                viewpoint_id=viewpoint_id, view_heading=view_heading,
                room_context=room_context, source=source,
                target_relevance=target_relevance, room_prior_score=room_prior_score,
                spatial_attrs=spatial_attrs, object_attrs=object_attrs,
            )
            node_id = self._map.add_node(
                _node_type().OBJECT,
                position=pos,
                embedding=emb,
                confidence=compute_semantic_confidence(ConfidenceFactors(
                    detection_score=confidence,
                    multi_view_count=1,
                    task_relevance=target_relevance,
                    room_prior_score=room_prior_score,
                )),
                label=label,
                attributes=attrs,
            )
            self.update_state(self._map._nodes[node_id], confirmed=confidence >= 0.55)
            self._map._nodes[node_id].attributes["best_approach_position"] = (
                self._compute_best_approach(node_id).tolist()
            )
            return node_id, False

        self._merge_observation(
            match, bbox=bbox, confidence=confidence, position=pos,
            embedding=emb, viewpoint_id=viewpoint_id, view_heading=view_heading,
            room_context=room_context, source=source,
            target_relevance=target_relevance, room_prior_score=room_prior_score,
            spatial_attrs=spatial_attrs, object_attrs=object_attrs,
        )
        match.attributes["best_approach_position"] = (
            self._compute_best_approach(match.node_id).tolist()
        )
        return match.node_id, True

    def update_state(
        self,
        node: "SemanticNode",
        *,
        confirmed: bool = False,
        rejected: bool = False,
        expired: bool = False,
    ) -> str:
        attrs = node.attributes
        state = update_memory_state(
            confidence=float(node.confidence),
            current_state=str(attrs.get("memory_state", "candidate")),
            confirmed=confirmed or str(attrs.get("semantic_role", "")) == "object_anchor",
            preserved=bool(attrs.get("cross_goal_preserved")),
            rejected=rejected,
            expired=expired,
            strong_negative_evidence=float(attrs.get("strong_negative_evidence", 0.0)),
            multi_view_count=int(attrs.get("multi_view_count", node.visit_count)),
            target_relevance=float(attrs.get("target_relevance", 0.0)),
        )
        attrs["memory_state"] = state
        attrs["memory_state_step"] = self._map.current_step
        return state

    # ── matching ────────────────────────────────────────────────────

    def _best_match(
        self,
        candidates: List[SemanticNode],
        label: str,
        bbox: Optional[List[float]],
        embedding: Optional[np.ndarray],
        view_heading: float,
        position: np.ndarray,
        room_context: Optional[str] = None,
    ) -> Optional[SemanticNode]:
        best = None
        best_score = -1.0
        for node in candidates:
            if node.label != label:
                continue
            if node.attributes.get("cross_goal_preserved"):
                continue
            if node.attributes.get("folded"):
                continue
            if not self._room_context_compatible(node, room_context):
                continue
            last_bbox = node.attributes.get("last_bbox") or bbox
            bbox_score = _bbox_iou(last_bbox, bbox)
            heading = float(node.attributes.get("last_view_heading", view_heading))
            heading_score = 1.0 if abs(_angle_delta(heading, view_heading)) <= 0.45 else 0.0
            emb_score = _embedding_similarity(node.embedding, embedding)
            dist = float(np.linalg.norm(node.position - position))
            score = max(bbox_score, heading_score * 0.8, emb_score)
            if dist < self._map.merge_radius:
                score = max(score, 0.5)
            if score > best_score and score >= 0.45:
                best = node
                best_score = score
        return best

    @staticmethod
    def _room_context_compatible(node: SemanticNode, room_context: Optional[str]) -> bool:
        if room_context is None:
            return True
        current = str(room_context).strip()
        if not current or current == "unknown":
            return True
        known_rooms = node.attributes.get("room_contexts")
        if known_rooms is None:
            previous = node.attributes.get("room_context")
            if previous is not None and str(previous).strip() != current:
                return False
            return True
        return current in known_rooms

    # ── attributes ──────────────────────────────────────────────────

    def _new_attributes(
        self,
        *,
        bbox: Optional[List[float]],
        confidence: float,
        viewpoint_id: Optional[str],
        view_heading: float,
        room_context: Optional[str],
        source: str,
        target_relevance: float,
        room_prior_score: float,
        spatial_attrs: Optional[Dict[str, Any]] = None,
        object_attrs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        obs = {
            "bbox": [float(v) for v in bbox] if bbox is not None else None,
            "confidence": float(confidence),
            "viewpoint_id": viewpoint_id,
            "view_heading": float(view_heading),
            "step_id": self._map.current_step,
            "source": source,
        }
        clean_spatial = self._clean_spatial(spatial_attrs)
        clean_obj = self._clean_object(object_attrs)
        obs.update(clean_spatial)
        if clean_obj:
            obs["attributes"] = dict(clean_obj)
        attrs = {
            "bbox_observations": [obs],
            "detection_scores": [float(confidence)],
            "viewpoints": [viewpoint_id] if viewpoint_id is not None else [],
            "first_seen_step": self._map.current_step,
            "last_seen_step": self._map.current_step,
            "multi_view_count": 1,
            "room_context": room_context,
            "room_contexts": [room_context] if room_context is not None else [],
            "granularity": "object",
            "last_bbox": [float(v) for v in bbox] if bbox is not None else None,
            "last_view_heading": float(view_heading),
            "target_relevance": float(target_relevance),
            "room_prior_score": float(room_prior_score),
            "redundancy_penalty": 0.0,
            "evidence_refs": [
                {
                    "step_id": self._map.current_step,
                    "frame_id": self._map.current_step,
                    "viewpoint_id": viewpoint_id,
                    "bbox": [float(v) for v in bbox] if bbox is not None else None,
                    "source": source,
                }
            ],
            "conflict_penalty": 0.0,
            "source": source,
            "memory_owner": "ObjectMemoryPool",
            "graph_role": "object_node_ref",
            "memory_state": "candidate",
            "memory_state_step": self._map.current_step,
        }
        attrs.update(clean_spatial)
        if clean_obj:
            attrs["object_attributes"] = dict(clean_obj)
        range_bin = clean_spatial.get("range_bin") or attrs.get("range_bin")
        self._record_best_anchor_score(attrs, bbox, range_bin, confidence)
        return attrs

    def _merge_observation(
        self,
        node: SemanticNode,
        *,
        bbox: Optional[List[float]],
        confidence: float,
        position: np.ndarray,
        embedding: Optional[np.ndarray],
        viewpoint_id: Optional[str],
        view_heading: float,
        room_context: Optional[str],
        source: str,
        target_relevance: float,
        room_prior_score: float,
        spatial_attrs: Optional[Dict[str, Any]] = None,
        object_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        attrs = node.attributes
        obs = {
            "bbox": [float(v) for v in bbox] if bbox is not None else None,
            "confidence": float(confidence),
            "viewpoint_id": viewpoint_id,
            "view_heading": float(view_heading),
            "step_id": self._map.current_step,
            "source": source,
        }
        clean_spatial = self._clean_spatial(spatial_attrs)
        clean_obj = self._clean_object(object_attrs)
        obs.update(clean_spatial)
        if clean_obj:
            obs["attributes"] = dict(clean_obj)
        attrs.setdefault("bbox_observations", []).append(obs)
        attrs.setdefault("evidence_refs", []).append({
            "step_id": self._map.current_step,
            "frame_id": self._map.current_step,
            "viewpoint_id": viewpoint_id,
            "bbox": [float(v) for v in bbox] if bbox is not None else None,
            "source": source,
        })
        attrs.setdefault("detection_scores", []).append(float(confidence))
        if viewpoint_id is not None and viewpoint_id not in attrs.setdefault("viewpoints", []):
            attrs["viewpoints"].append(viewpoint_id)
        attrs["last_seen_step"] = self._map.current_step
        if bbox is not None:
            attrs["last_bbox"] = [float(v) for v in bbox]
        attrs["last_view_heading"] = float(view_heading)
        attrs["multi_view_count"] = max(
            1, len(attrs.get("viewpoints", [])), len(attrs.get("bbox_observations", [])))
        if room_context is not None:
            attrs["room_context"] = room_context
            contexts = attrs.setdefault("room_contexts", [])
            if room_context not in contexts:
                contexts.append(room_context)
        attrs["target_relevance"] = max(float(attrs.get("target_relevance", 0.0)), float(target_relevance))
        attrs["room_prior_score"] = max(float(attrs.get("room_prior_score", 0.0)), float(room_prior_score))
        attrs.update(clean_spatial)
        if clean_obj:
            merged = attrs.setdefault("object_attributes", {})
            if isinstance(merged, dict):
                merged.update(clean_obj)
            else:
                attrs["object_attributes"] = dict(clean_obj)

        low_scores = [s for s in attrs.get("detection_scores", []) if float(s) < 0.35]
        attrs["redundancy_penalty"] = max(0.0, (len(low_scores) - 1) / 5.0)
        attrs["conflict_penalty"] = self._nearby_conflict(node, position)

        # Section 9: weak negative evidence = fraction of low-confidence observations
        all_scores = attrs.get("detection_scores", [confidence])
        neg_count = sum(1 for s in all_scores if float(s) < 0.2)
        attrs["negative_evidence"] = min(1.0, neg_count / max(len(all_scores), 1))

        # Section 9: strong negative evidence = VLM explicitly saw but with v. low confidence
        if source in ("vlm",) and float(confidence) < 0.3:
            attrs["strong_negative_evidence"] = float(attrs.get("strong_negative_evidence", 0.0)) + 0.10
        elif float(confidence) >= 0.3:
            # Recovering: reduce strong negative over time
            attrs["strong_negative_evidence"] = max(0.0, float(attrs.get("strong_negative_evidence", 0.0)) - 0.05)

        # Section 9: multi_frame_consistency = how many times same viewpoint confirmed
        # We track observed_viewpoints_times for this
        vp_counts = attrs.setdefault("viewpoint_observation_counts", {})
        if viewpoint_id is not None and source in ("vlm", "groundingdino", "heavy", "fresh_vlm"):
            vp_counts[viewpoint_id] = vp_counts.get(viewpoint_id, 0) + 1
            attrs["multi_frame_consistency"] = max(
                attrs.get("multi_frame_consistency", 0),
                vp_counts.get(viewpoint_id, 0),
            )

        # Section 10: attribute_confidence = avg of VLM attribute confidences
        obj_attrs = clean_obj if clean_obj else attrs.get("object_attributes", {})
        if isinstance(obj_attrs, dict) and obj_attrs:
            attr_confs = []
            for _ak in ("color", "shape", "material", "size", "state"):
                _av = obj_attrs.get(_ak, {})
                if isinstance(_av, dict):
                    attr_confs.append(float(_av.get("confidence", 0.0)))
                elif isinstance(_av, (int, float)):
                    attr_confs.append(float(_av))
            attrs["attribute_confidence"] = float(np.mean(attr_confs)) if attr_confs else 0.0
        else:
            attrs["attribute_confidence"] = attrs.get("attribute_confidence", 0.0)

        range_bin = clean_spatial.get("range_bin") or attrs.get("range_bin")
        if attrs.get("position_source") == "anchor_waypoint":
            self._map.promote_object_anchor_if_better(
                node, position, viewpoint_id, bbox, range_bin, confidence,
            )
        else:
            node.position = (
                node.position * max(1, node.visit_count) + position
            ) / (max(1, node.visit_count) + 1)
        node.visit_count += 1
        if embedding is not None:
            if node.embedding is None:
                node.embedding = embedding
            elif confidence >= max(attrs.get("detection_scores", [confidence])):
                node.embedding = embedding
            else:
                node.embedding = (0.8 * node.embedding + 0.2 * embedding).astype(np.float32)
        node.step_id = self._map.current_step

        section9_neg = float(attrs.get("negative_evidence", 0.0))
        section9_strong_neg = float(attrs.get("strong_negative_evidence", 0.0))
        section9_frame = int(attrs.get("multi_frame_consistency", 0))
        section9_attr = float(attrs.get("attribute_confidence", 0.0))

        node.confidence = compute_semantic_confidence(ConfidenceFactors(
            detection_score=max(float(s) for s in all_scores),
            multi_view_count=int(attrs["multi_view_count"]),
            task_relevance=float(attrs.get("target_relevance", 0.0)),
            room_prior_score=float(attrs.get("room_prior_score", 0.0)),
            redundancy_penalty=float(attrs.get("redundancy_penalty", 0.0)),
            conflict_penalty=float(attrs.get("conflict_penalty", 0.0)),
            negative_evidence=section9_neg,
            strong_negative_evidence=section9_strong_neg,
            multi_frame_consistency=section9_frame,
            attribute_confidence=section9_attr,
        ))
        self.update_state(node, confirmed=float(confidence) >= 0.55)

    @staticmethod
    def _clean_spatial(spatial_attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not spatial_attrs:
            return {}
        clean: Dict[str, Any] = {}
        for key in (
            "anchor_waypoint_id", "observed_from", "bearing", "range_bin",
            "visibility", "vlm_room_context", "position_source",
        ):
            value = spatial_attrs.get(key)
            if value is not None:
                clean[key] = str(value)
        if "visible" in spatial_attrs:
            clean["visible"] = bool(spatial_attrs.get("visible"))
        relation = spatial_attrs.get("spatial_relation", [])
        if isinstance(relation, str):
            relation = [relation]
        if relation is None:
            relation = []
        clean["spatial_relation"] = [str(x) for x in relation]
        return clean

    @staticmethod
    def _clean_object(object_attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(object_attrs, dict):
            return {}
        clean: Dict[str, Any] = {}
        for key, value in object_attrs.items():
            if value is None:
                continue
            text_key = str(key).strip()
            if not text_key:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[text_key] = value
            elif isinstance(value, (list, tuple)):
                clean[text_key] = [x for x in value if isinstance(x, (str, int, float, bool)) and str(x).strip()]
            else:
                clean[text_key] = str(value)
        return clean

    # ── anchor / approach ───────────────────────────────────────────

    def compute_best_approach(
        self, object_node_id: str, desired_stop: float = DESIRED_STANDOFF,
    ) -> np.ndarray:
        """Approach point = a small stand-off back from the object.

        The object node position is the estimated object location. We take the
        closest observation waypoint as a known-navigable reference on the
        viewing side, then place the approach point ``desired_stop`` metres back
        from the object toward that reference. The pathfinder snaps this onto the
        navmesh at probe time, so the agent ends up right next to the object
        instead of stopping at the original (far) viewing pose.
        """
        node = self._map.get_node(object_node_id)
        if node is None:
            return np.zeros(3, dtype=np.float32)
        obj_pos = np.asarray(node.position, dtype=np.float32)

        # Nearest observation waypoint = navigable reference on the viewing side.
        viewer_pos = None
        best_dist = float("inf")
        for vp_id in node.attributes.get("viewpoints", []):
            vp = self._map.get_node(vp_id)
            if vp is None:
                continue
            d = float(np.linalg.norm(vp.position - obj_pos))
            if d < best_dist:
                best_dist = d
                viewer_pos = np.asarray(vp.position, dtype=np.float32).copy()

        if viewer_pos is None:
            # No known viewer: hand the object position to the pathfinder, which
            # will snap it to the nearest navmesh point.
            return obj_pos.copy()

        delta = viewer_pos - obj_pos
        dist = float(np.linalg.norm(delta))
        if dist <= desired_stop or dist < 1e-6:
            # Viewer already inside the stand-off band; stand where we observed.
            return viewer_pos.astype(np.float32)

        standoff = obj_pos + (delta / dist) * desired_stop
        # Keep the floor height from a known navigable pose for a clean snap.
        standoff[1] = viewer_pos[1]
        return standoff.astype(np.float32)

    def _compute_best_approach(self, object_node_id: str) -> np.ndarray:
        return self.compute_best_approach(object_node_id)

    def anchor_position(self, node: SemanticNode) -> Optional[np.ndarray]:
        if node.attributes.get("position_source") != "anchor_waypoint":
            return None
        anchor_id = node.attributes.get("anchor_waypoint_id") or node.attributes.get("observed_from")
        anchor = self._map.get_node(anchor_id)
        if anchor is None:
            return None
        return anchor.position.copy()

    def _nearby_conflict(self, node: SemanticNode, position: np.ndarray) -> float:
        conflicts = 0
        for other in self._map.find_nodes_within_radius(
            position, self._map.merge_radius, _node_type().OBJECT,
        ):
            if other.node_id != node.node_id and other.label != node.label:
                conflicts += 1
        return min(1.0, conflicts / 3.0)

    def _observation_anchor_tuple(
        self, bbox: Optional[List[float]], range_bin: Optional[str], confidence: float,
    ) -> Tuple[int, float, float]:
        return (_range_bin_rank(range_bin), _bbox_area_from_list(bbox), float(confidence))

    def _stored_anchor_tuple(self, attrs: Dict[str, Any]) -> Tuple[int, float, float]:
        best = attrs.get("best_anchor_score")
        if isinstance(best, dict):
            return (int(best.get("range", 0)), float(best.get("bbox_area", 0.0)), float(best.get("confidence", 0.0)))
        return (
            _range_bin_rank(attrs.get("range_bin")),
            _bbox_area_from_list(attrs.get("last_bbox")),
            float(max(attrs.get("detection_scores", [0.0]))),
        )

    def _should_promote_anchor(
        self, attrs: Dict[str, Any], bbox: Optional[List[float]],
        range_bin: Optional[str], confidence: float,
    ) -> bool:
        return self._observation_anchor_tuple(bbox, range_bin, confidence) > self._stored_anchor_tuple(attrs)

    def _record_best_anchor_score(
        self, attrs: Dict[str, Any], bbox: Optional[List[float]],
        range_bin: Optional[str], confidence: float,
    ) -> None:
        rank, bbox_area, conf = self._observation_anchor_tuple(bbox, range_bin, confidence)
        attrs["best_anchor_score"] = {"range": rank, "bbox_area": bbox_area, "confidence": conf}
        attrs["best_anchor_step"] = self._map.current_step

    def promote_anchor_if_better(
        self, node: SemanticNode, position: np.ndarray,
        viewpoint_id: Optional[str], bbox: Optional[List[float]],
        range_bin: Optional[str], confidence: float,
    ) -> bool:
        if node.attributes.get("position_source") != "anchor_waypoint":
            return False
        if not self._should_promote_anchor(node.attributes, bbox, range_bin, confidence):
            return False
        node.position = position.copy()
        if viewpoint_id:
            node.attributes["anchor_waypoint_id"] = viewpoint_id
            node.attributes["observed_from"] = viewpoint_id
        self._record_best_anchor_score(node.attributes, bbox, range_bin, confidence)
        anchor_pos = self.anchor_position(node)
        best_pos = anchor_pos if anchor_pos is not None else self.compute_best_approach(node.node_id)
        node.attributes["best_approach_position"] = best_pos.tolist()
        return True

    # ── landmark promotion ──────────────────────────────────────────

    def merge_into_landmark(self, landmark: SemanticNode, obj: SemanticNode) -> None:
        landmark.attributes.setdefault("fused_object_ids", []).append(obj.node_id)
        landmark.attributes.setdefault("detection_scores", []).append(float(obj.confidence))
        landmark.visit_count += obj.visit_count
        landmark.confidence = max(float(landmark.confidence), float(obj.confidence))
        if obj.position is not None:
            landmark.position = (landmark.position + obj.position) * 0.5

    def promote_to_landmark_node(self, node: SemanticNode) -> None:
        from conftopo.core.landmark_roles import classify_landmark_role
        role = classify_landmark_role(node.label, node.attributes)
        node.attributes["promoted_from_object"] = True
        node.attributes["landmark_role"] = role
        node.attributes["promotion_step"] = self._map.current_step
        node.attributes.pop("semantic_role", None)
        if role == "structural":
            self._compress_object_history(node, reason="promoted_to_landmark")

    def fuse_object_to_landmark(self, node: SemanticNode) -> Optional[str]:
        from conftopo.core.landmark_roles import can_promote_object_to_landmark
        if not can_promote_object_to_landmark(node.label, node.attributes):
            return None
        nearby = self._map.find_nodes_within_radius(
            node.position, self._map.merge_radius * 3.0, _node_type().LANDMARK,
        )
        for landmark in nearby:
            if landmark.label == node.label:
                self.merge_into_landmark(landmark, node)
                self._map._nodes.pop(node.node_id, None)
                self._map.graph.remove_node(node.node_id)
                return landmark.node_id
        self._map._set_node_type(node, _node_type().LANDMARK)
        self.promote_to_landmark_node(node)
        return node.node_id

    # ── compression ─────────────────────────────────────────────────

    def compress_history(self, node: SemanticNode, reason: str) -> None:
        self._compress_object_history(node, reason)

    def _compress_object_history(self, node: SemanticNode, reason: str) -> None:
        attrs = node.attributes
        observations = list(attrs.get("bbox_observations", []))
        if len(observations) <= self.object_history_keep_recent:
            return
        scores = [float(obs.get("confidence", 0.0)) for obs in observations]
        best_idx = int(np.argmax(scores)) if scores else 0
        keep = {best_idx}
        keep.update(range(max(0, len(observations) - self.object_history_keep_recent), len(observations)))
        attrs["bbox_observations"] = [observations[i] for i in sorted(keep)]
        attrs["detection_scores"] = [float(observations[i].get("confidence", 0.0)) for i in sorted(keep)]
        attrs["viewpoints"] = [
            observations[i].get("viewpoint_id")
            for i in sorted(keep) if observations[i].get("viewpoint_id") is not None
        ]
        attrs["history_compressed"] = True
        attrs["history_compression_reason"] = reason
        attrs["history_original_observation_count"] = max(
            int(attrs.get("history_original_observation_count", 0)), len(observations),
        )
        attrs["history_kept_observation_count"] = len(attrs["bbox_observations"])
        attrs["history_best_confidence"] = max(scores) if scores else 0.0
        attrs["history_mean_confidence"] = float(np.mean(scores)) if scores else 0.0
        attrs["multi_view_count"] = max(
            int(attrs.get("multi_view_count", 1)),
            int(attrs["history_original_observation_count"]),
        )

    # ── query ───────────────────────────────────────────────────────

    def get_all(self) -> List[SemanticNode]:
        return self._map.get_nodes_by_type(_node_type().OBJECT)

    def get(self, node_id: str) -> Optional[SemanticNode]:
        return self._map.get_node(node_id)

    def find_within_radius(self, position: np.ndarray, radius: float) -> List[SemanticNode]:
        return self._map.find_nodes_within_radius(position, radius, _node_type().OBJECT)

    def clear(self) -> None:
        for node in self.get_all():
            self._map.remove_node(node.node_id)
