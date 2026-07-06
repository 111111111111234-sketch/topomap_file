"""Unified perception output for all ConfTopo backends (CLIP, GroundingDINO, VLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from conftopo.perception.heavy_perceiver import ObjectObservation


@dataclass
class PerceptionReport:
    """Single-step perception result consumed by MemoryAgent and PlannerAgent.

    Two modes of population:
    - **light** (every step): only room/goal/landmark scores + visual_embed.
      ``is_full`` is False and ``objects`` is empty.
    - **full** (triggered): additionally contains ``objects``, ``scene_summary``,
      ``portals``, ``goal_visible``, etc.  ``is_full`` is True.
    """

    # --- Room classification (from CLIP or VLM) ---
    room_label: str = "unknown"
    room_confidence: float = 0.0
    room_scores: List[Tuple[str, float]] = field(default_factory=list)

    # --- Goal / landmark matching ---
    goal_scores: List[Tuple[str, float]] = field(default_factory=list)
    landmark_scores: List[Tuple[str, float]] = field(default_factory=list)
    best_goal_sim: float = 0.0
    best_landmark_sim: float = 0.0

    # --- Object detections (heavy / VLM only) ---
    objects: List[ObjectObservation] = field(default_factory=list)

    # --- VLM-enhanced fields (only when source == "vlm") ---
    scene_summary: str = ""
    goal_visible: bool = False
    goal_match_confidence: float = 0.0
    target_direction: str = "unknown"
    target_visibility: str = "not_visible"
    apparent_scale: str = "unknown"
    relative_progress: str = "uncertain"
    stop_candidate: bool = False
    recommended_action: str = "search"
    goal_reason: str = ""
    portals: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty: float = 0.0

    # --- Meta ---
    visual_embed: Optional[np.ndarray] = None
    source: str = "none"  # "clip", "clip_groundingdino", "vlm", "none"
    is_full: bool = False
    step_id: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_label": self.room_label,
            "room_confidence": float(self.room_confidence),
            "room_scores": [(l, float(s)) for l, s in self.room_scores],
            "goal_scores": [(l, float(s)) for l, s in self.goal_scores],
            "landmark_scores": [(l, float(s)) for l, s in self.landmark_scores],
            "best_goal_sim": float(self.best_goal_sim),
            "best_landmark_sim": float(self.best_landmark_sim),
            "objects": [o.to_dict() for o in self.objects],
            "scene_summary": self.scene_summary,
            "goal_visible": self.goal_visible,
            "goal_match_confidence": float(self.goal_match_confidence),
            "target_direction": self.target_direction,
            "target_visibility": self.target_visibility,
            "apparent_scale": self.apparent_scale,
            "relative_progress": self.relative_progress,
            "stop_candidate": self.stop_candidate,
            "recommended_action": self.recommended_action,
            "goal_reason": self.goal_reason,
            "portals": [dict(p) if isinstance(p, dict) else {"label": str(p)} for p in self.portals],
            "uncertainty": float(self.uncertainty),
            "visual_embed": (
                self.visual_embed.tolist() if self.visual_embed is not None else None
            ),
            "source": self.source,
            "is_full": self.is_full,
            "step_id": int(self.step_id),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionReport":
        embed = data.get("visual_embed")
        return cls(
            room_label=str(data.get("room_label", "unknown")),
            room_confidence=float(data.get("room_confidence", 0.0)),
            room_scores=[
                (str(l), float(s)) for l, s in data.get("room_scores", [])
            ],
            goal_scores=[
                (str(l), float(s)) for l, s in data.get("goal_scores", [])
            ],
            landmark_scores=[
                (str(l), float(s)) for l, s in data.get("landmark_scores", [])
            ],
            best_goal_sim=float(data.get("best_goal_sim", 0.0)),
            best_landmark_sim=float(data.get("best_landmark_sim", 0.0)),
            objects=[
                ObjectObservation.from_dict(o) for o in data.get("objects", [])
            ],
            scene_summary=str(data.get("scene_summary", "")),
            goal_visible=bool(data.get("goal_visible", False)),
            goal_match_confidence=float(data.get("goal_match_confidence", 0.0)),
            target_direction=str(data.get("target_direction", "unknown")),
            target_visibility=str(data.get("target_visibility", "not_visible")),
            apparent_scale=str(data.get("apparent_scale", "unknown")),
            relative_progress=str(data.get("relative_progress", "uncertain")),
            stop_candidate=bool(data.get("stop_candidate", False)),
            recommended_action=str(data.get("recommended_action", "search")),
            goal_reason=str(data.get("goal_reason", "")),
            portals=[
                dict(p) if isinstance(p, dict) else {"label": str(p)}
                for p in data.get("portals", [])
            ],
            uncertainty=float(data.get("uncertainty", 0.0)),
            visual_embed=(
                np.array(embed, dtype=np.float32) if embed is not None else None
            ),
            source=str(data.get("source", "none")),
            is_full=bool(data.get("is_full", False)),
            step_id=int(data.get("step_id", 0)),
        )


EMPTY = PerceptionReport()
