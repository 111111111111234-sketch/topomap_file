"""VLM-backed perceiver: wraps any VLMBackendBase into a PerceptionReport producer."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from conftopo.perception.heavy_perceiver import ObjectObservation, normalize_bbox_flexible
from conftopo.perception.perception_report import PerceptionReport
from conftopo.perception.vlm_backend import VLMBackendBase


def _estimated_display_bbox(bearing: str, range_bin: str) -> Optional[list[float]]:
    """Approximate an on-screen box for audit overlays when VLM omits bbox.

    This box is intentionally kept separate from the real detection bbox. It
    helps visualize what the language model claimed to see, but downstream
    stop verification still requires an actual bbox.
    """
    bearing_key = str(bearing or "").lower().strip()
    range_key = str(range_bin or "").lower().strip()
    if bearing_key in {"left", "left_front"}:
        cx = 0.28
    elif bearing_key in {"right", "right_front"}:
        cx = 0.72
    elif bearing_key in {"center", "front"}:
        cx = 0.50
    else:
        return None
    if range_key in {"very_near", "close"}:
        bw, bh, cy = 0.36, 0.46, 0.58
    elif range_key == "near":
        bw, bh, cy = 0.28, 0.34, 0.56
    elif range_key in {"mid", "medium"}:
        bw, bh, cy = 0.22, 0.26, 0.52
    else:
        bw, bh, cy = 0.18, 0.20, 0.50
    return [
        max(0.0, cx - bw / 2.0),
        max(0.0, cy - bh / 2.0),
        min(1.0, cx + bw / 2.0),
        min(1.0, cy + bh / 2.0),
    ]


class VLMPerceiver:
    """High-level perceiver that calls a VLM backend and returns a unified report.

    Usage::

        perceiver = VLMPerceiver(backend=Qwen3VLBackend(...))
        report = perceiver.perceive(rgb, "Find the kitchen sink", step_id=42)
    """

    def __init__(self, backend: VLMBackendBase):
        self._backend = backend

    def perceive(
        self,
        rgb: np.ndarray,
        goal_text: str,
        visual_embed: Optional[np.ndarray] = None,
        step_id: int = 0,
        context: Optional[str] = None,
        mode: str = "explore",
        previous_rgb: Optional[np.ndarray] = None,
        action_history: Optional[Sequence[str]] = None,
    ) -> PerceptionReport:
        """Query the VLM and convert the response to a PerceptionReport.

        *mode*: ``"explore"`` for discovery or ``"confirm"`` for strict
        stop verification.  Selects the system prompt sent to the VLM.
        """
        if not getattr(self._backend, 'supports_multi_image', True):
            previous_rgb = None
        raw = self._backend.query(
            rgb,
            goal_text,
            context=context,
            mode=mode,
            previous_rgb=previous_rgb,
            action_history=action_history,
        )
        raw = dict(raw)
        raw["_mode"] = mode
        if mode == "confirm" and previous_rgb is None:
            raw["relative_progress"] = "uncertain"
        return self._build_report(raw, visual_embed, step_id)

    @staticmethod
    def _build_report(
        data: dict,
        visual_embed: Optional[np.ndarray],
        step_id: int,
    ) -> PerceptionReport:
        room = data.get("room") or {}
        room_label = str(room.get("label", "unknown"))
        room_conf = float(room.get("confidence", 0.0))

        objects = []
        for o in data.get("objects", []):
            relation = o.get("relation", o.get("spatial_relation", []))
            if isinstance(relation, str):
                relation = [relation]
            if relation is None:
                relation = []
            confidence = max(0.0, min(1.0, float(o.get("confidence", 0.0) or 0.0)))
            raw_bbox = (
                o.get("bbox")
                or o.get("box")
                or o.get("bounding_box")
                or o.get("bounds")
                or o.get("coordinates")
                or o.get("bbox_2d")
            )
            bbox = normalize_bbox_flexible(raw_bbox)
            bearing = str(o.get("bearing", "unknown"))
            range_bin = str(o.get("range", o.get("range_bin", "unknown")))
            attributes = dict(o.get("attributes") or {})
            if bbox is None and bool(o.get("visible", True)):
                display_bbox = _estimated_display_bbox(bearing, range_bin)
                if display_bbox is not None:
                    attributes["display_bbox"] = display_bbox
                    attributes["estimated_bbox"] = True
            objects.append(ObjectObservation(
                label=str(o.get("label", "")),
                bbox=bbox,
                confidence=confidence,
                source="vlm",
                step_id=step_id,
                visible=bool(o.get("visible", True)),
                visibility=str(o.get("visibility", "unknown")),
                bearing=bearing,
                range_bin=range_bin,
                spatial_relation=[str(x) for x in relation],
                bbox_confidence=str(o.get("bbox_confidence", "medium")),
                room_context=o.get("room_context"),
                attributes=attributes,
            ))

        query_ok = bool(data.get("_query_ok", True))
        parse_ok = bool(data.get("_parse_ok", True))
        is_full = bool(query_ok and parse_ok)

        return PerceptionReport(
            room_label=room_label,
            room_confidence=room_conf,
            room_scores=[(room_label, room_conf)] if room_label != "unknown" else [],
            objects=objects,
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
                for p in (data.get("portals") or [])
            ],
            uncertainty=float(data.get("uncertainty", 0.5)),
            visual_embed=visual_embed,
            source="vlm",
            is_full=is_full,
            step_id=step_id,
            raw=data,
        )
