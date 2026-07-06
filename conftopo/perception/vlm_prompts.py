"""VLM prompt templates and response parsing for navigation perception."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from conftopo.perception.heavy_perceiver import normalize_bbox, normalize_bbox_flexible


_JSON_SCHEMA_BLOCK = """\
{
  "room": {"label": "<room type>", "confidence": <0.0-1.0>},
  "objects": [
    {
      "label": "<object name>",
      "bbox": [x1, y1, x2, y2],
      "bbox_confidence": "<high|medium|low>",
      "visible": <true|false>,
      "visibility": "<visible|partially_visible|not_visible|unknown>",
      "bearing": "<left|center|right|left_front|right_front|front|unknown>",
      "range": "<close|very_near|near|mid|far|unknown>",
      "relation": ["<one short spatial phrase>"],
      "room_context": "<room or unknown>",
      "attributes": {"description": "<2-5 visual words>"},
      "confidence": <0.0-1.0>
    }
  ],
  "goal_visible": <true|false>,
  "goal_match_confidence": <0.0-1.0>,
  "target_direction": "<left|left_front|center|right_front|right|unknown>",
  "target_visibility": "<clear|partial|heavily_occluded|uncertain|not_visible>",
  "apparent_scale": "<tiny|small|medium|large|very_large|unknown>",
  "relative_progress": "<closer|unchanged|farther|uncertain>",
  "stop_candidate": <true|false>,
  "recommended_action": "<turn_left|turn_right|move_forward|hold_and_verify|search|stop_candidate>",
  "goal_reason": "<why you think the goal is visible or not>",
  "portals": [
    {
      "type": "<doorway|door|opening|passage|stairs|unknown>",
      "state": "<open|closed|uncertain>",
      "bearing": "<left|center|right|left_front|right_front|front|unknown>",
      "traversable": <true|false>
    }
  ],
  "scene_summary": "<one sentence describing the scene>",
  "uncertainty": <0.0-1.0>
}"""

_COMMON_RULES = """\
Rules:
- Every visible object you report MUST include a normalised bounding box [x1, y1, x2, y2] with values in [0, 1] (top-left origin). Never use null for a visible object's bbox. If the bounds are uncertain, still provide your best approximate bbox and set bbox_confidence=\"low\". If you truly cannot localize a visible object, omit it from objects[] and mention it only in scene_summary.
- Keep each object compact. For every object, include at most one attributes.description phrase with 2-5 visual words, not a sentence. Do not include color/shape/material/size/state as separate fields.
- Do not estimate metric distance. Use only close, very_near, near, mid, far, or unknown.
- Use at most one short relation phrase such as "beside cabinet" or "through doorway".
- room.label should be a concise room or area type such as kitchen, bedroom, \
utility room, entryway, pantry, hallway, closet, staircase, or unknown. Use a \
more specific type when the visual evidence supports it.
- Keep scene_summary under 30 words.
- range is advisory visual context only. It is not a final stop decision.
- stop_candidate is visual evidence for a controller, not a STOP command.
- If no previous reference image is provided, relative_progress must be uncertain.
- Do NOT include any text outside the JSON object."""

EXPLORE_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in EXPLORATION mode. \
Your job is high-recall open-vocabulary scene understanding for mapping. \
Analyse the egocentric RGB image and answer strictly in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
- Report up to 5 most visually salient and navigation-useful objects when present. \
Include fewer when only a few objects are clearly visible and useful for navigation or memory.
- Prioritize in this order: current goal or possible aliases; stable furniture \
and fixtures; distinctive landmarks useful for localization; doors, openings, \
passages, and large obstacles.
- It is okay to report ordinary scene objects such as doors, windows, pictures, \
switches, cabinets, chairs, shelves, appliances, plants, rugs, or decorations \
when they are visually clear and useful for room identity, navigation, or memory.
- Ignore only tiny, ambiguous, severely occluded, repetitive, or low-value clutter.
- Report portals as structured objects with type, state, bearing, and traversable. \
Include open, closed, or uncertain doors/openings/passages only when spatially meaningful.
- goal_visible should be true only when there is direct visual evidence matching \
the goal name, aliases, description, or visible attributes. Do not infer the \
target only from room type, context, or prior expectation.
- If the navigation goal or a likely alias is visible, include it first in \
objects[] with the most specific visible label and a compact description."""

CONFIRM_SYSTEM_PROMPT = f"""\
You are a navigation perception agent inside a house in CONFIRMATION mode. \
The robot is checking whether the target is visible from the current view. Your job is to \
verify whether the navigation goal is visible and suitable for local control. \
Analyse the egocentric RGB image and answer in JSON.

Your response MUST be a valid JSON object with exactly these keys:
{_JSON_SCHEMA_BLOCK}

{_COMMON_RULES}
- Set goal_visible=true only when there is direct visual evidence matching the \
target name, aliases, description, or visible attributes. Set goal_visible=false \
when the evidence is only room type, context, or prior expectation.
- Report at most 1 object: the target or best matching alias. If absent, use \
objects=[].
- For the target/alias object, include a 2-5 word attributes.description if it \
helps confirm identity.
- If the target is only seen through a doorway, behind a doorway, heavily \
off-center, or partially visible, set stop_candidate=false and recommend \
move_forward or turn toward it.
- bbox is used by the downstream controller as one piece of stop-verification evidence.  Do not enlarge or fabricate bbox to justify stopping.  Set stop_candidate=true ONLY when the target is clearly identifiable, \
centered, and sufficiently close.  When uncertain, set stop_candidate to false.
- Set target_visibility to "clear" if the target is at least partially \
unobstructed.
- Set target_direction to "center" if the target is roughly \
centered, even if slightly off-center.
- Pay special attention to range and bearing: the robot needs to know whether \
to turn, move forward, hold, or search.
- Compare the current image with the labelled previous reference image when \
one is provided. Judge whether the target appears closer, unchanged, or farther.
- If the target is not ready as a stop candidate, recommend one local action: \
turn_left, turn_right, move_forward, hold_and_verify, or search."""

SYSTEM_PROMPT = EXPLORE_SYSTEM_PROMPT


def build_user_prompt(
    goal_text: str,
    context: Optional[str] = None,
    mode: str = "explore",
    has_previous_image: bool = False,
    action_history: Optional[List[str]] = None,
) -> str:
    """Construct the user-turn text sent alongside the image."""
    parts = [f"Current navigation goal: {goal_text}"]
    if context:
        parts.append(f"Context: {context}")
    if action_history:
        parts.append("Recent robot actions: " + " -> ".join(action_history[-5:]))
    if mode == "confirm":
        comparison = (
            "A PREVIOUS reference image is provided before the CURRENT image. "
            "Use it only for relative_progress."
            if has_previous_image
            else
            "No previous reference image is available. "
            "Set relative_progress to uncertain."
        )
        parts.append(
            "The robot is checking the current view for the target. "
            "Focus only on the goal and target_aliases. Return objects[] with "
            "only the target or best matching alias if visible. Include bbox, "
            "bearing, range, visibility, stop_candidate, and one local action. "
            "Do not list unrelated objects. Say whether the target is a visual stop candidate "
            "and which local action is "
            f"appropriate. {comparison} Return the JSON perception report."
        )
    else:
        parts.append(
        "Analyse the image and return the JSON perception report. "
        "Report up to 5 visually salient, navigation-useful objects when present; "
        "include fewer when the view has fewer useful objects. "
        "Prioritize the goal and target_aliases, then stable furniture, fixtures, "
        "distinctive landmarks, and meaningful doors/openings/passages. "
        "Use open-vocabulary labels. Do not infer goal_visible from context alone. "
        "For portals, return structured type/state/bearing/traversable entries."
    )
    return "\n".join(parts)


def get_system_prompt(mode: str = "explore") -> str:
    """Return the appropriate system prompt for the given perception mode."""
    if mode == "confirm":
        return CONFIRM_SYSTEM_PROMPT
    return EXPLORE_SYSTEM_PROMPT


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)


def parse_vlm_json(raw: str) -> Dict[str, Any]:
    """Best-effort parse of VLM output into the canonical schema dict.

    Tries, in order:
    1. Direct ``json.loads`` on the raw string.
    2. Extract JSON from a fenced code block.
    3. Extract the first ``{...}`` substring.

    Falls back to a safe empty dict on failure.
    """
    text = raw.strip()

    for candidate in _candidates(text):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                parsed = _normalise(data)
                parsed["_parse_ok"] = True
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return _fallback()


def _candidates(text: str):
    yield text
    m = _JSON_BLOCK_RE.search(text)
    if m:
        yield m.group(1)
    m = _BARE_JSON_RE.search(text)
    if m:
        yield m.group(1)


def _normalise(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist with correct types."""
    room = data.get("room") or {}
    if not isinstance(room, dict):
        room = {}
    objects_raw = data.get("objects") or []
    if not isinstance(objects_raw, list):
        objects_raw = []
    objects: List[Dict[str, Any]] = []
    for o in objects_raw:
        if isinstance(o, dict) and "label" in o:
            relation = o.get("relation", o.get("spatial_relation", []))
            if isinstance(relation, str):
                relation = [relation]
            if not isinstance(relation, list):
                relation = []
            raw_attrs = o.get("attributes") or {}
            attributes = {}
            if isinstance(raw_attrs, dict):
                for _ak in ("color", "shape", "material", "size", "state", "description"):
                    _av = raw_attrs.get(_ak)
                    if isinstance(_av, dict) and "value" in _av:
                        attributes[_ak] = {
                            "value": str(_av.get("value", "")),
                            "confidence": _float01(_av.get("confidence", 0.5), 0.0),
                        }
                    elif isinstance(_av, str) and _av.strip():
                        attributes[_ak] = {"value": _av.strip(), "confidence": 0.5}
            raw_bbox = (
                o.get("bbox")
                or o.get("box")
                or o.get("bounding_box")
                or o.get("bounds")
                or o.get("coordinates")
                or o.get("bbox_2d")
            )
            objects.append({
                "label": str(o.get("label", "")),
                "bbox": normalize_bbox_flexible(raw_bbox),
                "visible": bool(o.get("visible", True)),
                "visibility": _enum(
                    o.get("visibility", "visible" if o.get("visible", True) else "not_visible"),
                    {"visible", "partially_visible", "not_visible", "unknown"},
                    "unknown",
                ),
                "bearing": _enum(
                    o.get("bearing", "unknown"),
                    {"left", "center", "right", "left_front", "right_front", "front", "unknown"},
                    "unknown",
                ),
                "range": _enum(
                    o.get("range", o.get("range_bin", "unknown")),
                    {"close", "very_near", "near", "mid", "far", "unknown"},
                    "unknown",
                ),
                "relation": [str(x) for x in relation],
                "room_context": str(o.get("room_context", "")) or None,
                "bbox_confidence": _enum(
                    o.get("bbox_confidence", "medium"),
                    {"high", "medium", "low"},
                    "medium",
                ),
                "attributes": attributes,
                "confidence": _float01(o.get("confidence", 0.0), 0.0),
            })
    return {
        "room": {
            "label": str(room.get("label", "unknown")),
            "confidence": _float01(room.get("confidence", 0.0), 0.0),
        },
        "objects": objects,
        "goal_visible": bool(data.get("goal_visible", False)),
        "goal_match_confidence": _float01(
            data.get("goal_match_confidence", 0.0), 0.0,
        ),
        "target_direction": _target_direction(data.get("target_direction", "unknown")),
        "target_visibility": _enum(
            data.get("target_visibility", "not_visible"),
            {"clear", "partial", "heavily_occluded", "uncertain", "not_visible"},
            "uncertain",
        ),
        "apparent_scale": _enum(
            data.get("apparent_scale", "unknown"),
            {"tiny", "small", "medium", "large", "very_large", "unknown"},
            "unknown",
        ),
        "relative_progress": _enum(
            data.get("relative_progress", "uncertain"),
            {"closer", "unchanged", "farther", "uncertain"},
            "uncertain",
        ),
        "stop_candidate": bool(data.get("stop_candidate", False)),
        "recommended_action": _enum(
            data.get("recommended_action", "search"),
            {
                "turn_left", "turn_right", "move_forward",
                "hold_and_verify", "search", "stop_candidate",
            },
            "search",
        ),
        "goal_reason": str(data.get("goal_reason", "")),
        "portals": _normalise_portals(data.get("portals") or []),
        "scene_summary": str(data.get("scene_summary", "")),
        "uncertainty": float(data.get("uncertainty", 0.5)),
    }


def _fallback() -> Dict[str, Any]:
    return {
        "room": {"label": "unknown", "confidence": 0.0},
        "objects": [],
        "goal_visible": False,
        "goal_match_confidence": 0.0,
        "target_direction": "unknown",
        "target_visibility": "not_visible",
        "apparent_scale": "unknown",
        "relative_progress": "uncertain",
        "stop_candidate": False,
        "recommended_action": "search",
        "goal_reason": "parse_error",
        "portals": [],
        "scene_summary": "",
        "uncertainty": 1.0,
        "_parse_ok": False,
    }


def _target_direction(value: Any) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    if text in {"front", "forward"}:
        return "center"
    return _enum(text, {"left", "left_front", "center", "right_front", "right", "unknown"}, "unknown")


def _normalise_portals(raw_portals: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_portals, list):
        return []
    portals: List[Dict[str, Any]] = []
    for item in raw_portals:
        if isinstance(item, dict):
            label_text = " ".join(str(v) for v in item.values()).lower()
            portal_type = _enum(
                item.get("type", item.get("label", "unknown")),
                {"doorway", "door", "opening", "passage", "stairs", "unknown"},
                "unknown",
            )
            state = _enum(item.get("state", "uncertain"), {"open", "closed", "uncertain"}, "uncertain")
            bearing = _enum(
                item.get("bearing", "unknown"),
                {"left", "center", "right", "left_front", "right_front", "front", "unknown"},
                "unknown",
            )
            traversable = bool(item.get("traversable", state == "open"))
        else:
            label_text = str(item or "").lower()
            portal_type = "unknown"
            for candidate in ("doorway", "opening", "passage", "stairs", "door"):
                if candidate in label_text:
                    portal_type = candidate
                    break
            if "closed" in label_text:
                state = "closed"
            elif "open" in label_text or "passage" in label_text or "opening" in label_text:
                state = "open"
            else:
                state = "uncertain"
            bearing = "unknown"
            for candidate in ("left_front", "right_front", "left", "right", "front", "center"):
                if candidate.replace("_", " ") in label_text or candidate in label_text:
                    bearing = candidate
                    break
            traversable = bool(state == "open")
        portals.append({
            "type": portal_type,
            "state": state,
            "bearing": bearing,
            "traversable": bool(traversable),
            "label": _portal_label(portal_type, state, bearing),
        })
    return portals


def _portal_label(portal_type: str, state: str, bearing: str) -> str:
    parts = [state if state != "uncertain" else "", portal_type if portal_type != "unknown" else "portal", bearing if bearing != "unknown" else ""]
    return " ".join(part for part in parts if part).strip() or "portal"


def _enum(value: Any, allowed: set, default: str) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return text if text in allowed else default


def _float01(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
